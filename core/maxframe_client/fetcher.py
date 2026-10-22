# Copyright 1999-2026 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import sys
import warnings
from abc import ABC, abstractmethod
from numbers import Integral
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from odps import ODPS
from odps.errors import ODPSError
from odps.models import ExternalVolume

from maxframe import utils
from maxframe.core import OBJECT_TYPE
from maxframe.dataframe.core import DATAFRAME_TYPE
from maxframe.io.objects import get_object_io_handler
from maxframe.io.odpsio import (
    ODPSTableIO,
    ODPSVolumeReader,
    TunnelTableIO,
    arrow_to_pandas,
    build_dataframe_table_meta,
    build_liteframe_table_meta,
    odps_schema_to_pandas_dtypes,
)
from maxframe.io.odpsio.schema import decode_blob_table_comment
from maxframe.lib import wrapped_pickle as pickle
from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.protocol import (
    ConstantResultInfo,
    DataFrameTableMeta,
    LiteFrameTableMeta,
    ODPSTableResultInfo,
    ODPSVolumeResultInfo,
    ResultInfo,
    ResultType,
    TensorTableMeta,
)
from maxframe.tensor.core import TENSOR_TYPE
from maxframe.tensor.utils import slice_split
from maxframe.typing_ import TileableType
from maxframe.utils import (
    ToThreadMixin,
    estimate_pandas_size,
    estimate_table_size,
    get_storage_api_endpoint,
    sync_pyodps_options,
)

try:
    from odps.maxstorage import (
        MaxStorageClient,
        RowRangeInputSplit,
        SplitMode,
        SplitOptions,
    )
except ImportError:
    MaxStorageClient = None
    SplitMode = None
    SplitOptions = None
    RowRangeInputSplit = None

_result_fetchers: Dict[ResultType, Type["ResultFetcher"]] = dict()
_FetchIndexType = Optional[List[Union[None, Integral, slice]]]
_VOLUME_LOAD_RETRY_TIMES = 5
_MAX_BLOB_TABLE_ROWS = 100000
_BLOB_READ_SESSION_READY_TIMEOUT = 300


def register_fetcher(fetcher_cls: Type["ResultFetcher"]):
    _result_fetchers[fetcher_cls.result_type] = fetcher_cls
    return fetcher_cls


def get_fetcher_cls(result_type: ResultType) -> Type["ResultFetcher"]:
    return _result_fetchers[result_type]


class ResultFetcher(ABC):
    result_type = None

    def __init__(self, odps_entry: ODPS):
        self._odps_entry = odps_entry

    @abstractmethod
    async def update_tileable_meta(
        self,
        tileable: TileableType,
        info: ResultInfo,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def fetch(
        self,
        tileable: TileableType,
        info: ResultInfo,
        indexes: _FetchIndexType,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def estimate_size(
        self,
        tileable: TileableType,
        info: ResultInfo,
    ) -> Union[int, float]:
        raise NotImplementedError


@register_fetcher
class NullFetcher(ResultFetcher):
    result_type = ResultType.NULL

    async def update_tileable_meta(
        self,
        tileable: TileableType,
        info: ResultInfo,
    ) -> None:
        return

    async def fetch(
        self,
        tileable: TileableType,
        info: ODPSTableResultInfo,
        indexes: _FetchIndexType,
    ) -> None:
        return

    def estimate_size(
        self,
        tileable: TileableType,
        info: ResultInfo,
    ) -> Union[int, float]:
        return 0


@register_fetcher
class ConstantFetcher(ResultFetcher):
    result_type = ResultType.CONSTANT

    async def update_tileable_meta(
        self,
        tileable: TileableType,
        info: ConstantResultInfo,
    ) -> None:
        if isinstance(tileable, DATAFRAME_TYPE) and tileable.dtypes is None:
            tileable.refresh_from_dtypes(info.data.dtypes)
        if tileable.shape and any(pd.isna(x) for x in tileable.shape):
            tileable.params = {"shape": info.data.shape}

    async def fetch(
        self,
        tileable: TileableType,
        info: ConstantResultInfo,
        indexes: List[Union[None, Integral, slice]],
    ) -> Any:
        result = info.data
        if indexes:
            if isinstance(indexes, List):
                indexes = tuple(indexes)
            if isinstance(result, (pd.DataFrame, pd.Series)):
                result = result.iloc[indexes]
            else:
                result = result[indexes]
        return result

    def estimate_size(
        self,
        tileable: TileableType,
        info: ConstantResultInfo,
    ) -> Union[int, float]:
        if isinstance(info.data, (pd.DataFrame, pd.Series, pd.Index)):
            return estimate_pandas_size(info.data)
        else:
            return sys.getsizeof(info.data)


@register_fetcher
class ODPSTableFetcher(ToThreadMixin, ResultFetcher):
    result_type = ResultType.ODPS_TABLE

    def _get_table_comment(self, table_name: str) -> Optional[str]:
        table = self._odps_entry.get_table(table_name)
        return getattr(table, "comment", None)

    def _get_table_record_num(self, table, part_specs):
        if not part_specs or part_specs == [None]:
            rec_num = getattr(table, "record_num", None)
        else:
            rec_num = 0
            for part_spec in part_specs:
                pt_rec_num = getattr(table.get_partition(part_spec), "record_num", None)
                if pt_rec_num is None or pt_rec_num < 0:
                    rec_num = None
                    break
                else:
                    rec_num += pt_rec_num

        if rec_num is not None and rec_num >= 0:
            return rec_num

        try:
            part_sessions = TunnelTableIO.create_download_sessions(
                self._odps_entry, table.full_table_name, part_specs
            )
            return sum(session.count for session in part_sessions.values())
        except ODPSError as ex:
            if "StatusConflict" not in str(ex):
                raise
            warnings.warn(
                f"Failed to obtain record count of table {table.full_table_name}. "
                f"The original error is:\n{ex}"
            )
            return None

    async def update_tileable_meta(
        self,
        tileable: TileableType,
        info: ODPSTableResultInfo,
    ) -> None:
        if isinstance(tileable, TENSOR_TYPE) and isinstance(
            info.table_meta, TensorTableMeta
        ):
            # Tensor blob table: read metadata from TensorTableMeta
            meta = info.table_meta
            params = {}
            if meta.dtype is not None:
                params["dtype"] = np.dtype(meta.dtype)
            if meta.shape is not None:
                params["shape"] = tuple(meta.shape)
            tileable.params = params
            return

        if isinstance(tileable, OBJECT_TYPE) and info.table_meta is None:
            # Object blob table: no metadata update needed
            return

        if (
            isinstance(tileable, DATAFRAME_TYPE)
            and tileable.dtypes is None
            and info.table_meta is not None
        ):
            if info.table_meta.pd_column_dtypes is not None:
                tileable.refresh_from_table_meta(info.table_meta)
            else:
                # need to get meta directly from table
                table = self._odps_entry.get_table(info.full_table_name)
                pd_dtypes = odps_schema_to_pandas_dtypes(table.table_schema).drop(
                    info.table_meta.table_index_column_names
                )
                tileable.refresh_from_dtypes(pd_dtypes)

        if tileable.shape and any(pd.isna(x) for x in tileable.shape):
            part_specs = [None] if not info.partition_specs else info.partition_specs

            with sync_pyodps_options():
                table = self._odps_entry.get_table(info.full_table_name)
                if isinstance(tileable, DATAFRAME_TYPE) and tileable.dtypes is None:
                    dtypes = odps_schema_to_pandas_dtypes(table.table_schema)
                    tileable.refresh_from_dtypes(dtypes)

                total_records = self._get_table_record_num(table, part_specs)
                total_records = np.nan if total_records is None else total_records

            new_shape_list = list(tileable.shape)
            new_shape_list[0] = total_records
            tileable.params = {"shape": tuple(new_shape_list)}

    @staticmethod
    def _align_selection_with_shape(
        row_sel: slice, shape: Tuple[Optional[int], ...]
    ) -> dict:
        size = shape[0]
        if not row_sel.start and not row_sel.stop:
            return {}
        is_reversed = row_sel.step is not None and row_sel.step < 0
        read_kw = {
            "start": row_sel.start,
            "stop": row_sel.stop,
            "reverse_range": is_reversed,
        }
        if pd.isna(size):
            return read_kw

        if is_reversed and row_sel.start is not None:
            read_kw["start"] = min(size - 1, row_sel.start)
        if not is_reversed and row_sel.stop is not None:
            read_kw["stop"] = min(size, row_sel.stop)
        return read_kw

    def _read_single_source(
        self,
        table_meta: Union[DataFrameTableMeta, LiteFrameTableMeta],
        info: ODPSTableResultInfo,
        indexes: List[Union[None, Integral, slice]],
        shape: Tuple[Optional[int], ...],
        tileable: Optional[TileableType] = None,
    ):
        table_io = ODPSTableIO(self._odps_entry)
        read_kw = {}
        row_step = None
        if indexes:
            if len(indexes) < 2:
                indexes += [None]
            row_sel, col_sel = indexes
            if isinstance(row_sel, slice):
                row_step = row_sel.step
                read_kw = self._align_selection_with_shape(row_sel, shape)
            elif isinstance(row_sel, int):
                read_kw["start"] = row_sel
                read_kw["stop"] = row_sel + 1
                row_step = None
            elif row_sel is not None:  # pragma: no cover
                raise NotImplementedError(f"Does not support row index {row_sel!r}")

            if col_sel is not None and not isinstance(col_sel, (int, slice)):
                raise NotImplementedError(f"Does not support column index {col_sel!r}")

            # For LiteFrame, use tileable's columns directly (no pandas-style index columns)
            if isinstance(table_meta, LiteFrameTableMeta):
                if isinstance(col_sel, (int, slice)):
                    if tileable is not None and hasattr(tileable, "columns"):
                        columns = list(tileable.columns)
                        data_cols = columns[col_sel]
                        if isinstance(col_sel, int):
                            data_cols = [data_cols]
                        # The temp ODPS table may store sanitized physical
                        # column names (e.g. `_column_0` for logical "0");
                        # the tunnel read must select the physical names.
                        read_kw["columns"] = self._liteframe_physical_columns(
                            table_meta, data_cols
                        )
                    # If tileable not provided, read all columns (will use dtypes from table)
            else:
                # For DataFrame, use table_meta's column structure
                if isinstance(col_sel, (int, slice)):
                    data_cols = table_meta.table_column_names[col_sel]
                    if isinstance(col_sel, int):
                        data_cols = [data_cols]
                    read_kw["columns"] = table_meta.table_index_column_names + data_cols

        with table_io.open_reader(
            info.full_table_name, info.partition_specs, **read_kw
        ) as reader:
            result = reader.read_all()
            reader_count = result.num_rows

        if not row_step:
            return result

        if row_step >= 0:
            slice_start = 0
            slice_stop = reader_count
        else:
            slice_start = reader_count - 1
            slice_stop = None
        return result[slice_start:slice_stop:row_step]

    @staticmethod
    def _deserialize_chunk(blob_bytes: bytes, blob_format: str) -> np.ndarray:
        """Deserialize a chunk from blob bytes based on format."""
        buf = io.BytesIO(blob_bytes)
        if blob_format == "npz":
            npz_file = np.load(buf, allow_pickle=False)
            return npz_file["data"]
        else:
            return pickle.load(buf)

    @staticmethod
    def _assemble_tensor_chunks(chunks: dict, nsplits: tuple) -> np.ndarray:
        """Assemble chunk arrays into a single numpy array.

        chunks: dict mapping chunk_index_tuple -> np.ndarray
        nsplits: tuple of per-dimension split sizes
        """
        ndim = len(nsplits)
        if ndim == 0:
            # Scalar tensor: should have exactly one chunk with key ()
            return chunks[()]

        def _assemble_along_dim(dim: int, prefix: tuple) -> np.ndarray:
            if dim == ndim - 1:
                # Last dimension: concatenate along axis=dim
                arrays = [chunks[prefix + (i,)] for i in range(len(nsplits[dim]))]
                return np.concatenate(arrays, axis=dim)
            else:
                # Recurse along the next dimension, then concatenate along axis=dim
                sub_arrays = [
                    _assemble_along_dim(dim + 1, prefix + (i,))
                    for i in range(len(nsplits[dim]))
                ]
                return np.concatenate(sub_arrays, axis=dim)

        return _assemble_along_dim(0, ())

    @staticmethod
    def _create_blob_read_session(client, table, split_number: int):
        """Create a read session with ROW_OFFSET split mode (required for
        append 2.0 tables).

        ``MaxStorageClient.create_table_read_session`` polls until the
        session is ``NORMAL`` before returning.
        """
        split_options = SplitOptions(
            split_mode=SplitMode.ROW_OFFSET,
            split_number=split_number,
        )
        return client.create_table_read_session(
            table,
            split_options=split_options,
            session_ready_timeout=_BLOB_READ_SESSION_READY_TIMEOUT,
        )

    @staticmethod
    def _read_arrow_table(reader, max_rows: int = None) -> pa.Table:
        """Read all batches from an ArrowReader into a pa.Table.

        If *max_rows* is given, raises ValueError when the accumulated row
        count exceeds the limit.
        """
        batches = []
        total_rows = 0
        while True:
            batch = reader.read()
            if batch is None:
                break
            batches.append(batch)
            total_rows += batch.num_rows
            if max_rows is not None and total_rows > max_rows:
                raise ValueError(
                    f"Number of rows ({total_rows}) exceeds the limit ({max_rows}). "
                    f"Consider reducing the number of chunks."
                )
        return pa.Table.from_batches(batches) if batches else pa.table({})

    @sync_pyodps_options
    def _fetch_blob_table_tensor(
        self, info: ODPSTableResultInfo, indexes: _FetchIndexType = None
    ) -> np.ndarray:
        """Fetch tensor data from a blob table using odps.maxstorage.

        Supports slice and integer indexes to fetch only the needed portions
        of the tensor.  When *indexes* is None or all-None the full tensor
        is returned (backward-compatible).
        """
        if MaxStorageClient is None:
            raise NotImplementedError(
                "Tensor blob table fetch requires odps.maxstorage. "
                "You need to install PyODPS >= 0.13.1 to enable this."
            )
        table = self._odps_entry.get_table(info.full_table_name)
        client = MaxStorageClient(
            self._odps_entry, tunnel_endpoint=get_storage_api_endpoint()
        )

        read_session = self._create_blob_read_session(
            client, table, split_number=_MAX_BLOB_TABLE_ROWS + 1
        )

        # Read all rows to get chunk indices, format, and blob references
        reader = read_session.open_arrow_reader(
            RowRangeInputSplit(read_session.id, 0, _MAX_BLOB_TABLE_ROWS + 1)
        )
        all_data = self._read_arrow_table(reader, max_rows=_MAX_BLOB_TABLE_ROWS)
        num_rows = all_data.num_rows
        if num_rows == 0:
            raise ValueError("Blob table is empty, no tensor chunks found.")

        # Extract ndim from column names (idx_0, idx_1, ..., format, content)
        ndim = len([c for c in all_data.column_names if c.startswith("idx_")])

        # Use table_meta nsplits if available, otherwise decode from table comment
        if info.table_meta and info.table_meta.nsplits:
            nsplits = tuple(tuple(sp) for sp in info.table_meta.nsplits)
        else:
            tensor_meta = (
                decode_blob_table_comment(table.comment) if table.comment else {}
            )
            nsplits = tuple(tuple(sp) for sp in tensor_meta.get("nsplits", []))

        # Collect blob references, format, and chunk indices
        chunk_refs = []
        for row_idx in range(num_rows):
            chunk_index = tuple(
                int(all_data.column(f"idx_{i}")[row_idx].as_py()) for i in range(ndim)
            )
            blob_format = all_data.column("format")[row_idx].as_py()
            blob_ref = all_data.column("content")[row_idx].as_py()
            chunk_refs.append((chunk_index, blob_format, blob_ref))

        blob_manager = client.open_blob_manager(table)

        # --- Normalize indexes ---
        no_slice = indexes is None or all(i is None for i in indexes)
        if no_slice:
            # Full fetch path (backward-compatible)
            blob_refs = [ref for _, _, ref in chunk_refs]
            blob_iter = blob_manager.read_blobs(blob_refs)
            chunks = {}
            for (chunk_index, blob_format, _), record in zip(chunk_refs, blob_iter):
                chunks[chunk_index] = self._deserialize_chunk(record.data, blob_format)
            return self._assemble_tensor_chunks(chunks, nsplits)

        # Pad indexes to ndim and replace None with slice(None)
        normalized = []
        for d in range(ndim):
            if d < len(indexes):
                idx = indexes[d]
                normalized.append(idx if idx is not None else slice(None))
            else:
                normalized.append(slice(None))

        # --- Determine per-dimension affected chunks via slice_split ---
        dim_splits = []
        for d in range(ndim):
            dim_splits.append(slice_split(normalized[d], list(nsplits[d])))

        # --- Filter chunk refs: keep only chunks whose index appears in all dim_splits ---
        filtered_refs = []
        for chunk_index, blob_format, blob_ref in chunk_refs:
            if all(chunk_index[d] in dim_splits[d] for d in range(ndim)):
                filtered_refs.append((chunk_index, blob_format, blob_ref))

        # --- Fetch only filtered blobs ---
        blob_refs = [ref for _, _, ref in filtered_refs]
        blob_iter = blob_manager.read_blobs(blob_refs)

        # --- Deserialize and sub-slice each chunk ---
        sub_chunks = {}
        for (chunk_index, blob_format, _), record in zip(filtered_refs, blob_iter):
            chunk_data = self._deserialize_chunk(record.data, blob_format)
            # Build per-dim sub-index for this chunk
            sub_idx = []
            for d in range(ndim):
                local = dim_splits[d][chunk_index[d]]
                sub_idx.append(local)
            chunk_data = chunk_data[tuple(sub_idx)]
            sub_chunks[chunk_index] = chunk_data

        # --- Separate dimensions into slice-dims (kept) and int-dims (reduced) ---
        slice_dims = []
        int_dims = []
        for d in range(ndim):
            if isinstance(normalized[d], Integral):
                int_dims.append(d)
            else:
                slice_dims.append(d)

        if not slice_dims:
            # All dimensions are integer indexes -> scalar result
            # There should be exactly one chunk
            result = list(sub_chunks.values())[0]
            return result.item() if hasattr(result, "item") else result

        # Map original chunk index to reduced-dim key for assembly ordering.
        # slice_dims are kept, int_dims are dropped from the key.
        # Chunk indices are reindexed to be contiguous (0-based) so that
        # _assemble_tensor_chunks can iterate from 0 without gaps.
        # Preserve the order from dim_splits (which follows the slice
        # direction) so reverse slices like [::-1] assemble correctly.
        reduced_chunks = {}
        dim_reindex = []
        for d in slice_dims:
            touched = [
                ci
                for ci in dim_splits[d].keys()
                if any(sc[d] == ci for sc in sub_chunks)
            ]
            dim_reindex.append({orig: new for new, orig in enumerate(touched)})

        reduced_nsplits_list = []
        for idx, d in enumerate(slice_dims):
            touched = list(dim_reindex[idx].keys())
            reduced_nsplits_list.append(tuple(nsplits[d][i] for i in touched))

        for chunk_index, chunk_data in sub_chunks.items():
            reduced_key = tuple(
                dim_reindex[idx][chunk_index[d]] for idx, d in enumerate(slice_dims)
            )
            reduced_chunks[reduced_key] = chunk_data

        reduced_nsplits = tuple(reduced_nsplits_list)

        # Assemble along the remaining (slice) dimensions
        result = self._assemble_tensor_chunks(reduced_chunks, reduced_nsplits)

        return result

    @sync_pyodps_options
    def _fetch_blob_table_object(self, info: ODPSTableResultInfo) -> Any:
        """Fetch object data from a blob table using odps.maxstorage."""
        if MaxStorageClient is None:
            raise NotImplementedError(
                "Object blob table fetch requires odps.maxstorage. "
                "You need to install PyODPS >= 0.13.1 to enable this."
            )
        table = self._odps_entry.get_table(info.full_table_name)
        client = MaxStorageClient(
            self._odps_entry, tunnel_endpoint=get_storage_api_endpoint()
        )

        read_session = self._create_blob_read_session(client, table, split_number=2)

        # Read the single row
        reader = read_session.open_arrow_reader(
            RowRangeInputSplit(read_session.id, 0, 1)
        )
        all_data = self._read_arrow_table(reader, max_rows=1)

        if all_data.num_rows != 1:
            raise ValueError(
                f"Object blob table should have exactly 1 row, got {all_data.num_rows}"
            )
        # Get blob reference from the content column
        blob_ref = all_data.column("content")[0].as_py()
        # Read blob content
        blob_manager = client.open_blob_manager(table)
        blob_iter = blob_manager.read_blobs([blob_ref])
        blob_bytes = next(iter(blob_iter)).data

        # Unpickle the object
        buf = io.BytesIO(blob_bytes)
        return pickle.load(buf)

    @staticmethod
    def _liteframe_physical_columns(
        table_meta: LiteFrameTableMeta, logical_columns: List[str]
    ) -> List[str]:
        """Map logical LiteFrame column names to their physical ODPS table names.

        ``LiteFrameTableMeta.table_column_names`` is positionally aligned with
        ``dtypes.index``; when the physical names differ (e.g. ``_column_0``
        for the int-string logical name ``"0"``), the tunnel read must select
        the physical names.  Returns the logical names unchanged when no
        mapping is carried.
        """
        table_column_names = table_meta.table_column_names
        dtypes = table_meta.dtypes
        if table_column_names and dtypes is not None:
            index_by_logical = {logical: i for i, logical in enumerate(dtypes.index)}
            physical = []
            for col in logical_columns:
                idx = index_by_logical.get(col)
                physical.append(
                    table_column_names[idx]
                    if idx is not None and idx < len(table_column_names)
                    else col
                )
            return physical
        return logical_columns

    async def fetch(
        self,
        tileable: TileableType,
        info: ODPSTableResultInfo,
        indexes: _FetchIndexType,
    ) -> Any:
        # Handle tensor/object blob tables
        if isinstance(tileable, TENSOR_TYPE) and isinstance(
            info.table_meta, TensorTableMeta
        ):
            return await self.to_thread(self._fetch_blob_table_tensor, info, indexes)
        if isinstance(tileable, OBJECT_TYPE) and info.table_meta is None:
            return await self.to_thread(self._fetch_blob_table_object, info)

        # Use build_liteframe_table_meta for LiteFrame objects
        if isinstance(tileable, LITEFRAME_TYPE):
            table_meta = build_liteframe_table_meta(tileable)
        else:
            table_meta = build_dataframe_table_meta(tileable)
        arrow_table: pa.Table = await self.to_thread(
            self._read_single_source,
            table_meta,
            info,
            indexes,
            tileable.shape,
            tileable,
        )
        # For LiteFrame, convert arrow table to pandas DataFrame
        # The pandas DataFrame will be converted to polars by LiteFrame logic
        if isinstance(table_meta, LiteFrameTableMeta):
            pdf = arrow_table.to_pandas()
            # Restore user-facing (logical) column names when the temp table
            # stored sanitized physical names (e.g. `_column_0` -> "0").
            table_column_names = table_meta.table_column_names
            if table_column_names and table_meta.dtypes is not None:
                rename_map = {
                    phys: logical
                    for phys, logical in zip(
                        table_column_names, table_meta.dtypes.index
                    )
                    if phys != logical
                }
                if rename_map:
                    pdf = pdf.rename(columns=rename_map)
            return pdf
        return arrow_to_pandas(arrow_table, table_meta)

    def estimate_size(
        self,
        tileable: TileableType,
        info: ODPSTableResultInfo,
    ) -> Union[int, float]:
        return estimate_table_size(
            self._odps_entry, info.full_table_name, info.partition_specs
        )


@register_fetcher
class ODPSVolumeFetcher(ToThreadMixin, ResultFetcher):
    result_type = ResultType.ODPS_VOLUME

    async def update_tileable_meta(
        self,
        tileable: TileableType,
        info: ODPSVolumeResultInfo,
    ) -> None:
        def volume_fetch_func():
            reader = ODPSVolumeReader(
                self._odps_entry,
                info.volume_name,
                info.volume_path,
                replace_internal_host=True,
            )
            io_handler = get_object_io_handler(tileable)()
            return utils.call_with_retry(
                io_handler.read_object_meta,
                reader,
                tileable,
                retry_timeout=_VOLUME_LOAD_RETRY_TIMES,
                delay=2,
            )

        volume = await self.to_thread(self._odps_entry.get_volume, info.volume_name)
        if isinstance(volume, ExternalVolume):
            meta = await self.to_thread(volume_fetch_func)
            meta.pop("nsplits", None)
            tileable.params = meta
        else:
            raise NotImplementedError(f"Volume type {type(volume)} not supported")

    async def _fetch_object(
        self,
        tileable: TileableType,
        info: ODPSVolumeResultInfo,
        indexes: List[Union[Integral, slice]],
    ) -> Any:
        def volume_fetch_func():
            reader = ODPSVolumeReader(
                self._odps_entry,
                info.volume_name,
                info.volume_path,
                replace_internal_host=True,
            )
            io_handler = get_object_io_handler(tileable)()
            return io_handler.read_object(reader, tileable, indexes)

        volume = await self.to_thread(self._odps_entry.get_volume, info.volume_name)
        if isinstance(volume, ExternalVolume):
            return await self.to_thread(volume_fetch_func)
        else:
            raise NotImplementedError(f"Volume type {type(volume)} not supported")

    async def fetch(
        self,
        tileable: TileableType,
        info: ODPSVolumeResultInfo,
        indexes: _FetchIndexType,
    ) -> Any:
        if isinstance(tileable, (OBJECT_TYPE, TENSOR_TYPE)):
            return await self._fetch_object(tileable, info, indexes)
        raise NotImplementedError(f"Fetching {type(tileable)} not implemented")

    def estimate_size(
        self,
        tileable: TileableType,
        info: ResultInfo,
    ) -> Union[int, float]:
        # todo estimate size of volume
        return float("inf")
