# Copyright 1999-2025 Alibaba Group Holding Ltd.
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
    odps_schema_to_pandas_dtypes,
)
from maxframe.protocol import (
    ConstantResultInfo,
    DataFrameTableMeta,
    ODPSTableResultInfo,
    ODPSVolumeResultInfo,
    ResultInfo,
    ResultType,
)
from maxframe.tensor.core import TENSOR_TYPE
from maxframe.typing_ import PandasObjectTypes, TileableType
from maxframe.utils import (
    ToThreadMixin,
    estimate_pandas_size,
    estimate_table_size,
    sync_pyodps_options,
)

_result_fetchers: Dict[ResultType, Type["ResultFetcher"]] = dict()

_FetchIndexType = Optional[List[Union[None, Integral, slice]]]

_VOLUME_LOAD_RETRY_TIMES = 5


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
        table_meta: DataFrameTableMeta,
        info: ODPSTableResultInfo,
        indexes: List[Union[None, Integral, slice]],
        shape: Tuple[Optional[int], ...],
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

            if isinstance(col_sel, (int, slice)):
                data_cols = table_meta.table_column_names[col_sel]
                if isinstance(col_sel, int):
                    data_cols = [data_cols]
                read_kw["columns"] = table_meta.table_index_column_names + data_cols
            elif col_sel is not None:  # pragma: no cover
                raise NotImplementedError(f"Does not support column index {row_sel!r}")

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

    async def fetch(
        self,
        tileable: TileableType,
        info: ODPSTableResultInfo,
        indexes: _FetchIndexType,
    ) -> PandasObjectTypes:
        table_meta = build_dataframe_table_meta(tileable)
        arrow_table: pa.Table = await self.to_thread(
            self._read_single_source, table_meta, info, indexes, tileable.shape
        )
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
