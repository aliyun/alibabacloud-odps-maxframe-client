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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from odps import ODPS
from odps.types import Column, OdpsSchema, PartitionSpec

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.codegen.spe.liteframe.datasource import _infer_compression
from maxframe.config import options
from maxframe.io.odpsio import ODPSTableIO, pandas_to_arrow, pandas_to_odps_schema
from maxframe.lib.filesystem import get_fs, open_file, to_arrow_file_system
from maxframe.liteframe.operators.core import LiteFrameDataFormat
from maxframe.liteframe.operators.datastore import (
    LiteFrameLakeDataStore,
    LiteFrameToLance,
    LiteFrameToODPSTable,
)
from maxframe.protocol import ResultInfo
from maxframe.utils import TypeDispatcher

try:
    import lance
except (ImportError, AttributeError):  # pragma: no cover
    # lance not installed or not compatible
    lance = None


@register_op_adapter(LiteFrameToODPSTable)
class LiteFrameToODPSTableAdapter(SPEOperatorAdapter):
    @classmethod
    def _write_as_table(
        cls,
        df,
        table_name: str,
        partition_spec: Optional[str] = None,
        partition_columns: Optional[List[str]] = None,
        dtypes=None,
        mode: str = "overwrite",
        unknown_as_string: Optional[bool] = True,
        lifecycle: Optional[int] = None,
        table_properties: Optional[Dict] = None,
        primary_key: Optional[List[str]] = None,
    ):
        partition_spec = (
            PartitionSpec(partition_spec) if partition_spec is not None else None
        )
        odps_schema, odps_meta = pandas_to_odps_schema(
            df, unknown_as_string=unknown_as_string
        )

        exclude_cols = []
        result_schema = cls._get_result_schema(
            partition_spec, partition_columns, odps_schema, exclude_cols
        )

        odps_entry = ODPS.from_environments() or ODPS.from_global()
        if not odps_entry.exist_table(table_name):
            odps_entry.create_table(
                table_name,
                result_schema,
                lifecycle=lifecycle,
                table_properties=table_properties,
                primary_key=primary_key,
            )
            part_cols = [col.name for col in result_schema.partitions]
        else:
            table = odps_entry.get_table(table_name)
            cls._check_table_schema(table.table_schema, result_schema)
            result_schema = table.table_schema
            part_cols = [col.name for col in table.table_schema.partitions]

        part_and_group = []
        if partition_columns is not None:
            groupby_obj = df.groupby(partition_columns)
            for group in groupby_obj.groups:
                group = (group,) if len(partition_columns) == 1 else group
                try:
                    group_data = groupby_obj.get_group(group)
                except KeyError:
                    if len(group) != 1:
                        raise
                    group_data = groupby_obj.get_group(group[0])

                group_data = group_data.loc[
                    :, ~group_data.columns.isin(partition_columns)
                ]

                part_dict = dict(zip(partition_columns, group))
                part_dict.update(dict(partition_spec.items() if partition_spec else {}))
                part_spec = ",".join(f"{col}={part_dict[col]}" for col in part_cols)
                part_and_group.append((part_spec, group_data))
        else:
            part_and_group.append((partition_spec, df))

        overwrite = mode == "overwrite"
        for part, data in part_and_group:
            if part is not None:
                odps_entry.get_table(table_name).create_partition(
                    part, if_not_exists=True
                )

            with ODPSTableIO(odps_entry).open_writer(
                table_name, part, overwrite=overwrite
            ) as writer:
                arrow_val, _ = pandas_to_arrow(data)
                data_array_dict = dict(zip(arrow_val.column_names, arrow_val.columns))
                cols, arrays = [], []
                for col in result_schema.simple_columns:
                    cols.append(col.name)
                    arrays.append(data_array_dict[col.name])
                writer.write(pa.Table.from_arrays(arrays, cols))

        return None

    @classmethod
    def _get_result_schema(
        cls,
        partition_spec: Optional[str],
        partition_columns: Optional[List[str]],
        src_schema: OdpsSchema,
        exclude_cols: List[str],
    ) -> OdpsSchema:
        part_spec = (
            PartitionSpec(partition_spec) if partition_spec is not None else None
        )
        exclude_col_set = set(exclude_cols or ())

        parts = []
        if part_spec is not None:
            parts.extend(Column(k, "string") for k in part_spec.keys())
        if partition_columns is not None:
            part_column_set = set(partition_columns)
            cols = [
                col
                for col in src_schema.columns
                if col.name not in part_column_set and col.name not in exclude_col_set
            ]
            parts.extend(
                col for col in src_schema.columns if col.name in part_column_set
            )
        else:
            cols = [
                col for col in src_schema.columns if col.name not in exclude_col_set
            ]
        return OdpsSchema(cols, parts)

    @classmethod
    def _check_table_schema(cls, table_schema: OdpsSchema, data_schema: OdpsSchema):
        table_cols = [col.name for col in table_schema.simple_columns]
        data_cols = [col.name for col in data_schema.simple_columns]
        if set(table_cols) != set(data_cols):
            raise ValueError(
                f"Source and dest columns does not match: "
                f"{sorted(data_cols)} does not equal to {sorted(table_cols)}"
            )
        table_part_cols = [col.name for col in table_schema.partitions]
        data_part_cols = [col.name for col in data_schema.partitions]
        if set(table_part_cols) != set(data_part_cols):
            raise ValueError(
                f"Source and dest partitions does not match: "
                f"{sorted(data_part_cols)} does not equal to {sorted(table_part_cols)}"
            )

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        context.put_tileable_result_info(op.outputs[0], ResultInfo())
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        res_var = context.get_output_tileable_variable(op.outputs[0])

        args = self._translate_call_args(
            context,
            op.table_name,
            op.partition_spec,
            op.partition_columns,
            op.dtypes,
            mode=op.mode,
            unknown_as_string=op.unknown_as_string,
            lifecycle=op.lifecycle or options.session.table_lifecycle,
            table_properties=op.table_properties,
            primary_key=op.primary_key,
        )
        return [
            f"{cls_name}._write_as_table({input_var}, {', '.join(args)})",
            f"{res_var} = None",
        ]


def _resolve_write_path(path: str, chunk_index: int = 0) -> str:
    """Resolve a wildcard write path for a single (non-chunked) write.

    The LiteFrame ``to_*`` APIs require a wildcard ``*`` (or partition_cols)
    when writing multiple files.  In SPE there is no chunking, so the wildcard
    is replaced with ``chunk_index`` (0).  Paths without a wildcard are
    returned unchanged (single-file targets).
    """
    if "*" in path:
        return path.replace("*", str(chunk_index))
    return path


def _write_partitioned(
    df, base_path, partition_cols, write_fn, storage_options, ext: str = ""
):
    """Write *df* partitioned by *partition_cols* into subdirectories.

    Groups the DataFrame by *partition_cols* and writes each group to
    ``base_path/col1=val1/col2=val2/data_0{ext}`` via *write_fn*. Mirrors the
    DPE lake datastore partition writer, which emits ``data_{chunk}{ext}``.
    """
    base_path = base_path.rstrip("/")
    fs = get_fs(base_path, storage_options)
    for key, group_df in df.groupby(partition_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        part_keys = "/".join(f"{col}={val}" for col, val in zip(partition_cols, key))
        part_path = f"{base_path}/{part_keys}"
        fs.mkdir(part_path, create_parents=True)
        file_path = f"{part_path}/data_0{ext}"
        write_fn(group_df, file_path)


_spe_data_store_handlers = TypeDispatcher()


def register_spe_data_store_handler(fmt: LiteFrameDataFormat):
    def wrapper(cls):
        _spe_data_store_handlers.register(fmt, cls)
        return cls

    return wrapper


class SPEDataStoreHandler(ABC):
    """Format-specific strategy for SPE data store code generation."""

    @classmethod
    @abstractmethod
    def write_as_files(cls, df, path, **kwargs) -> None:
        pass

    @classmethod
    def generate_code(cls, adapter, op, context: SPECodeContext) -> List[str]:
        context.put_tileable_result_info(op.outputs[0], ResultInfo())
        cls_name = cls.__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        res_var = context.get_output_tileable_variable(op.outputs[0])
        args = adapter._translate_call_args(
            context,
            op.path,
            partition_cols=op.partition_cols,
            compression=op.compression,
            storage_options=op.storage_options,
            write_kwargs=op.write_kwargs,
        )
        return [
            f"{cls_name}.write_as_files({input_var}, {', '.join(args)})",
            f"{res_var} = None",
        ]


@register_spe_data_store_handler(LiteFrameDataFormat.CSV)
class CSVDataStoreHandler(SPEDataStoreHandler):
    """SPE handler for writing a pandas DataFrame to CSV files."""

    _UNSUPPORTED_CSV_KWARGS = {
        "float_format",
        "quoting",
        "doublequote",
        "escapechar",
        "decimal",
    }
    _CSV_DEFAULTS = {
        "float_format": None,
        "quoting": None,
        "doublequote": True,
        "escapechar": None,
        "decimal": ".",
    }

    @staticmethod
    def resolve_write_kwargs(write_kwargs: Optional[dict]) -> dict:
        """Translate LiteFrame to_csv options into pandas ``to_csv`` kwargs."""
        sep = write_kwargs.get("sep", ",")
        header = write_kwargs.get("header", True)
        na_rep = write_kwargs.get("na_rep", "")
        result: dict = {"sep": sep or ","}

        if isinstance(header, list):
            raise NotImplementedError(
                "to_csv with header as a list of column aliases is not "
                "supported for LiteFrame yet."
            )
        result["header"] = True if header is None else bool(header)

        if na_rep is not None:
            result["na_rep"] = na_rep

        for key, value in write_kwargs.items():
            if key in CSVDataStoreHandler._UNSUPPORTED_CSV_KWARGS:
                if (
                    value is not None
                    and value != CSVDataStoreHandler._CSV_DEFAULTS.get(key)
                ):
                    raise NotImplementedError(
                        f"to_csv does not support kwarg '{key}' for LiteFrame."
                    )
                continue
            if key in (
                "sep",
                "header",
                "na_rep",
                "columns",
                "mode",
                "encoding",
                "chunksize",
                "compression",
                "partition_cols",
                "storage_options",
            ):
                continue
            if key == "quotechar":
                result["quotechar"] = value
            elif key == "line_terminator":
                result["lineterminator"] = value
            elif key == "date_format":
                result["date_format"] = value
            elif key == "encoding":
                result["encoding"] = value
        return result

    @classmethod
    def write_data(
        cls,
        df,
        path,
        write_kwargs: Optional[dict] = None,
        columns: Optional[List[str]] = None,
        compression="infer",
        storage_options=None,
    ):
        resolved = cls.resolve_write_kwargs(write_kwargs or {})

        if columns:
            df = df[columns]

        if compression == "infer":
            compression = _infer_compression(path)

        csv_str = df.to_csv(index=False, **resolved)
        open_kw = dict(mode="wb", storage_options=storage_options)
        if compression:
            open_kw["compression"] = compression
        with open_file(path, **open_kw) as f:
            f.write(csv_str.encode("utf-8"))

    @classmethod
    def write_as_files(
        cls,
        df,
        path,
        partition_cols: Optional[List[str]] = None,
        compression="infer",
        storage_options=None,
        write_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        if kwargs:
            write_kwargs = {**(write_kwargs or {}), **kwargs}
        columns = (write_kwargs or {}).get("columns")
        write_fn = lambda g, fp: cls.write_data(
            g, fp, write_kwargs, columns, compression, storage_options
        )
        if partition_cols:
            _write_partitioned(
                df, path, partition_cols, write_fn, storage_options, ext=".csv"
            )
        else:
            cls.write_data(
                df,
                _resolve_write_path(path, 0),
                write_kwargs,
                columns,
                compression,
                storage_options,
            )


@register_spe_data_store_handler(LiteFrameDataFormat.JSON)
class JSONDataStoreHandler(SPEDataStoreHandler):
    """SPE handler for writing a pandas DataFrame to JSON files."""

    @classmethod
    def write_data(
        cls,
        df,
        path,
        write_kwargs: Optional[dict] = None,
        compression=None,
        storage_options=None,
    ):
        write_kwargs = write_kwargs or {}
        lines = write_kwargs.get("lines", True)
        orient = write_kwargs.get("orient", "records")
        json_kwargs = {
            k: v
            for k, v in write_kwargs.items()
            if k
            not in (
                "lines",
                "orient",
                "compression",
                "partition_cols",
                "storage_options",
            )
        }
        if lines:
            json_str = df.to_json(orient="records", lines=True, **json_kwargs)
        elif orient != "records":
            json_str = df.to_json(orient=orient, lines=False, **json_kwargs)
        else:
            json_str = df.to_json(orient="records", lines=False, **json_kwargs)

        open_kw = dict(mode="wb", storage_options=storage_options)
        if compression:
            open_kw["compression"] = compression
        with open_file(path, **open_kw) as f:
            f.write(json_str.encode("utf-8"))

    @classmethod
    def write_as_files(
        cls,
        df,
        path,
        partition_cols: Optional[List[str]] = None,
        compression=None,
        storage_options=None,
        write_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        if kwargs:
            write_kwargs = {**(write_kwargs or {}), **kwargs}
        write_fn = lambda g, fp: cls.write_data(
            g, fp, write_kwargs, compression, storage_options
        )
        if partition_cols:
            _write_partitioned(
                df, path, partition_cols, write_fn, storage_options, ext=".json"
            )
        else:
            cls.write_data(
                df,
                _resolve_write_path(path, 0),
                write_kwargs,
                compression,
                storage_options,
            )


@register_spe_data_store_handler(LiteFrameDataFormat.PARQUET)
class ParquetDataStoreHandler(SPEDataStoreHandler):
    """SPE handler for writing a pandas DataFrame to Parquet files."""

    _SUPPORTED_KWARGS = {
        "compression_level",
        "statistics",
        "row_group_size",
        "data_page_size",
    }

    @staticmethod
    def resolve_write_kwargs(write_kwargs: Optional[dict]) -> dict:
        """Forward top-level write_kwargs entries (except ``engine``) to the
        pyarrow Parquet writer after validating against the supported set.
        """
        result: dict = {}
        for key, value in (write_kwargs or {}).items():
            if key == "engine":
                continue
            if key in ParquetDataStoreHandler._SUPPORTED_KWARGS:
                result[key] = value
            else:
                raise NotImplementedError(
                    f"to_parquet does not support kwarg '{key}' for LiteFrame. "
                    f"Supported additional kwargs: "
                    f"{sorted(ParquetDataStoreHandler._SUPPORTED_KWARGS)}"
                )
        return result

    @staticmethod
    def _to_pyarrow_kwargs(resolved: dict) -> dict:
        """Translate polars-compatible keys to their pyarrow equivalents.

        polars ``write_parquet`` uses ``statistics``; ``pyarrow.parquet``
        writers expect ``write_statistics``.
        """
        if "statistics" in resolved:
            resolved["write_statistics"] = resolved.pop("statistics")
        return resolved

    @classmethod
    def write_data(
        cls,
        df,
        path,
        compression="snappy",
        storage_options=None,
        write_kwargs: Optional[dict] = None,
    ):
        comp = "none" if compression is None else compression
        resolved = cls._to_pyarrow_kwargs(cls.resolve_write_kwargs(write_kwargs))

        table = pa.Table.from_pandas(df, preserve_index=False)
        fs = get_fs(path, storage_options)
        pq.write_table(
            table,
            path,
            filesystem=to_arrow_file_system(fs),
            compression=comp,
            **resolved,
        )

    @classmethod
    def write_as_files(
        cls,
        df,
        path,
        partition_cols: Optional[List[str]] = None,
        compression="snappy",
        storage_options=None,
        write_kwargs: Optional[dict] = None,
    ):
        if partition_cols:
            comp = "none" if compression is None else compression
            resolved = cls._to_pyarrow_kwargs(cls.resolve_write_kwargs(write_kwargs))
            table = pa.Table.from_pandas(df, preserve_index=False)
            fs = get_fs(path, storage_options)
            pq.write_to_dataset(
                table,
                path,
                filesystem=to_arrow_file_system(fs),
                partition_cols=partition_cols,
                compression=comp,
                **resolved,
            )
        else:
            cls.write_data(
                df,
                _resolve_write_path(path, 0),
                compression,
                storage_options,
                write_kwargs,
            )


@register_op_adapter(LiteFrameLakeDataStore)
class LiteFrameLakeDataStoreAdapter(SPEOperatorAdapter):
    """SPE adapter for LiteFrame lake data store writes.

    Delegates format-specific code generation to SPEDataStoreHandler
    subclasses registered via ``register_spe_data_store_handler``.
    """

    @staticmethod
    def _get_handler(op):
        return _spe_data_store_handlers.get_handler(op.data_format)

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        return self._get_handler(op).generate_code(self, op, context)


@register_op_adapter(LiteFrameToLance)
class LiteFrameToLanceAdapter(SPEOperatorAdapter):
    """SPE adapter for writing a pandas DataFrame to a Lance dataset.

    Supports ``overwrite`` and ``append`` modes for local debugging.  The
    ``merge`` mode requires distributed fragment alignment (sharding by
    ``_fragment_id`` derived from ``_rowaddr``) and is not supported in
    SPE's single-pass codegen; use the DPE engine for merge writes.
    """

    @classmethod
    def _write_as_lance(
        cls,
        df,
        path,
        mode="overwrite",
        left_on=None,
        right_on=None,
        schema_bytes=None,
        storage_options=None,
        lance_kwargs=None,
    ):
        if lance is None:
            raise ImportError(
                "Please install pylance which is required to write Lance datasets"
            )
        if mode == "merge":
            raise NotImplementedError(
                "Lance 'merge' mode is not supported in SPE local execution; it "
                "requires distributed fragment alignment. Use the DPE engine "
                "for merge writes."
            )

        table = pa.Table.from_pandas(df, preserve_index=False)

        # Apply the user-provided schema= type override: fields naming input
        # columns are re-typed, the rest keep their auto-detected types.
        if schema_bytes is not None:
            user_schema = pa.ipc.read_schema(pa.BufferReader(schema_bytes))
            user_fields = {f.name: f for f in user_schema}
            merged_fields = [
                user_fields[f.name] if f.name in user_fields else f
                for f in table.schema
            ]
            table = table.cast(pa.schema(merged_fields, metadata=table.schema.metadata))

        write_kwargs = {}
        if storage_options:
            write_kwargs["storage_options"] = storage_options
        write_kwargs.update(lance_kwargs or {})
        lance.write_dataset(
            table,
            path,
            schema=table.schema,
            mode=mode or "overwrite",
            **write_kwargs,
        )

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        context.put_tileable_result_info(op.outputs[0], ResultInfo())
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        res_var = context.get_output_tileable_variable(op.outputs[0])
        args = self._translate_call_args(
            context,
            op.path,
            mode=op.mode,
            left_on=op.left_on,
            right_on=op.right_on,
            schema_bytes=op.schema_bytes,
            storage_options=op.storage_options,
            lance_kwargs=op.lance_kwargs,
        )
        return [
            f"{cls_name}._write_as_lance({input_var}, {', '.join(args)})",
            f"{res_var} = None",
        ]
