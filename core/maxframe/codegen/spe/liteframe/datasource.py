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

import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from odps import ODPS

from maxframe.codegen.spe.core import (
    SPECodeContext,
    SPEOperatorAdapter,
    register_op_adapter,
)
from maxframe.dataframe.datasource.read_lance import normalize_lance_filters
from maxframe.io.odpsio import ODPSTableIO
from maxframe.lib.filesystem import get_fs, has_magic, open_file
from maxframe.liteframe.datasource.utils import infer_dtypes_from_pandas
from maxframe.liteframe.operators.core import LiteFrameDataFormat
from maxframe.liteframe.operators.datasource import (
    LiteFrameFromLocal,
    LiteFrameFromTensor,
    LiteFrameLakeDataSource,
    LiteFrameReadLance,
    LiteFrameReadODPSTable,
)
from maxframe.utils import (
    TypeDispatcher,
    convert_filters_to_arrow_expression,
    wrap_arrow_dtype,
)

try:
    import lance
except (ImportError, AttributeError):  # pragma: no cover
    # lance not installed or not compatible
    lance = None


@register_op_adapter(LiteFrameFromLocal)
class LiteFrameFromLocalAdapter(SPEOperatorAdapter):
    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        data_var = context.register_operator_constants(op.data)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        return [f"{res_var_name} = {data_var}"]


@register_op_adapter(LiteFrameFromTensor)
class LiteFrameFromTensorAdapter(SPEOperatorAdapter):
    """SPE adapter for ``LiteFrameFromTensor``.

    Mirrors the DPE executor: a 1-d tensor becomes a single-column
    LiteFrame and a 2-d tensor becomes one column per tensor column, named
    after ``op.dtypes.index``.  The tensor chunk is materialized as a numpy
    array in SPE.
    """

    @staticmethod
    def _from_tensor(tensor_data, col_names) -> pd.DataFrame:
        if tensor_data.ndim == 1:
            # 1-d tensor -> exactly one column.
            data = {col_names[0]: tensor_data.reshape(-1)}
        else:
            # 2-d tensor (m x n) -> n columns, one per tensor column.
            data = {
                col_names[j]: np.ascontiguousarray(tensor_data[:, j])
                for j in range(tensor_data.shape[1])
            }
        return pd.DataFrame(data)

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        input_var = context.get_input_tileable_variable(op.inputs[0])
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        col_names = list(op.dtypes.index)
        col_names_var = context.register_operator_constants(col_names)
        return [
            f"{res_var_name} = {cls_name}._from_tensor({input_var}, {col_names_var})"
        ]


@register_op_adapter(LiteFrameReadODPSTable)
class LiteFrameReadODPSTableAdapter(SPEOperatorAdapter):
    @staticmethod
    def _read_as_pandas(
        table_name: str,
        partitions: Optional[List[str]],
        columns: Optional[list],
        dtypes,
        nrows: Optional[int] = None,
        table_column_names: Optional[list] = None,
    ):
        # The temp table may store sanitized physical column names (e.g.
        # `_column_0` for the int-string logical name "0"); select the
        # physical names and rename the result back to the logical ones.
        read_columns = columns
        rename_map = None
        if table_column_names and columns is not None and dtypes is not None:
            index_by_logical = {logical: i for i, logical in enumerate(dtypes.index)}
            physical = []
            for col in columns:
                idx = index_by_logical.get(col)
                physical.append(
                    table_column_names[idx]
                    if idx is not None and idx < len(table_column_names)
                    else col
                )
            if physical != columns:
                read_columns = physical
                rename_map = {p: l for p, l in zip(physical, columns) if p != l}

        o = ODPS.from_environments() or ODPS.from_global()

        with ODPSTableIO(o).open_reader(table_name, partitions, read_columns) as reader:
            arrow_result = reader.read_all()

        if nrows is not None:
            arrow_result = arrow_result.slice(0, nrows)

        res = arrow_result.to_pandas(types_mapper=pd.ArrowDtype)
        if rename_map:
            res = res.rename(columns=rename_map)
        return res

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])
        args = self._translate_call_args(
            context,
            op.table_name,
            op.partitions,
            op.columns,
            op.dtypes,
            nrows=op.nrows,
        )
        args += ["table_column_names=" + repr(op.table_column_names)]
        return [f"{res_var_name} = {cls_name}._read_as_pandas({', '.join(args)})"]


# Mapping from common compression file extensions to compression names used by
# MaxFrame's filesystem layer (mirrors the DPE lake datasource executor).
_compression_type_map = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "xz",
    ".zst": "zstd",
    ".zip": "zip",
}


def _infer_compression(path: str) -> Optional[str]:
    """Infer compression type from a file path's extension."""
    _, ext = os.path.splitext(path)
    return _compression_type_map.get(ext)


def _expand_paths(path, storage_options=None) -> List[str]:
    """Resolve a path/glob/directory into a list of concrete file paths.

    Works for both local and cloud paths via MaxFrame's filesystem
    abstraction.  A bare path or a list of paths is accepted.  Directory
    paths are listed and hidden files (starting with ``_`` or ``.``) are
    skipped, matching the behaviour of the public ``read_*`` APIs.
    """
    paths = list(path) if isinstance(path, (list, tuple)) else [path]
    result: List[str] = []
    for p in paths:
        fs = get_fs(p, storage_options)
        if has_magic(p):
            result.extend(fs.glob(p))
        elif fs.isdir(p):
            for f in fs.ls(p):
                if not fs.isdir(f) and not os.path.basename(f).startswith(("_", ".")):
                    result.append(f)
        else:
            result.append(p)
    return result


def _normalize_columns(df):
    """Lowercase all column names of *df* in-place and return it.

    LiteFrame stores column names in lowercase internally.  Column names read
    from files retain their original casing, so they must be normalized before
    the DataFrame is returned to the SPE runtime.
    """
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _apply_dtype_overrides(df, dtypes):
    """Cast columns of *df* to the Arrow dtypes in *dtypes* (case-insensitive).

    ``dtypes`` is a :class:`pandas.Series` indexed by lowercased column names
    produced by dtype inference / user override.  Only columns present in both
    *df* and *dtypes* are cast; missing columns are left untouched.
    """
    if dtypes is None or len(dtypes) == 0:
        return df
    col_map = {str(c).lower(): c for c in df.columns}
    casts = {}
    for col, dt in dtypes.items():
        actual = col_map.get(str(col).lower())
        if actual is not None:
            casts[actual] = dt
    if casts:
        df = df.astype(casts)
    return df


def _apply_nrows(df, nrows):
    """Truncate *df* to *nrows* rows when set by head-pushdown optimization."""
    if nrows is not None and len(df) > nrows:
        df = df.iloc[:nrows]
    return df


def _finalize_read_df(dfs, dtypes, nrows):
    """Concatenate per-file frames and apply the standard read post-processing."""
    df = pd.concat(dfs) if len(dfs) > 1 else dfs[0]
    df = _normalize_columns(df)
    df = _apply_dtype_overrides(df, dtypes)
    df = _apply_nrows(df, nrows)
    return df


def _dtypes_to_json_list(df):
    """Infer dtypes from *df* and serialise to the ``[{key, value}]`` schema."""
    dtypes = infer_dtypes_from_pandas(df)
    return [{"key": col, "value": str(dt)} for col, dt in dtypes.items()]


def _extract_filter_columns(filters):
    """Extract the set of column names referenced by *filters*.

    Supports both the simple format ``[(col, op, val), ...]`` and the CNF
    format ``[[(col, op, val), ...], ...]``.
    """
    columns = set()
    if not filters:
        return columns
    groups = filters if isinstance(filters[0], list) else [filters]
    for group in groups:
        for condition in group:
            if isinstance(condition, (list, tuple)) and len(condition) >= 1:
                columns.add(condition[0])
    return columns


class SPEDataSourceHandler(ABC):
    """Format-specific strategy for SPE data source code generation."""

    @staticmethod
    @abstractmethod
    def read_as_pandas(path, **kwargs) -> pd.DataFrame:
        pass

    @classmethod
    def read_dtypes(cls, path, read_kwargs=None, storage_options=None):
        """Read the schema (column name -> dtype) of the first file."""
        paths = _expand_paths(path, storage_options)
        first_path = paths[0] if paths else None
        if first_path is None:
            return []
        df = cls.read_as_pandas(
            first_path,
            columns=None,
            dtype=None,
            read_kwargs=read_kwargs,
            storage_options=storage_options,
            nrows=1,
        )
        return _dtypes_to_json_list(df)

    @classmethod
    @abstractmethod
    def generate_code(cls, adapter, op, context: SPECodeContext) -> List[str]:
        pass


_spe_data_source_handlers = TypeDispatcher()


def register_spe_data_source_handler(fmt: LiteFrameDataFormat):
    """Decorator: register an SPEDataSourceHandler subclass for a format."""

    def wrapper(cls):
        _spe_data_source_handlers.register(fmt, cls)
        return cls

    return wrapper


@register_spe_data_source_handler(LiteFrameDataFormat.CSV)
class CSVDataSourceHandler(SPEDataSourceHandler):
    @staticmethod
    def read_as_pandas(
        path,
        columns=None,
        dtype=None,
        read_kwargs=None,
        storage_options=None,
        dtypes=None,
        nrows: Optional[int] = None,
    ):
        rk = read_kwargs or {}
        names = rk.get("names")
        sep = rk.get("sep", ",")
        header = rk.get("header", "infer")
        compression = rk.get("compression", "infer")

        paths = _expand_paths(path, storage_options)
        if not paths:
            raise FileNotFoundError(f"No files found matching path: {path}")

        extra_kwargs = {
            k: v
            for k, v in rk.items()
            if k not in ("names", "sep", "header", "compression", "head_bytes")
        }
        dfs = []
        remaining = nrows
        for pth in paths:
            comp = compression
            if comp == "infer":
                comp = _infer_compression(pth)
            file_nrows = remaining if remaining is not None else None
            with open_file(pth, compression=comp, storage_options=storage_options) as f:
                df = pd.read_csv(
                    f,
                    usecols=columns,
                    dtype=dtype,
                    names=names,
                    sep=sep,
                    header=header,
                    compression=None,
                    dtype_backend="pyarrow",
                    nrows=file_nrows,
                    **extra_kwargs,
                )
            dfs.append(df)
            if remaining is not None:
                remaining -= len(df)
                if remaining <= 0:
                    break

        return _finalize_read_df(dfs, dtypes, nrows)

    @classmethod
    def generate_code(cls, adapter, op, context: SPECodeContext) -> List[str]:
        cls_name = cls.__name__
        context.register_import(__name__, from_item=cls_name)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        if op.read_stage == "get_dtypes":
            context.register_import("json")
            args = adapter._translate_call_args(
                context,
                op.path,
                read_kwargs=op.read_kwargs,
                storage_options=op.storage_options,
            )
            return [
                f"{res_var_name} = json.dumps({cls_name}.read_dtypes("
                f"{', '.join(args)}))"
            ]

        args = adapter._translate_call_args(
            context,
            op.path,
            columns=op.columns,
            dtype=op.dtype,
            read_kwargs=op.read_kwargs,
            storage_options=op.storage_options,
            dtypes=op.dtypes,
            nrows=op.nrows,
        )
        return [f"{res_var_name} = {cls_name}.read_as_pandas({', '.join(args)})"]


@register_spe_data_source_handler(LiteFrameDataFormat.JSON)
class JSONDataSourceHandler(SPEDataSourceHandler):
    @staticmethod
    def read_as_pandas(
        path,
        columns=None,
        dtype=None,
        read_kwargs=None,
        storage_options=None,
        dtypes=None,
        nrows: Optional[int] = None,
    ):
        rk = read_kwargs or {}
        lines = rk.get("lines", True)
        compression = rk.get("compression", "infer")

        paths = _expand_paths(path, storage_options)
        if not paths:
            raise FileNotFoundError(f"No files found matching path: {path}")

        extra_kwargs = {
            k: v
            for k, v in rk.items()
            if k not in ("lines", "compression", "head_bytes")
        }
        dfs = []
        for pth in paths:
            comp = compression
            if comp == "infer":
                comp = _infer_compression(pth)
            with open_file(pth, compression=comp, storage_options=storage_options) as f:
                df = pd.read_json(
                    f,
                    lines=lines,
                    dtype=dtype,
                    compression=None,
                    dtype_backend="pyarrow",
                    **extra_kwargs,
                )
            if columns is not None:
                cols = columns if isinstance(columns, list) else [columns]
                col_map = {str(c).lower(): c for c in df.columns}
                select_cols = [col_map[c.lower()] for c in cols if c.lower() in col_map]
                df = df[select_cols]
            dfs.append(df)

        return _finalize_read_df(dfs, dtypes, nrows)

    @classmethod
    def generate_code(cls, adapter, op, context: SPECodeContext) -> List[str]:
        cls_name = cls.__name__
        context.register_import(__name__, from_item=cls_name)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        if op.read_stage == "get_dtypes":
            context.register_import("json")
            args = adapter._translate_call_args(
                context,
                op.path,
                read_kwargs=op.read_kwargs,
                storage_options=op.storage_options,
            )
            return [
                f"{res_var_name} = json.dumps({cls_name}.read_dtypes("
                f"{', '.join(args)}))"
            ]

        args = adapter._translate_call_args(
            context,
            op.path,
            columns=op.columns,
            dtype=op.dtype,
            read_kwargs=op.read_kwargs,
            storage_options=op.storage_options,
            dtypes=op.dtypes,
            nrows=op.nrows,
        )
        return [f"{res_var_name} = {cls_name}.read_as_pandas({', '.join(args)})"]


@register_spe_data_source_handler(LiteFrameDataFormat.PARQUET)
class ParquetDataSourceHandler(SPEDataSourceHandler):
    @staticmethod
    def read_as_pandas(
        path,
        columns=None,
        filters=None,
        storage_options=None,
        read_kwargs=None,
        dtypes=None,
        nrows: Optional[int] = None,
    ):
        paths = _expand_paths(path, storage_options)
        if not paths:
            raise FileNotFoundError(f"No files found matching path: {path}")

        filter_expr = convert_filters_to_arrow_expression(filters) if filters else None

        read_columns = columns
        if filter_expr is not None:
            filter_columns = _extract_filter_columns(filters)
            if filter_columns:
                requested = set(columns) if columns else set()
                extra = filter_columns - requested
                if extra:
                    read_columns = (
                        list(columns) + sorted(extra) if columns else sorted(extra)
                    )

        dfs = []
        for pth in paths:
            with open_file(pth, storage_options=storage_options) as f:
                pq_file = pq.ParquetFile(f)
                if filter_expr is not None:
                    table = pq_file.read(columns=read_columns).filter(filter_expr)
                    if columns is not None:
                        table = table.select(columns)
                else:
                    table = pq_file.read(columns=columns)
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
            dfs.append(df)

        return _finalize_read_df(dfs, dtypes, nrows)

    @staticmethod
    def read_dtypes(path, storage_options=None):
        paths = _expand_paths(path, storage_options)
        first_path = paths[0] if paths else None
        if first_path is None:
            return []
        with open_file(first_path, storage_options=storage_options) as f:
            schema = pq.ParquetFile(f).schema_arrow
        return [
            {"key": field.name, "value": str(wrap_arrow_dtype(field.type))}
            for field in schema
        ]

    @classmethod
    def generate_code(cls, adapter, op, context: SPECodeContext) -> List[str]:
        cls_name = cls.__name__
        context.register_import(__name__, from_item=cls_name)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        if op.read_stage == "get_dtypes":
            context.register_import("json")
            args = adapter._translate_call_args(
                context,
                op.path,
                storage_options=op.storage_options,
            )
            return [
                f"{res_var_name} = json.dumps({cls_name}.read_dtypes("
                f"{', '.join(args)}))"
            ]

        args = adapter._translate_call_args(
            context,
            op.path,
            columns=op.columns,
            filters=op.filters,
            storage_options=op.storage_options,
            read_kwargs=op.read_kwargs,
            dtypes=op.dtypes,
            nrows=op.nrows,
        )
        return [f"{res_var_name} = {cls_name}.read_as_pandas(" f"{', '.join(args)})"]


@register_op_adapter(LiteFrameLakeDataSource)
class LiteFrameLakeDataSourceAdapter(SPEOperatorAdapter):
    """SPE adapter for reading Parquet/CSV/JSON files into a pandas DataFrame.

    Delegates format-specific code generation to SPEDataSourceHandler
    subclasses registered via ``register_spe_data_source_handler``.
    Generated code references the handler's own class name (e.g.
    ``CSVDataSourceHandler.read_as_pandas(...)``).
    """

    @staticmethod
    def _get_handler(op):
        return _spe_data_source_handlers.get_handler(op.data_format)

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        handler = self._get_handler(op)
        return handler.generate_code(self, op, context)


@register_op_adapter(LiteFrameReadLance)
class LiteFrameReadLanceAdapter(SPEOperatorAdapter):
    """SPE adapter for reading Lance datasets into a pandas DataFrame.

    Lance datasets are read via ``pylance`` and returned as a pandas DataFrame
    with Arrow-backed dtypes, mirroring the DPE executor.  Column names are
    normalized to lowercase to match LiteFrame's logical namespace.
    """

    @staticmethod
    def _open_dataset(
        path,
        version=None,
        asof=None,
        storage_options=None,
        read_kwargs=None,
    ):
        if lance is None:
            raise ImportError(
                "Please install pylance which is required to read Lance datasets"
            )
        ds_kwargs = {}
        if version is not None:
            ds_kwargs["version"] = version
        if asof is not None:
            ds_kwargs["asof"] = asof
        if storage_options:
            ds_kwargs["storage_options"] = storage_options
        ds_kwargs.update(read_kwargs or {})
        return lance.dataset(path, **ds_kwargs)

    @staticmethod
    def _read_as_pandas(
        path,
        columns=None,
        version=None,
        asof=None,
        filters=None,
        with_row_address=False,
        storage_options=None,
        read_kwargs=None,
        dtypes=None,
        nrows: Optional[int] = None,
    ):
        ds = LiteFrameReadLanceAdapter._open_dataset(
            path, version, asof, storage_options, read_kwargs
        )

        filter_expr = normalize_lance_filters(filters)
        to_table_kwargs = {}
        if columns is not None:
            to_table_kwargs["columns"] = columns
        if filter_expr is not None:
            to_table_kwargs["filter"] = filter_expr
        if with_row_address:
            to_table_kwargs["with_row_address"] = True

        table = ds.to_table(**to_table_kwargs)
        df = table.to_pandas(types_mapper=pd.ArrowDtype)
        # Normalize column names to lowercase to match LiteFrame semantics.
        # Lance schemas may use mixed-case physical names, but LiteFrame
        # normalizes inferred dtypes and expressions to lowercase.
        df = _normalize_columns(df)
        df = _apply_dtype_overrides(df, dtypes)
        df = _apply_nrows(df, nrows)
        return df

    @staticmethod
    def _read_dtypes(
        path,
        version=None,
        asof=None,
        storage_options=None,
        read_kwargs=None,
    ):
        ds = LiteFrameReadLanceAdapter._open_dataset(
            path, version, asof, storage_options, read_kwargs
        )
        return [{"key": field.name, "value": str(field.type)} for field in ds.schema]

    def generate_code(self, op, context: SPECodeContext) -> List[str]:
        cls_name = type(self).__name__
        context.register_import(__name__, from_item=cls_name)
        res_var_name = context.get_output_tileable_variable(op.outputs[0])

        if op.read_stage == "get_dtypes":
            context.register_import("json")
            args = self._translate_call_args(
                context,
                op.path,
                version=op.version,
                asof=op.asof,
                storage_options=op.storage_options,
                read_kwargs=op.read_kwargs,
            )
            return [
                f"{res_var_name} = json.dumps({cls_name}._read_dtypes("
                f"{', '.join(args)}))"
            ]

        args = self._translate_call_args(
            context,
            op.path,
            columns=op.columns,
            version=op.version,
            asof=op.asof,
            filters=op.filters,
            with_row_address=op.with_row_address,
            storage_options=op.storage_options,
            read_kwargs=op.read_kwargs,
            dtypes=op.dtypes,
            nrows=op.nrows,
        )
        return [f"{res_var_name} = {cls_name}._read_as_pandas({', '.join(args)})"]
