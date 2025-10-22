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

import os
from typing import Dict, MutableMapping, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None

try:
    import fastparquet
except ImportError:
    fastparquet = None

from ... import opcodes
from ...config import options
from ...lib.dtypes_extension import ArrowDtype
from ...lib.filesystem import FileSystem, get_fs, glob, open_file
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    Int32Field,
    Int64Field,
    ListField,
    StringField,
)
from ...utils import lazy_import
from ..operators import OutputType
from ..utils import parse_index, to_arrow_dtypes
from .core import (
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
    IncrementalIndexDatasource,
)

PARQUET_MEMORY_SCALE = 15
STRING_FIELD_OVERHEAD = 50
cudf = lazy_import("cudf")


def check_engine(engine):
    if engine == "auto":
        if pa is not None:
            return "pyarrow"
        elif fastparquet is not None:  # pragma: no cover
            return "fastparquet"
        else:  # pragma: no cover
            raise RuntimeError("Please install either pyarrow or fastparquet.")
    elif engine == "pyarrow":
        if pa is None:  # pragma: no cover
            raise RuntimeError("Please install pyarrow first.")
        return engine
    elif engine == "fastparquet":
        if fastparquet is None:  # pragma: no cover
            raise RuntimeError("Please install fastparquet first.")
        return engine
    else:  # pragma: no cover
        raise RuntimeError("Unsupported engine {} to read parquet.".format(engine))


def get_engine(engine):
    if engine == "pyarrow":
        return ArrowEngine()
    elif engine == "fastparquet":
        return FastpaquetEngine()
    else:  # pragma: no cover
        raise RuntimeError("Unsupported engine {}".format(engine))


class ParquetEngine:
    def get_row_num(self, f):
        raise NotImplementedError

    def read_dtypes(self, f, **kwargs):
        raise NotImplementedError

    def read_to_pandas(self, f, columns=None, nrows=None, dtype_backend=None, **kwargs):
        raise NotImplementedError

    def read_group_to_pandas(
        self, f, group_index, columns=None, nrows=None, dtype_backend=None, **kwargs
    ):
        raise NotImplementedError

    def read_partitioned_to_pandas(
        self,
        f,
        partitions: Dict,
        partition_keys: Dict,
        columns=None,
        nrows=None,
        dtype_backend=None,
        **kwargs,
    ):
        raw_df = self.read_to_pandas(
            f, columns=columns, nrows=nrows, dtype_backend=dtype_backend, **kwargs
        )
        for col, value in partition_keys.items():
            dictionary = partitions[col]
            raw_df[col] = pd.Series(
                value,
                dtype=pd.CategoricalDtype(categories=dictionary.tolist()),
                index=raw_df.index,
            )
        return raw_df

    def read_partitioned_dtypes(self, fs: FileSystem, directory, storage_options):
        # As ParquetDataset will iterate all files,
        # here we just find one file to infer dtypes
        current_path = directory
        partition_cols = []
        while fs.isdir(current_path):
            _, dirs, files = next(fs.walk(current_path))
            dirs = [d for d in dirs if not d.startswith(".")]
            files = [f for f in files if not f.startswith(".")]
            if len(files) == 0:
                # directory as partition
                partition_cols.append(dirs[0].split("=", 1)[0])
                current_path = os.path.join(current_path, dirs[0])
            elif len(dirs) == 0:
                # parquet files in deepest directory
                current_path = os.path.join(current_path, files[0])
            else:  # pragma: no cover
                raise ValueError(
                    "Files and directories are mixed in an intermediate directory"
                )

        # current path is now a parquet file
        with open_file(current_path, storage_options=storage_options) as f:
            dtypes = self.read_dtypes(f)
        for partition in partition_cols:
            dtypes[partition] = pd.CategoricalDtype()
        return dtypes


def _parse_prefix(path):
    path_prefix = ""
    if isinstance(path, str):
        parsed_path = urlparse(path)
        if parsed_path.scheme:
            path_prefix = f"{parsed_path.scheme}://{parsed_path.netloc}"
    return path_prefix


class ArrowEngine(ParquetEngine):
    def get_row_num(self, f):
        file = pq.ParquetFile(f)
        return file.metadata.num_rows

    def read_dtypes(self, f, **kwargs):
        file = pq.ParquetFile(f)
        return file.schema_arrow.empty_table().to_pandas().dtypes

    @classmethod
    def _table_to_pandas(cls, t, nrows=None, dtype_backend=None):
        if nrows is not None:
            t = t.slice(0, nrows)
        if dtype_backend == "pyarrow":
            df = t.to_pandas(types_mapper={pa.string(): ArrowDtype(pa.string())}.get)
        else:
            df = t.to_pandas()
        return df

    def read_to_pandas(self, f, columns=None, nrows=None, dtype_backend=None, **kwargs):
        file = pq.ParquetFile(f)
        t = file.read(columns=columns, **kwargs)
        return self._table_to_pandas(t, nrows=nrows, dtype_backend=dtype_backend)

    def read_group_to_pandas(
        self, f, group_index, columns=None, nrows=None, dtype_backend=None, **kwargs
    ):
        file = pq.ParquetFile(f)
        t = file.read_row_group(group_index, columns=columns, **kwargs)
        return self._table_to_pandas(t, nrows=nrows, dtype_backend=dtype_backend)


class FastpaquetEngine(ParquetEngine):
    def get_row_num(self, f):
        file = fastparquet.ParquetFile(f)
        return file.count()

    def read_dtypes(self, f, **kwargs):
        file = fastparquet.ParquetFile(f)
        dtypes_dict = file._dtypes()
        return pd.Series(dict((c, dtypes_dict[c]) for c in file.columns))

    def read_to_pandas(self, f, columns=None, nrows=None, dtype_backend=None, **kwargs):
        file = fastparquet.ParquetFile(f)
        df = file.to_pandas(columns, **kwargs)
        if nrows is not None:
            df = df.head(nrows)
        if dtype_backend == "pyarrow":
            df = df.astype(to_arrow_dtypes(df.dtypes).to_dict())
        return df


class CudfEngine:
    @classmethod
    def read_to_cudf(cls, file, columns: list = None, nrows: int = None, **kwargs):
        df = cudf.read_parquet(file, columns=columns, **kwargs)
        if nrows is not None:
            df = df.head(nrows)
        return df

    def read_group_to_cudf(
        self, file, group_index: int, columns: list = None, nrows: int = None, **kwargs
    ):
        return self.read_to_cudf(
            file, columns=columns, nrows=nrows, row_groups=group_index, **kwargs
        )

    @classmethod
    def read_partitioned_to_cudf(
        cls,
        file,
        partitions: Dict,
        partition_keys: Dict,
        columns=None,
        nrows=None,
        **kwargs,
    ):
        # cudf will read entire partitions even if only one partition provided,
        # so we just read with pyarrow and convert to cudf DataFrame
        file = pq.ParquetFile(file)
        t = file.read(columns=columns, **kwargs)
        t = t.slice(0, nrows) if nrows is not None else t
        t = pa.table(t.columns, names=t.column_names)
        raw_df = cudf.DataFrame.from_arrow(t)
        for col, value in partition_keys.items():
            dictionary = partitions[col].tolist()
            codes = cudf.core.column.as_column(
                dictionary.index(value), length=len(raw_df)
            )
            raw_df[col] = cudf.core.column.build_categorical_column(
                categories=dictionary,
                codes=codes,
                size=codes.size,
                offset=codes.offset,
                ordered=False,
            )
        return raw_df


class DataFrameReadParquet(
    IncrementalIndexDatasource,
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
):
    _op_type_ = opcodes.READ_PARQUET

    path = AnyField("path")
    engine = StringField("engine")
    columns = ListField("columns")
    dtype_backend = StringField("dtype_backend", default=None)
    groups_as_chunks = BoolField("groups_as_chunks", default=None)
    group_index = Int32Field("group_index", default=None)
    read_kwargs = DictField("read_kwargs", default=None)
    incremental_index = BoolField("incremental_index", default=None)
    storage_options = DictField("storage_options", default=None)
    is_partitioned = BoolField("is_partitioned", default=None)
    merge_small_files = BoolField("merge_small_files", default=None)
    merge_small_file_options = DictField("merge_small_file_options", default=None)
    # for chunk
    partitions = DictField("partitions", default=None)
    partition_keys = DictField("partition_keys", default=None)
    num_group_rows = Int64Field("num_group_rows", default=None)
    # as read meta may be too time-consuming when number of files is large,
    # thus we only read first file to get row number and raw file size
    first_chunk_row_num = Int64Field("first_chunk_row_num", default=None)
    first_chunk_raw_bytes = Int64Field("first_chunk_raw_bytes", default=None)

    def get_columns(self):
        return self.columns

    def set_pruned_columns(self, columns, *, keep_order=None):
        self.columns = columns

    def __call__(self, index_value=None, columns_value=None, dtypes=None):
        self._output_types = [OutputType.dataframe]
        shape = (np.nan, len(dtypes))
        return self.new_dataframe(
            None,
            shape,
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
        )

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "DataFrameReadParquet"
    ):  # pragma: no cover
        # todo implement this to facilitate local computation
        ctx[op.outputs[0].key] = float("inf")


def read_parquet(
    path,
    engine: str = "auto",
    columns: list = None,
    groups_as_chunks: bool = False,
    dtype_backend: str = None,
    incremental_index: bool = False,
    storage_options: dict = None,
    memory_scale: int = None,
    merge_small_files: bool = True,
    merge_small_file_options: dict = None,
    gpu: bool = None,
    **kwargs,
):
    """
    Load a parquet object from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL.
        For file URLs, a host is expected. A local file could be:
        ``file://localhost/path/to/table.parquet``.
        A file URL can also be a path to a directory that contains multiple
        partitioned parquet files. Both pyarrow and fastparquet support
        paths to directories as well as file URLs. A directory path could be:
        ``file://localhost/path/to/tables``.
        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handler (e.g. via builtin ``open`` function)
        or ``StringIO``.
    engine : {'auto', 'pyarrow', 'fastparquet'}, default 'auto'
        Parquet library to use. The default behavior is to try 'pyarrow',
        falling back to 'fastparquet' if 'pyarrow' is unavailable.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    groups_as_chunks : bool, default False
        if True, each row group correspond to a chunk.
        if False, each file correspond to a chunk.
        Only available for 'pyarrow' engine.
    incremental_index: bool, default False
        If index_col not specified, ensure range index incremental,
        gain a slightly better performance if setting False.
    dtype_backend: {'numpy', 'pyarrow'}, default 'numpy'
        Back-end data type applied to the resultant DataFrame (still experimental).
    storage_options: dict, optional
        Options for storage connection.
    memory_scale: int, optional
        Scale that real memory occupation divided with raw file size.
    merge_small_files: bool, default True
        Merge small files whose size is small.
    merge_small_file_options: dict
        Options for merging small files
    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    MaxFrame DataFrame
    """

    engine_type = check_engine(engine)
    engine = get_engine(engine_type)

    single_path = path[0] if isinstance(path, list) else path
    fs = get_fs(single_path, storage_options)
    is_partitioned = False
    if fs.isdir(single_path):
        paths = fs.ls(path)
        if all(fs.isdir(p) for p in paths):
            # If all are directories, it is read as a partitioned dataset.
            dtypes = engine.read_partitioned_dtypes(fs, path, storage_options)
            is_partitioned = True
        else:
            with fs.open(paths[0], mode="rb") as f:
                dtypes = engine.read_dtypes(f)
    else:
        if not isinstance(path, list):
            file_path = glob(path, storage_options=storage_options)[0]
        else:
            file_path = path[0]

        with open_file(file_path, storage_options=storage_options) as f:
            dtypes = engine.read_dtypes(f)

    if columns:
        dtypes = dtypes[columns]

    if dtype_backend is None:
        dtype_backend = options.dataframe.dtype_backend
    if dtype_backend == "pyarrow":
        dtypes = to_arrow_dtypes(dtypes)

    index_value = parse_index(pd.RangeIndex(-1))
    columns_value = parse_index(dtypes.index, store_data=True)
    op = DataFrameReadParquet(
        path=path,
        engine=engine_type,
        columns=columns,
        groups_as_chunks=groups_as_chunks,
        dtype_backend=dtype_backend,
        read_kwargs=kwargs,
        incremental_index=incremental_index,
        storage_options=storage_options,
        is_partitioned=is_partitioned,
        memory_scale=memory_scale,
        merge_small_files=merge_small_files,
        merge_small_file_options=merge_small_file_options,
        gpu=gpu,
    )
    return op(index_value=index_value, columns_value=columns_value, dtypes=dtypes)
