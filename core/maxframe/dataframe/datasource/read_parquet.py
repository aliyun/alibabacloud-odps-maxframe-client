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

from typing import MutableMapping, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from ...protocol import DefaultIndexType

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None

from ... import opcodes
from ...config import options
from ...serialization.serializables import (
    BoolField,
    DictField,
    Int32Field,
    Int64Field,
    ListField,
    StringField,
)
from ...utils import make_dtypes, no_default
from ..operators import OutputType
from ..utils import parse_index, to_arrow_dtypes, validate_default_index_type
from .core import (
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
    LakeDataSource,
)
from .utils import get_lake_output_info, iter_local_files

PARQUET_MEMORY_SCALE = 15
STRING_FIELD_OVERHEAD = 50


def check_engine(engine):
    if engine == "auto":
        return "pyarrow"
    elif engine in ("pyarrow",):
        return engine
    else:  # pragma: no cover
        raise RuntimeError("Unsupported engine {} to read parquet.".format(engine))


class DataFrameReadParquet(
    LakeDataSource,
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
):
    _op_type_ = opcodes.READ_PARQUET

    engine = StringField("engine")
    columns = ListField("columns")
    groups_as_chunks = BoolField("groups_as_chunks", default=None)
    group_index = Int32Field("group_index", default=None)
    read_kwargs = DictField("read_kwargs", default=None)
    # for chunk
    partitions = DictField("partitions", default=None)
    partition_keys = DictField("partition_keys", default=None)
    # deprecated fields
    num_group_rows = Int64Field("num_group_rows", default=None)
    first_chunk_row_num = Int64Field("first_chunk_row_num", default=None)
    first_chunk_raw_bytes = Int64Field("first_chunk_raw_bytes", default=None)

    def get_columns(self):
        return self.columns

    def set_pruned_columns(self, columns, *, keep_order=None):
        self.columns = columns

    def __call__(self, index_value=None, columns_value=None, dtypes=None):
        if self.read_stage is not None:
            # output for planning or meta fetching
            self._output_types = [OutputType.scalar]
            return self.new_tileable(None, shape=(), dtype=np.dtype("O"))
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
    dtype_backend: str = no_default,
    default_index_type: Union[DefaultIndexType, str] = None,
    storage_options: dict = None,
    use_nullable_dtypes: bool = no_default,
    *,
    dtypes: pd.Series = None,
    index_dtypes: pd.Series = None,
    memory_scale: int = None,
    merge_small_files: bool = True,
    merge_small_file_options: dict = None,
    gpu: bool = None,
    session=None,
    run_kwargs: dict = None,
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
    engine : {'auto', 'pyarrow'}, default 'auto'
        Parquet library to use. The default behavior is to try 'pyarrow',
    storage_options: dict, optional
        Options for storage connection.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    groups_as_chunks : bool, default False
        if True, each row group correspond to a chunk.
        if False, each file correspond to a chunk.
        Only available for 'pyarrow' engine.
    default_index_type: {None, 'range', 'incremental'}, default None
        If index_col not specified, specify type of index to generate.
        If not specified, `options.dataframe.default_index_type` will be used.
    dtype_backend: {'numpy', 'pyarrow'}, default 'numpy'
        Back-end data type applied to the resultant DataFrame (still experimental).
    storage_options: dict, optional
        Options for storage connection.
    memory_scale: int, optional
        Scale that real memory occupation divided with raw file size.
    merge_small_files: bool, default True
        Merge small files whose size is small.
    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    MaxFrame DataFrame
    """
    from .dataframe import from_pandas

    engine_type = check_engine(engine)
    default_index_type = validate_default_index_type(default_index_type, **kwargs)

    single_path = path[0] if isinstance(path, list) else path
    parsed_path = urlparse(single_path)
    local_test_mode = kwargs.pop("_local_test_mode", False)
    if not local_test_mode and (
        not parsed_path.scheme or parsed_path.scheme.lower() == "file"
    ):
        # todo chunk with multiple files and / or row groups?
        # just read locally when path is not remote
        local_dfs = []
        for path, part_keys in iter_local_files(path):
            kw = {}
            if use_nullable_dtypes is not no_default:
                kw = {"use_nullable_dtypes": use_nullable_dtypes}
            if dtype_backend is not no_default:
                kw = {"dtype_backend": dtype_backend}
            local_df = pd.read_parquet(path, engine=engine_type, columns=columns, **kw)
            for k, v in part_keys or ():
                local_df[k] = v
            local_dfs.append(local_df)
        df = pd.concat(local_dfs) if len(local_dfs) > 1 else local_dfs[0]
        return from_pandas(df)

    common_kwargs = dict(
        engine=engine_type,
        columns=columns,
        groups_as_chunks=groups_as_chunks,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
        read_kwargs=kwargs,
    )
    # Get dtypes, index_dtypes and index_value using the common utility function
    result = get_lake_output_info(
        DataFrameReadParquet,
        path=path,
        default_index_type=default_index_type,
        dtype=dtypes,
        index_dtypes=index_dtypes,
        session=session,
        run_kwargs=run_kwargs,
        **common_kwargs,
    )

    dtypes = result.dtypes
    index_value = result.index_value
    index_dtypes = result.index_dtypes
    is_partitioned = result.is_partitioned

    # Ensure dtypes are properly formatted
    if dtypes is not None:
        dtypes = make_dtypes(dtypes)

    # Apply column filtering if needed
    if columns and dtypes is not None:
        dtypes = dtypes[columns]

    if dtype_backend is None:
        dtype_backend = options.dataframe.dtype_backend
    if dtype_backend == "pyarrow":
        dtypes = to_arrow_dtypes(dtypes)

    columns_value = parse_index(dtypes.index, store_data=True)
    default_index_type = None if index_dtypes is not None else default_index_type
    op = DataFrameReadParquet(
        path=path,
        default_index_type=default_index_type,
        is_partitioned=is_partitioned,
        memory_scale=memory_scale,
        merge_small_files=merge_small_files,
        merge_small_file_options=merge_small_file_options,
        gpu=gpu,
        **common_kwargs,
    )
    return op(index_value=index_value, columns_value=columns_value, dtypes=dtypes)
