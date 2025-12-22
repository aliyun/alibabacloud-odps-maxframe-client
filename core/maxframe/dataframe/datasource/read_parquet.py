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

import glob
import json
from collections import OrderedDict
from typing import MutableMapping, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from ...utils import make_dtypes

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
from ...utils import no_default
from ..operators import OutputType
from ..utils import parse_index, to_arrow_dtypes
from .core import (
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
    LakeDataSource,
)

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


def _resolve_dict_dtypes(dtypes_dict) -> pd.Series:
    if isinstance(dtypes_dict, list):
        dtypes_dict = OrderedDict([(d["key"], d["value"]) for d in dtypes_dict])
    names = list(dtypes_dict.keys())
    vals = [pd.api.types.pandas_dtype(dt) for dt in dtypes_dict.values()]
    return pd.Series(vals, index=names)


def read_parquet(
    path,
    engine: str = "auto",
    columns: list = None,
    groups_as_chunks: bool = False,
    dtype_backend: str = no_default,
    incremental_index: bool = False,
    storage_options: dict = None,
    use_nullable_dtypes: bool = no_default,
    *,
    dtypes: pd.Series = None,
    index_dtypes: pd.Series = None,
    memory_scale: int = None,
    merge_small_files: bool = True,
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
    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    MaxFrame DataFrame
    """
    from .dataframe import from_pandas

    engine_type = check_engine(engine)

    single_path = path[0] if isinstance(path, list) else path
    parsed_path = urlparse(single_path)
    local_test_mode = kwargs.pop("_local_test_mode", False)
    if not local_test_mode and (
        not parsed_path.scheme or parsed_path.scheme.lower() == "file"
    ):
        # todo chunk with multiple files and / or row groups?
        # just read locally when path is not remote
        local_dfs = []
        paths = path if isinstance(path, list) else [path]
        real_paths = []
        for path in paths:
            if "*" in path or "?" in path:
                real_paths.extend(glob.glob(path))
            else:
                real_paths.append(path)
        for path in real_paths:
            kw = {}
            if use_nullable_dtypes is not no_default:
                kw = {"use_nullable_dtypes": use_nullable_dtypes}
            if dtype_backend is not no_default:
                kw = {"dtype_backend": dtype_backend}
            local_dfs.append(
                pd.read_parquet(path, engine=engine_type, columns=columns, **kw)
            )
        df = pd.concat(local_dfs) if len(local_dfs) > 1 else local_dfs[0]
        return from_pandas(df)

    is_partitioned = None
    if dtypes is not None:
        dtypes = make_dtypes(dtypes)
        if index_dtypes is not None:
            index_dtypes = make_dtypes(index_dtypes)
    else:
        # need to submit a job to get dtypes
        dt_op = DataFrameReadParquet(
            path=path,
            engine=engine_type,
            columns=columns,
            groups_as_chunks=groups_as_chunks,
            dtype_backend=dtype_backend,
            read_kwargs=kwargs,
            incremental_index=incremental_index,
            storage_options=storage_options,
            memory_scale=memory_scale,
            merge_small_files=merge_small_files,
            read_stage="get_dtypes",
        )
        run_kwargs = run_kwargs or {}
        dt_result = json.loads(
            dt_op().execute(session=session, **run_kwargs).fetch(session=session),
            object_pairs_hook=OrderedDict,
        )
        is_partitioned = dt_result["is_partitioned"]
        dtypes = _resolve_dict_dtypes(dt_result["dtypes"])
        if dt_result.get("index_dtypes"):
            index_dtypes = _resolve_dict_dtypes(dt_result["index_dtypes"])

    if columns:
        dtypes = dtypes[columns]

    if dtype_backend is None:
        dtype_backend = options.dataframe.dtype_backend
    if dtype_backend == "pyarrow":
        dtypes = to_arrow_dtypes(dtypes)

    if index_dtypes is None:
        index_value = parse_index(pd.RangeIndex(-1))
    else:
        idx_df = pd.DataFrame([], columns=index_dtypes.index).astype(index_dtypes)
        pd_idx = pd.MultiIndex.from_frame(idx_df)
        if len(index_dtypes) == 1:
            pd_idx = pd_idx.get_level_values(0)
        index_value = parse_index(pd_idx, store_data=False)
        incremental_index = False

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
        gpu=gpu,
    )
    return op(index_value=index_value, columns_value=columns_value, dtypes=dtypes)
