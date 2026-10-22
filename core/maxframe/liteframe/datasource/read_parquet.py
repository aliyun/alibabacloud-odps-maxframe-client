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

import json
from urllib.parse import urlparse

import pandas as pd

from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.datasource.utils import iter_local_files
from maxframe.liteframe.operators.core import LiteFrameDataFormat
from maxframe.liteframe.operators.datasource import LiteFrameLakeDataSource
from maxframe.liteframe.utils import normalize_dtypes_index


def read_parquet(
    path,
    columns=None,
    filters=None,
    default_index_type=None,
    storage_options=None,
    chunk_bytes="64M",
    merge_small_files=True,
    groups_as_chunks=True,
    session=None,
    **kwargs,
):
    """Read a parquet file into a LiteFrame with Arrow dtypes.

    Local (``file://`` or bare path) files are read with pandas and
    wrapped via :func:`from_local_df`. Remote paths are read in chunks
    by the MaxFrame backend; when a ``session`` is available the schema
    is inferred remotely, otherwise the tiler infers types from the
    Parquet file's built-in schema metadata at execution time.

    Parameters
    ----------
    path : str or list of str
        Path or list of paths to the parquet file(s) to read. Local
        paths and remote URIs are both supported.
    columns : list of str, optional
        Column names to select. If None, all columns are read.
    filters : list of tuple or list of list of tuple, optional
        Predicate filters to apply via parquet row-group pushdown.
    default_index_type : str, optional
        Index type for the resulting LiteFrame. If None, a default is
        chosen by the backend.
    storage_options : dict, optional
        Extra options for the storage backend (e.g. credentials).
    chunk_bytes : str, default '64M'
        Target chunk size in bytes for splitting remote files.
    merge_small_files : bool, default True
        Whether to merge small files into larger chunks.
    groups_as_chunks : bool, default True
        Whether to treat each parquet row group as a separate chunk.
    session : Session, optional
        Session to use for remote schema inference and execution. If
        None, the default session is used when available.
    **kwargs
        Additional keyword arguments forwarded to the parquet reader.

    Returns
    -------
    LiteFrame
        A lazy LiteFrame with Arrow dtypes representing the parquet
        data. Call ``.execute()`` to materialize the results.

    See Also
    --------
    LiteFrame.to_parquet : Write a LiteFrame to a Parquet file.
    read_csv : Read a CSV file into a LiteFrame.
    """
    local_test_mode = kwargs.pop("_local_test_mode", False)
    single_path = path[0] if isinstance(path, list) else path
    parsed_path = urlparse(single_path)
    if not local_test_mode and (
        not parsed_path.scheme or parsed_path.scheme.lower() == "file"
    ):
        local_dfs = []
        for file_path, _ in iter_local_files(path):
            local_dfs.append(
                pd.read_parquet(file_path, columns=columns, filters=filters, **kwargs)
            )
        if not local_dfs:
            raise FileNotFoundError(f"No files found matching path: {path}")
        df = pd.concat(local_dfs) if len(local_dfs) > 1 else local_dfs[0]
        return from_local_df(df)

    dtypes = pd.Series(dtype=object)
    physical_columns = None
    # Note: when columns is specified without a session, dtypes is left
    # empty so the tiler can infer correct types from the Parquet file's
    # built-in schema metadata.  Hardcoding pa.string() would produce
    # incorrect type information.

    if dtypes.empty and session is not None:
        dt_op = LiteFrameLakeDataSource(
            path=path,
            data_format=LiteFrameDataFormat.PARQUET,
            dtypes=pd.Series(dtype=object),
            columns=columns,
            filters=filters,
            default_index_type=default_index_type,
            storage_options=storage_options,
            chunk_bytes=chunk_bytes,
            merge_small_files=merge_small_files,
            groups_as_chunks=groups_as_chunks,
            read_kwargs=kwargs or None,
            read_stage="get_dtypes",
        )
        result = json.loads(
            dt_op._new_liteframe_from_source()
            .execute(session=session)
            .fetch(session=session)
        )
        # Capture the file's physical (original-case) column names for the
        # predicate-pushdown pusher to map logical -> physical names.
        physical_columns = [d["key"] for d in result]
        dtypes = pd.Series(
            {d["key"]: pd.api.types.pandas_dtype(d["value"]) for d in result}
        )
        dtypes = normalize_dtypes_index(dtypes)
        if columns:
            dtypes = dtypes[dtypes.index.isin([c.lower() for c in columns])]

    op = LiteFrameLakeDataSource(
        path=path,
        data_format=LiteFrameDataFormat.PARQUET,
        dtypes=dtypes,
        columns=columns,
        filters=filters,
        default_index_type=default_index_type,
        storage_options=storage_options,
        chunk_bytes=chunk_bytes,
        merge_small_files=merge_small_files,
        groups_as_chunks=groups_as_chunks,
        physical_columns=physical_columns,
        read_kwargs=kwargs or None,
    )
    return op._new_liteframe_from_source()
