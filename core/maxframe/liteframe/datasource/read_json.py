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


def read_json(
    path,
    *,
    lines=True,
    dtype=None,
    usecols=None,
    chunk_bytes="64M",
    compression="infer",
    head_bytes="100k",
    default_index_type=None,
    storage_options=None,
    merge_small_files=True,
    session=None,
    **kwargs,
):
    """
    Read a JSON file into a LiteFrame with Arrow dtypes.

    By default reads line-delimited JSON (JSONL/NDJSON) which is the natural
    fit for chunked distributed I/O.

    Parameters
    ----------
    path : str
        File path or glob pattern. Supports local and cloud storage paths.
    lines : bool, default True
        Read the file as line-delimited JSON (one JSON object per line).
    dtype : dict, optional
        Column name to dtype mapping for explicit type specification.
    usecols : list, optional
        Return a subset of columns.
    chunk_bytes : str, default '64M'
        Target chunk size for splitting files.
    compression : str, default 'infer'
        Compression type. Inferred from file extension if 'infer'.
    head_bytes : str, default '100k'
        Bytes to read for schema inference.
    default_index_type : str, optional
        Type of default index to generate.
    storage_options : dict, optional
        Extra options for storage connection.
    merge_small_files : bool, default True
        Merge small files into larger chunks.
    session : session, optional
        Execution session for remote dtype inference.

    Returns
    -------
    LiteFrame

    See Also
    --------
    LiteFrame.to_json : Write a LiteFrame to a JSON file.
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
            df = pd.read_json(
                file_path,
                lines=lines,
                dtype=dtype,
                compression=compression,
            )
            if usecols is not None:
                cols = usecols if isinstance(usecols, list) else [usecols]
                df = df[cols]
            local_dfs.append(df)
        if not local_dfs:
            raise FileNotFoundError(f"No files found matching path: {path}")
        df = pd.concat(local_dfs) if len(local_dfs) > 1 else local_dfs[0]
        return from_local_df(df)

    dtypes = pd.Series(dtype=object)
    user_dtypes = None
    if dtype is not None:
        user_dtypes = normalize_dtypes_index(pd.Series(dtype))

    # Determine whether user-provided dtypes fully cover the requested columns.
    # If usecols is specified and user_dtypes covers all of them, skip remote
    # inference.  Otherwise, infer the full schema remotely and overlay
    # user-provided overrides on top.
    user_dtypes_complete = (
        user_dtypes is not None
        and usecols is not None
        and set(c.lower() for c in usecols).issubset(set(user_dtypes.index))
    )

    if user_dtypes_complete:
        dtypes = user_dtypes
    elif session is not None:
        dt_op = LiteFrameLakeDataSource(
            path=path,
            data_format=LiteFrameDataFormat.JSON,
            dtypes=pd.Series(dtype=object),
            columns=usecols,
            dtype=dtype,
            chunk_bytes=chunk_bytes,
            default_index_type=default_index_type,
            storage_options=storage_options,
            merge_small_files=merge_small_files,
            read_kwargs={
                "lines": lines,
                "compression": compression,
                "head_bytes": head_bytes,
                **kwargs,
            }
            or None,
            read_stage="get_dtypes",
        )
        result = json.loads(
            dt_op._new_liteframe_from_source()
            .execute(session=session)
            .fetch(session=session)
        )
        dtypes = pd.Series(
            {d["key"]: pd.api.types.pandas_dtype(d["value"]) for d in result}
        )
        dtypes = normalize_dtypes_index(dtypes)
        if usecols:
            dtypes = dtypes[dtypes.index.isin([c.lower() for c in usecols])]
        # Overlay user-provided dtype overrides on top of inferred dtypes.
        if user_dtypes is not None and len(user_dtypes) > 0:
            for col, dt in user_dtypes.items():
                dtypes[col] = dt
    elif user_dtypes is not None:
        dtypes = user_dtypes

    op = LiteFrameLakeDataSource(
        path=path,
        data_format=LiteFrameDataFormat.JSON,
        dtypes=dtypes,
        columns=usecols,
        dtype=dtype,
        chunk_bytes=chunk_bytes,
        default_index_type=default_index_type,
        storage_options=storage_options,
        merge_small_files=merge_small_files,
        read_kwargs={
            "lines": lines,
            "compression": compression,
            "head_bytes": head_bytes,
            **kwargs,
        }
        or None,
    )
    return op._new_liteframe_from_source()
