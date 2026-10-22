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


def read_csv(
    path,
    usecols=None,
    dtype=None,
    names=None,
    sep=",",
    header="infer",
    default_index_type=None,
    storage_options=None,
    chunk_bytes="64M",
    merge_small_files=True,
    compression="infer",
    head_bytes="100k",
    session=None,
    **kwargs,
):
    """Read a CSV file into a LiteFrame with Arrow dtypes.

    Local (``file://`` or bare path) files are read with pandas and
    wrapped via :func:`from_local_df`. Remote paths are read in chunks
    by the MaxFrame backend; when a ``session`` is available the schema
    is inferred remotely, otherwise the tiler infers types at execution
    time.

    Parameters
    ----------
    path : str or list of str
        Path or list of paths to the CSV file(s) to read. Local paths
        and remote URIs are both supported.
    usecols : list of str, optional
        Column names to select. If None, all columns are read.
    dtype : dict or pandas.Series, optional
        Mapping of column name to dtype. Overrides inferred types for
        the given columns.
    names : list of str, optional
        Custom column names. Used when the file has no header.
    sep : str, default ','
        Field delimiter.
    header : int, 'infer', or None, default 'infer'
        Row number to use as column names, or ``'infer'`` to detect
        automatically, or ``None`` with explicit ``names``.
    default_index_type : str, optional
        Index type for the resulting LiteFrame. If None, a default is
        chosen by the backend.
    storage_options : dict, optional
        Extra options for the storage backend (e.g. credentials).
    chunk_bytes : str, default '64M'
        Target chunk size in bytes for splitting remote files.
    merge_small_files : bool, default True
        Whether to merge small files into larger chunks.
    compression : str, default 'infer'
        Compression of the input file (e.g. ``'gzip'``, ``'infer'``).
    head_bytes : str, default '100k'
        Number of bytes to read from the file head for schema inference.
    session : Session, optional
        Session to use for remote schema inference and execution. If
        None, the default session is used when available.
    **kwargs
        Additional keyword arguments forwarded to the CSV reader.

    Returns
    -------
    LiteFrame
        A lazy LiteFrame with Arrow dtypes representing the CSV data.
        Call ``.execute()`` to materialize the results.

    See Also
    --------
    LiteFrame.to_csv : Write a LiteFrame to a CSV file.
    read_parquet : Read a Parquet file into a LiteFrame.
    read_json : Read a JSON file into a LiteFrame.
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
                pd.read_csv(
                    file_path,
                    usecols=usecols,
                    dtype=dtype,
                    names=names,
                    sep=sep,
                    header=header,
                    compression=compression,
                    **kwargs,
                )
            )
        if not local_dfs:
            raise FileNotFoundError(f"No files found matching path: {path}")
        df = pd.concat(local_dfs) if len(local_dfs) > 1 else local_dfs[0]
        return from_local_df(df, default_index_type=default_index_type)

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
            data_format=LiteFrameDataFormat.CSV,
            dtypes=pd.Series(dtype=object),
            columns=usecols,
            dtype=dtype,
            default_index_type=default_index_type,
            storage_options=storage_options,
            chunk_bytes=chunk_bytes,
            merge_small_files=merge_small_files,
            read_kwargs={
                "names": names,
                "sep": sep,
                "header": header,
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
        data_format=LiteFrameDataFormat.CSV,
        dtypes=dtypes,
        columns=usecols,
        dtype=dtype,
        default_index_type=default_index_type,
        storage_options=storage_options,
        chunk_bytes=chunk_bytes,
        merge_small_files=merge_small_files,
        read_kwargs={
            "names": names,
            "sep": sep,
            "header": header,
            "compression": compression,
            "head_bytes": head_bytes,
            **kwargs,
        }
        or None,
    )
    return op._new_liteframe_from_source()
