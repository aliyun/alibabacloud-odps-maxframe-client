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
import pyarrow as pa

from maxframe.dataframe.datasource.read_lance import normalize_lance_filters
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.operators.datasource import LiteFrameReadLance
from maxframe.liteframe.utils import normalize_dtypes_index
from maxframe.utils import arrow_type_from_str, wrap_arrow_dtype

try:
    import lance
except (ImportError, AttributeError):  # pragma: no cover
    lance = None


def _read_lance_local(
    path, version=None, asof=None, columns=None, filters=None, with_row_address=False
):
    """Read a local Lance dataset and return a pandas DataFrame."""
    if lance is None:
        raise ImportError(
            "Please install pylance which is required to read Lance datasets"
        )

    ds = lance.dataset(path, version=version, asof=asof)
    filter_expr = normalize_lance_filters(filters)
    table = ds.to_table(
        columns=columns, filter=filter_expr, with_row_address=with_row_address
    )
    # Keep columns pyarrow-backed so downstream arrow-type conversion works.
    return table.to_pandas(types_mapper=pd.ArrowDtype)


def read_lance(
    path,
    version=None,
    *,
    asof=None,
    columns=None,
    filters=None,
    with_row_address=False,
    storage_options=None,
    chunk_bytes="64M",
    merge_small_files=True,
    session=None,
    **kwargs,
):
    """Read a Lance dataset into a LiteFrame.

    Parameters
    ----------
    path : str
        Path to the Lance dataset. Supports local, OSS, and S3 URIs.
    version : int, optional
        Specific version of the dataset to read.
    asof : str, optional
        ISO 8601 timestamp for point-in-time read. Cannot be used with version.
    columns : list, optional
        Column names to read. None reads all columns.
    filters : str or list, optional
        SQL filter string or CNF filter list for predicate pushdown.
    with_row_address : bool, default False
        If True, include Lance's ``_rowaddr`` column as a hidden column.
    storage_options : dict, optional
        Storage connection options (e.g. role_arn for OSS).
    chunk_bytes : str, default "64M"
        Target chunk size for splitting fragments.
    merge_small_files : bool, default True
        Whether to merge small fragments into larger chunks.
    **kwargs
        Additional keyword arguments passed to Lance.

    Returns
    -------
    LiteFrame

    See Also
    --------
    LiteFrame.to_lance : Write a LiteFrame to a Lance dataset.
    read_parquet : Read a Parquet file into a LiteFrame.
    """
    if version is not None and asof is not None:
        raise ValueError("Cannot specify both 'version' and 'asof' parameters")

    if filters is None:
        filters = kwargs.pop("filter", None)

    local_test_mode = kwargs.pop("_local_test_mode", False)

    filters = normalize_lance_filters(filters)

    parsed_path = urlparse(path)
    if not local_test_mode and (
        not parsed_path.scheme or parsed_path.scheme.lower() == "file"
    ):
        local_df = _read_lance_local(
            path=path,
            version=version,
            asof=asof,
            columns=columns,
            filters=filters,
            with_row_address=with_row_address,
        )
        lf = from_local_df(local_df)
        if with_row_address:
            lf._data.frame_metadata = FrameMetadata(hidden_columns=["_rowaddr"])
        return lf

    dtypes = pd.Series(dtype=object)
    physical_columns = None
    # Note: when columns is specified we must NOT hardcode pa.string() for
    # every selected column -- that would report incorrect types to
    # downstream logic (column-pruning metadata, operator fusion, type
    # promotion).  Instead, leave dtypes empty and, when a session is
    # available, infer the real per-column Arrow types from the Lance
    # dataset schema via a get_dtypes stage (mirroring read_parquet).
    if dtypes.empty and session is not None:
        dt_op = LiteFrameReadLance(
            path=path,
            dtypes=pd.Series(dtype=object),
            columns=columns,
            version=version,
            asof=asof,
            filters=filters,
            with_row_address=with_row_address,
            storage_options=storage_options,
            chunk_bytes=chunk_bytes,
            merge_small_files=merge_small_files,
            read_kwargs=kwargs or None,
            read_stage="get_dtypes",
        )
        result = json.loads(
            dt_op._new_liteframe_from_source()
            .execute(session=session)
            .fetch(session=session)
        )
        # Capture the dataset's physical (original-case) column names for
        # the predicate-pushdown pusher to map logical -> physical names.
        physical_columns = [d["key"] for d in result]
        dtypes = pd.Series(
            {
                d["key"]: wrap_arrow_dtype(arrow_type_from_str(d["value"]))
                for d in result
            }
        )
        dtypes = normalize_dtypes_index(dtypes)
        if columns:
            dtypes = dtypes[dtypes.index.isin([c.lower() for c in columns])]

    if with_row_address and not dtypes.empty:
        # Lance synthesizes the virtual ``_rowaddr`` (uint64) on read even
        # though it is absent from the dataset schema, so include it in the
        # planned physical dtypes. Otherwise the tiled width would be one
        # short of the executed width (the local test-session shape check
        # catches this) and the merge tiler could not index
        # ``_physical_dtypes["_rowaddr"]``. It stays out of user-visible
        # columns via ``hidden_columns`` below.
        dtypes = pd.Series({**dtypes, "_rowaddr": wrap_arrow_dtype(pa.uint64())})

    hidden_columns = ["_rowaddr"] if with_row_address else None

    op = LiteFrameReadLance(
        path=path,
        dtypes=dtypes,
        columns=columns,
        version=version,
        asof=asof,
        filters=filters,
        with_row_address=with_row_address,
        hidden_columns=hidden_columns,
        physical_columns=physical_columns,
        storage_options=storage_options,
        chunk_bytes=chunk_bytes,
        merge_small_files=merge_small_files,
        read_kwargs=kwargs or None,
    )
    return op._new_liteframe_from_source()
