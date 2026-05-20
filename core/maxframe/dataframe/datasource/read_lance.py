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
from typing import MutableMapping, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from maxframe import opcodes
from maxframe.config import options
from maxframe.dataframe.datasource.core import (
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
    LakeDataSource,
)
from maxframe.dataframe.datasource.dataframe import from_pandas
from maxframe.dataframe.datasource.utils import get_lake_output_info
from maxframe.dataframe.operators import OutputType
from maxframe.dataframe.utils import (
    parse_index,
    to_arrow_dtypes,
    validate_default_index_type,
)
from maxframe.protocol import DefaultIndexType
from maxframe.serialization.serializables import (
    AnyField,
    DictField,
    Int64Field,
    ListField,
    StringField,
)
from maxframe.utils import convert_filters_to_sql, make_dtypes, no_default

try:
    import lance
except (ImportError, AttributeError):  # pragma: no cover
    # lance not installed or not compatible
    lance = None


class DataFrameReadLance(
    LakeDataSource,
    ColumnPruneSupportedDataSourceMixin,
    DtypeBackendCompatibleMixin,
):
    _op_type_ = opcodes.READ_LANCE

    columns = ListField("columns")
    version = Int64Field("version", default=None)
    asof = StringField("asof", default=None)
    filters = AnyField("filters", default=None)
    index_col = AnyField("index_col", default=None)  # int, str, sequence, or False
    read_kwargs = DictField("read_kwargs", default=None)

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
        cls, ctx: MutableMapping[str, Union[int, float]], op: "DataFrameReadLance"
    ):  # pragma: no cover
        # todo implement this to facilitate local computation
        ctx[op.outputs[0].key] = float("inf")

    @staticmethod
    def normalize_index_col(index_col, column_names):
        """
        Normalize index_col to list of column names.

        Returns list of str or None if no index specified.
        """
        if index_col is None or index_col is False:
            return None
        # Accept list/tuple; treat empty as None
        if isinstance(index_col, (list, tuple)):
            if len(index_col) == 0:
                return None
            return [column_names[c] if isinstance(c, int) else c for c in index_col]
        if isinstance(index_col, int):
            return [column_names[index_col]]
        if isinstance(index_col, str):
            return [index_col]
        raise TypeError("index_col must be int/str or a list/tuple of int/str")

    @staticmethod
    def extract_preserved_index_col_from_schema(schema):
        """
        Extract preserved index column names from pandas metadata in Arrow schema.

        Returns None if no preserved index (e.g., RangeIndex or no pandas metadata).
        """
        if not schema.metadata:
            return None
        pd_meta_bytes = schema.metadata.get(b"pandas")
        if not pd_meta_bytes:
            return None
        pd_meta = json.loads(pd_meta_bytes)
        idx_cols = pd_meta.get("index_columns", [])
        # RangeIndex is stored as a dict, not a column name string
        if not idx_cols or isinstance(idx_cols[0], dict):
            return None
        return idx_cols

    @classmethod
    def check_index_col_conflict(cls, index_col, preserved_index, column_names):
        """
        Check if user-specified index_col conflicts with preserved index.

        Raises ValueError if index_col conflicts with preserved_index.
        """
        if preserved_index is None:
            return
        if index_col is None or index_col is False:
            return

        normalized = cls.normalize_index_col(index_col, column_names)
        if normalized != preserved_index:
            raise ValueError(
                f"index_col={index_col!r} conflicts with preserved index "
                f"columns {preserved_index!r} from pandas metadata. "
                f"Use index_col=None to keep the preserved index, "
                f"or ensure index_col matches the preserved index columns."
            )

    @classmethod
    def apply_index_col_to_df(cls, df, index_col):
        """
        Apply index_col to a DataFrame.

        Returns DataFrame with index applied, or unchanged if index_col is None/False.
        """
        if index_col is None or index_col is False:
            return df
        idx_cols = cls.normalize_index_col(index_col, df.columns)
        return df.set_index(cls.compute_target_index(idx_cols, index_col))

    @classmethod
    def resolve_index_columns(cls, schema, index_col, columns=None):
        """
        Resolve preserved index, normalize index_col, and validate required columns.

        Returns (preserved_index_col, normalized_index_col).
        """
        preserved_index_col = cls.extract_preserved_index_col_from_schema(schema)
        schema_columns = [f.name for f in schema]
        cls.check_index_col_conflict(index_col, preserved_index_col, schema_columns)

        normalized_index_col = cls.normalize_index_col(index_col, schema_columns)
        required_idx_cols = normalized_index_col or preserved_index_col
        if columns and required_idx_cols:
            missing = set(required_idx_cols) - set(columns)
            if missing:
                raise ValueError(
                    f"index_col requires columns {sorted(missing)} but they are not in selected columns {columns}"
                )

        return preserved_index_col, normalized_index_col

    @staticmethod
    def compute_target_index(normalized_index_col, index_col):
        """Choose final target for set_index based on normalized/preserved index."""
        if normalized_index_col:
            return (
                normalized_index_col
                if len(normalized_index_col) > 1
                else normalized_index_col[0]
            )
        return index_col


def _read_lance_local(
    path: str,
    *,
    version: int,
    asof: str,
    columns: list,
    filters: str,
    index_col,
):
    if lance is None:
        raise ImportError(
            "Please install pylance which is required to read Lance datasets"
        )

    ds = lance.dataset(path, version=version, asof=asof)
    (
        preserved_index_col,
        normalized_index_col,
    ) = DataFrameReadLance.resolve_index_columns(ds.schema, index_col, columns)

    local_df = ds.to_table(
        columns=columns, filter=normalize_lance_filters(filters)
    ).to_pandas()
    # Apply index_col if user explicitly specified (not None and not False).
    # When index_col is None/False, Lance's to_pandas() already restores preserved index
    # from pandas metadata automatically (including names, levels, and structure).
    if index_col is not None and index_col is not False:
        if normalized_index_col != preserved_index_col:
            # apply user-specified index if it differs from preserved index
            local_df = DataFrameReadLance.apply_index_col_to_df(local_df, index_col)
    return local_df


def read_lance(
    path,
    version: int = None,
    asof: str = None,
    columns: list = None,
    filters: str = None,
    index_col=None,
    dtype_backend: str = no_default,
    default_index_type: Union[DefaultIndexType, str] = None,
    storage_options: dict = None,
    *,
    dtypes: pd.Series = None,
    index_dtypes: pd.Series = None,
    memory_scale: int = None,
    merge_small_files: bool = True,
    merge_small_file_options: dict = None,
    session=None,
    run_kwargs: dict = None,
    **kwargs,
):
    """
    Load a Lance dataset from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str
        Any valid string path is acceptable. The string could be a URL.
        For Aliyun OSS URLs, the format is: ``oss://<endpoint>/<bucket>/<path>``.
        Example: ``oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset``.
        For S3 URLs, the format is: ``s3://<bucket>/<path>``.
    version : int, optional
        Specific version of the dataset to read. If not specified,
        reads the latest version.
    asof : str, optional
        Timestamp for point-in-time read. Format: ISO 8601 datetime string.
        Cannot be used together with version.
    columns : list, optional
        If not None, only these columns will be read from the dataset.
    filters : str or list, optional
        Filter expression for predicate pushdown. Supports:
        - SQL-like filter strings accepted by Lance
        - CNF filters like ``[[('age', '>', 18), ('city', '==', 'Beijing')]]``
          used by MaxFrame predicate pushdown
        The alias ``filter=...`` is also accepted for compatibility.
    index_col : int, str, sequence of int/str, or False, default None
        Column(s) to use as the row labels of the DataFrame, either given as
        string name or column index. If a sequence of int/str is given, a
        MultiIndex is used. If False, no column is used as index (ignoring
        any pandas metadata in the dataset). If None, uses pandas metadata
        if available, otherwise falls back to default_index_type.
    default_index_type: {None, 'range', 'incremental'}, default None
        If index_col not specified, specify type of index to generate.
        If not specified, `options.dataframe.default_index_type` will be used.
    dtype_backend: {'numpy', 'pyarrow'}, default 'numpy'
        Back-end data type applied to the resultant DataFrame.
    storage_options: dict, optional
        Options for storage connection.
        For Aliyun OSS with RAM role: ``{'role_arn': 'acs:ram::xxx:role/name'}``
    memory_scale: int, optional
        Scale that real memory occupation divided with raw file size.
    merge_small_files: bool, default True
        Merge small Lance fragments into larger chunks for better
        parallel processing efficiency.
    merge_small_file_options: dict, optional
        Options for merging small files.
    **kwargs
        Any additional kwargs are passed to lance.

    Returns
    -------
    MaxFrame DataFrame

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> # Read from Aliyun OSS with RAM role
    >>> df = md.read_lance(
    ...     "oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset",
    ...     storage_options={"role_arn": "acs:ram::1234567890:role/maxframe-oss"}
    ... )
    >>> # Read specific version
    >>> df = md.read_lance(
    ...     "oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset",
    ...     version=1,
    ...     storage_options={"role_arn": "acs:ram::1234567890:role/maxframe-oss"}
    ... )
    >>> # Read with filter
    >>> df = md.read_lance(
    ...     "oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset",
    ...     filters="`age` > 18",
    ...     storage_options={"role_arn": "acs:ram::1234567890:role/maxframe-oss"}
    ... )
    """

    if version is not None and asof is not None:
        raise ValueError("Cannot specify both 'version' and 'asof' parameters")
    if filters is None:
        filters = kwargs.pop("filter", None)

    default_index_type = validate_default_index_type(default_index_type, **kwargs)

    # read_lance expects a single string path
    if not isinstance(path, str):
        raise TypeError("read_lance path must be a string")

    parsed_path = urlparse(path)
    local_test_mode = kwargs.pop("_local_test_mode", False)
    if not local_test_mode and (
        not parsed_path.scheme or parsed_path.scheme.lower() == "file"
    ):
        local_df = _read_lance_local(
            path=path,
            version=version,
            asof=asof,
            columns=columns,
            filters=filters,
            index_col=index_col,
        )
        return from_pandas(local_df)

    # Remote/engine path
    common_kwargs = dict(
        columns=columns,
        version=version,
        asof=asof,
        filters=filters,
        index_col=index_col,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
        read_kwargs=kwargs,
        memory_scale=memory_scale,
        merge_small_files=merge_small_files,
        merge_small_file_options=merge_small_file_options,
    )

    result = get_lake_output_info(
        DataFrameReadLance,
        path=path,
        default_index_type=default_index_type,
        session=session,
        run_kwargs=run_kwargs,
        dtype=dtypes,
        index_dtypes=index_dtypes,
        **common_kwargs,
    )

    dtypes = result.dtypes
    index_dtypes = result.index_dtypes
    index_value = result.index_value
    is_partitioned = result.is_partitioned

    if dtypes is not None:
        dtypes = make_dtypes(dtypes)
    if columns and dtypes is not None:
        dtypes = dtypes[columns]

    if dtype_backend is None:
        dtype_backend = options.dataframe.dtype_backend
    if dtype_backend == "pyarrow":
        dtypes = to_arrow_dtypes(dtypes)

    columns_value = parse_index(dtypes.index, store_data=True)
    # If index_dtypes exist (preserved index), don't use default_index_type to avoid standardize_range_index
    default_index_type = None if index_dtypes is not None else default_index_type
    op = DataFrameReadLance(
        path=path,
        columns=columns,
        version=version,
        asof=asof,
        filters=filters,
        index_col=index_col,
        dtype_backend=dtype_backend,
        read_kwargs=kwargs,
        default_index_type=default_index_type,
        storage_options=storage_options,
        is_partitioned=is_partitioned,
        memory_scale=memory_scale,
        merge_small_files=merge_small_files,
        merge_small_file_options=merge_small_file_options,
    )
    return op(index_value=index_value, columns_value=columns_value, dtypes=dtypes)


def normalize_lance_filters(filters):
    """Normalize read_lance filters to the SQL string Lance expects."""
    if filters is None or isinstance(filters, str):
        return filters
    if isinstance(filters, list):
        return convert_filters_to_sql(filters, errors="raise")
    return filters
