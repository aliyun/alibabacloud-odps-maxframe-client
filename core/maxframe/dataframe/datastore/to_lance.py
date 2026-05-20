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

from maxframe import opcodes
from maxframe.dataframe.datastore.core import LakeDataStore
from maxframe.dataframe.utils import parse_index
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    StringField,
)


class DataFrameToLance(LakeDataStore):
    _op_type_ = opcodes.TO_LANCE

    mode = StringField("mode", default=None)
    index = BoolField("index", default=None)
    index_label = AnyField("index_label", default=None)
    schema = AnyField("schema", default=None)  # PyArrow schema for column customization
    lance_kwargs = DictField("lance_kwargs", default=None)

    def __call__(self, df):
        index_value = parse_index(df.index_value.to_pandas()[:0], df)
        columns_value = parse_index(df.columns_value.to_pandas()[:0], store_data=True)
        return self.new_dataframe(
            [df],
            shape=(0, 0),
            dtypes=df.dtypes[:0],
            index_value=index_value,
            columns_value=columns_value,
        )


def to_lance(
    df,
    path,
    mode: str = "create",
    index: bool = None,
    index_label=None,
    schema=None,
    storage_options: dict = None,
    **kwargs,
):
    """
    Write a DataFrame to a Lance dataset.

    Lance is a columnar data format optimized for ML workloads and
    vector search. Each chunk will be written as a fragment.

    Parameters
    ----------
    path : str
        Target path for the Lance dataset. Can be a local path, Aliyun OSS URL,
        or S3 URL.
        For Aliyun OSS URLs, the format is: ``oss://<endpoint>/<bucket>/<path>``.
        Example: ``oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset``.
        For S3 URLs, the format is: ``s3://<bucket>/<path>``.
    mode : {'create', 'append', 'overwrite'}, default 'create'
        How to handle existing data:
        - 'create': Create a new dataset (fails if exists)
        - 'append': Append to existing dataset
        - 'overwrite': Overwrite existing dataset
    index : bool, optional
        If True, write DataFrame index as a column. If False, do not write the
        index. If None (default), write the index only if it's not a simple
        RangeIndex (same as ``pa.Table.from_pandas`` default behavior).
    index_label : str or list of str, optional
        Column label(s) for the index column(s). If None (default) and `index`
        is True, the index names are used. Use this to rename the index column
        when writing (e.g., from '_idx_0' to 'id').
    schema : pyarrow.Schema, optional
        PyArrow schema to specify column types, compression, encoding, etc.
        Columns in this schema will override auto-detected types from DataFrame.
        Columns not specified will use types inferred from DataFrame.
    storage_options : dict, optional
        Options for storage connection.
        For Aliyun OSS with RAM role: ``{'role_arn': 'acs:ram::xxx:role/name'}``
    **kwargs
        Additional keyword arguments passed to lance.fragment.write_fragments().

    Returns
    -------
    DataFrame
        An empty DataFrame (write operation result).

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    >>> # Write to Aliyun OSS with RAM role
    >>> df.to_lance(
    ...     'oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset',
    ...     storage_options={'role_arn': 'acs:ram::1234567890:role/maxframe-oss'}
    ... ).execute()  # doctest: +SKIP
    >>> # Append mode
    >>> df.to_lance(
    ...     'oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset',
    ...     mode='append',
    ...     storage_options={'role_arn': 'acs:ram::1234567890:role/maxframe-oss'}
    ... ).execute()  # doctest: +SKIP
    >>> # Overwrite mode
    >>> df.to_lance(
    ...     'oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset',
    ...     mode='overwrite',
    ...     storage_options={'role_arn': 'acs:ram::1234567890:role/maxframe-oss'}
    ... ).execute()  # doctest: +SKIP
    """
    if mode not in ("create", "append", "overwrite"):
        raise ValueError(
            f"mode must be 'create', 'append', or 'overwrite', got '{mode}'"
        )

    # Normalize index_label to list
    if isinstance(index_label, str):
        index_label = [index_label]

    # Validate index_label length matches index levels
    if index_label is not None and len(index_label) != df.index.nlevels:
        raise ValueError(
            f"index_label needs {df.index.nlevels} labels "
            f"but only {len(index_label)} provided"
        )

    schema_bytes = None
    if schema is not None:
        schema_bytes = schema.serialize().to_pybytes()

    op = DataFrameToLance(
        path=path,
        mode=mode,
        index=index,
        index_label=index_label,
        schema=schema_bytes,
        storage_options=storage_options,
        lance_kwargs=kwargs,
    )
    return op(df)
