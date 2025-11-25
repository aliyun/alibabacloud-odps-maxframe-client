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

from ... import opcodes
from ...serialization.serializables import BoolField, DictField, StringField
from ..datasource.read_parquet import check_engine
from ..utils import parse_index
from .core import LakeDataStore


class DataFrameToParquet(LakeDataStore):
    _op_type_ = opcodes.TO_PARQUET

    engine = StringField("engine")
    index = BoolField("index")
    additional_kwargs = DictField("additional_kwargs")

    @classmethod
    def _get_path(cls, path, i):
        if "*" not in path:
            return path
        return path.replace("*", str(i))

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


def to_parquet(
    df,
    path,
    engine="auto",
    compression="snappy",
    index=None,
    partition_cols=None,
    storage_options: dict = None,
    **kwargs,
):
    """
    Write a DataFrame to the binary parquet format, each chunk will be
    written to a Parquet file.

    Parameters
    ----------
    path : str or file-like object
        If path is a string with wildcard e.g. '/to/path/out-*.parquet',
        `to_parquet` will try to write multiple files, for instance,
        chunk (0, 0) will write data into '/to/path/out-0.parquet'.
        If path is a string without wildcard, we will treat it as a directory.

    engine : {'auto', 'pyarrow', 'fastparquet'}, default 'auto'
        Parquet library to use. The default behavior is to try 'pyarrow',
        falling back to 'fastparquet' if 'pyarrow' is unavailable.

    compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.

    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output.
        If ``False``, they will not be written to the file.
        If ``None``, similar to ``True`` the dataframe's index(es)
        will be saved. However, instead of being saved as values,
        the RangeIndex will be stored as a range in the metadata so it
        doesn't require much space and is faster. Other indexes will
        be included as columns in the file output.

    partition_cols : list, optional, default None
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
        Must be None if path is not a string.

    **kwargs
        Additional arguments passed to the parquet library.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    >>> df.to_parquet('*.parquet.gzip',
    ...               compression='gzip').execute()  # doctest: +SKIP
    >>> md.read_parquet('*.parquet.gzip').execute()  # doctest: +SKIP
       col1  col2
    0     1     3
    1     2     4

    >>> import io
    >>> f = io.BytesIO()
    >>> df.to_parquet(f).execute()
    >>> f.seek(0)
    0
    >>> content = f.read()
    """
    engine = check_engine(engine)
    op = DataFrameToParquet(
        path=path,
        engine=engine,
        compression=compression,
        index=index,
        partition_cols=partition_cols,
        storage_options=storage_options,
        additional_kwargs=kwargs,
    )
    return op(df)
