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

from maxframe.dataframe.datasource.read_parquet import check_engine
from maxframe.liteframe.operators.core import LiteFrameDataFormat
from maxframe.liteframe.operators.datastore import LiteFrameLakeDataStore


def to_parquet(
    lf,
    path,
    engine="auto",
    compression="snappy",
    partition_cols=None,
    storage_options: dict = None,
    **kwargs,
):
    """
    Write a LiteFrame to the binary parquet format, each chunk will be
    written to a Parquet file.

    Simplified version for LiteFrame with no index semantics.

    Parameters
    ----------
    path : str or file-like object
        If path is a string with wildcard e.g. '/to/path/out-*.parquet',
        `to_parquet` will try to write multiple files, for instance,
        chunk (0, 0) will write data into '/to/path/out-0.parquet'.
        If path is a string without wildcard and partition_cols is None,
        all data will be written into a single file.
        If path is a string with wildcard or partition_cols is not None,
        we will treat it as a directory.

    engine : {'auto', 'pyarrow'}, default 'auto'
        Parquet library to use. Only 'pyarrow' is currently supported.

    compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.

    partition_cols : list, optional, default None
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
        Must be None if path is not a string.

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection.

    **kwargs
        Additional arguments passed to the parquet library.

    Returns
    -------
    None or LiteFrame
        Returns a stub LiteFrame for execution.

    See Also
    --------
    read_parquet : Read a Parquet file into a LiteFrame.
    to_csv : Write a LiteFrame to a CSV file.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    >>> lf.to_parquet('*.parquet.gzip',
    ...               compression='gzip').execute()  # doctest: +SKIP
    >>> mlf.read_parquet('*.parquet.gzip').execute()  # doctest: +SKIP
       col1  col2
    0     1     3
    1     2     4
    """
    engine = check_engine(engine)

    op = LiteFrameLakeDataStore(
        path=path,
        data_format=LiteFrameDataFormat.PARQUET,
        compression=compression,
        partition_cols=partition_cols,
        storage_options=storage_options,
        write_kwargs={
            "engine": engine,
            **kwargs,
        }
        or None,
    )
    return op(lf)
