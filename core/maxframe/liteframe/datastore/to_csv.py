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

from maxframe.liteframe.operators.core import LiteFrameDataFormat
from maxframe.liteframe.operators.datastore import LiteFrameLakeDataStore
from maxframe.utils import check_unexpected_kwargs


def to_csv(
    lf,
    path,
    sep=",",
    na_rep="",
    float_format=None,
    columns=None,
    header=True,
    mode="w",
    encoding=None,
    compression="infer",
    quoting=None,
    quotechar='"',
    lineterminator=None,
    chunksize=None,
    date_format=None,
    doublequote=True,
    escapechar=None,
    decimal=".",
    partition_cols=None,
    storage_options=None,
    **kw,
):
    r"""
    Write LiteFrame to a comma-separated values (csv) file.

    Simplified version for LiteFrame with no index semantics.

    Parameters
    ----------
    path : str
        File path.
        If path is a string with wildcard e.g. '/to/path/out-*.csv',
        to_csv will try to write multiple files, for instance,
        chunk (0, 0) will write data into '/to/path/out-0.csv'.
        If path is a string without wildcard,
        all data will be written into a single file.
    sep : str, default ','
        String of length 1. Field delimiter for the output file.
    na_rep : str, default ''
        Missing data representation.
    float_format : str, default None
        Format string for floating point numbers.
    columns : sequence, optional
        Columns to write.
    header : bool or list of str, default True
        Write out the column names. If a list of strings is given it is
        assumed to be aliases for the column names.
    mode : str
        Python write mode, default 'w'.
    encoding : str, optional
        A string representing the encoding to use in the output file,
        defaults to 'utf-8'.
    compression : str or dict, default 'infer'
        If str, represents compression mode. If dict, value at 'method' is
        the compression mode. Compression mode may be any of the following
        possible values: {'infer', 'gzip', 'bz2', 'zip', 'xz', None}. If
        compression mode is 'infer' and `path_or_buf` is path-like, then
        detect compression mode from the following extensions: '.gz',
        '.bz2', '.zip' or '.xz'. (otherwise no compression). If dict given
        and mode is 'zip' or inferred as 'zip', other entries passed as
        additional compression options.
    quoting : optional constant from csv module
        Defaults to csv.QUOTE_MINIMAL. If you have set a `float_format`
        then floats are converted to strings and thus csv.QUOTE_NONNUMERIC
        will treat them as non-numeric.
    quotechar : str, default '\"'
        String of length 1. Character used to quote fields.
    lineterminator : str, optional
        The newline character or character sequence to use in the output
        file. Defaults to `os.linesep`, which depends on the OS in which
        this method is called ('\n' for linux, '\r\n' for Windows, i.e.).
    chunksize : int or None
        Rows to write at a time.
    date_format : str, default None
        Format string for datetime objects.
    doublequote : bool, default True
        Control quoting of `quotechar` inside a field.
    escapechar : str, default None
        String of length 1. Character used to escape `sep` and `quotechar`
        when appropriate.
    decimal : str, default '.'
        Character recognized as decimal separator. E.g. use ',' for
        European data.
    partition_cols : list, optional, default None
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection.

    Returns
    -------
    LiteFrame
        Returns a stub LiteFrame for execution.

    See Also
    --------
    read_csv : Load a CSV file into a LiteFrame.
    to_parquet : Write a LiteFrame to a Parquet file.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'name': ['Raphael', 'Donatello'],
    ...                     'mask': ['red', 'purple'],
    ...                     'weapon': ['sai', 'bo staff']})
    >>> lf.to_csv('out.csv').execute()  # doctest: +SKIP
    >>> # Write partitioned dataset
    >>> lf.to_csv('dataset', partition_cols=['mask']).execute()  # doctest: +SKIP
    """
    if lineterminator is None:
        lineterminator = kw.pop("line_terminator", None)
    else:
        kw.pop("line_terminator", None)
    check_unexpected_kwargs(kw)

    if mode != "w":
        raise NotImplementedError("only support to_csv with mode 'w' for now")

    op = LiteFrameLakeDataStore(
        path=path,
        data_format=LiteFrameDataFormat.CSV,
        compression=compression,
        partition_cols=partition_cols,
        storage_options=storage_options,
        write_kwargs={
            "sep": sep,
            "na_rep": na_rep,
            "columns": columns,
            "header": header,
            "mode": mode,
            "encoding": encoding,
            "chunksize": chunksize,
            "float_format": float_format,
            "quoting": quoting,
            "quotechar": quotechar,
            "line_terminator": lineterminator,
            "date_format": date_format,
            "doublequote": doublequote,
            "escapechar": escapechar,
            "decimal": decimal,
            **kw,
        }
        or None,
    )
    return op(lf)
