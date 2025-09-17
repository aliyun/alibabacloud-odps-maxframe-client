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
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    Int32Field,
    Int64Field,
    ListField,
    StringField,
)
from ..utils import parse_index
from .core import DataFrameDataStore


class DataFrameToCSV(DataFrameDataStore):
    _op_type_ = opcodes.TO_CSV

    path = AnyField("path", default=None)
    sep = StringField("sep", default=None)
    na_rep = StringField("na_rep", default=None)
    float_format = StringField("float_format", default=None)
    columns = ListField("columns", default=None)
    header = AnyField("header", default=None)
    index = BoolField("index", default=None)
    index_label = AnyField("index_label", default=None)
    mode = StringField("mode", default=None)
    encoding = StringField("encoding", default=None)
    compression = AnyField("compression", default=None)
    quoting = Int32Field("quoting", default=None)
    quotechar = StringField("quotechar", default=None)
    line_terminator = StringField("line_terminator", default=None)
    chunksize = Int64Field("chunksize", default=None)
    date_format = StringField("date_format", default=None)
    doublequote = BoolField("doublequote", default=None)
    escapechar = StringField("escapechar", default=None)
    decimal = StringField("decimal", default=None)
    storage_options = DictField("storage_options", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @property
    def one_file(self):
        # if wildcard in path, write csv into multiple files
        return "*" not in self.path

    def __call__(self, df):
        index_value = parse_index(df.index_value.to_pandas()[:0], df)
        if df.ndim == 2:
            columns_value = parse_index(
                df.columns_value.to_pandas()[:0], store_data=True
            )
            return self.new_dataframe(
                [df],
                shape=(0, 0),
                dtypes=df.dtypes[:0],
                index_value=index_value,
                columns_value=columns_value,
            )
        else:
            return self.new_series(
                [df], shape=(0,), dtype=df.dtype, index_value=index_value
            )


def to_csv(
    df,
    path,
    sep=",",
    na_rep="",
    float_format=None,
    columns=None,
    header=True,
    index=True,
    index_label=None,
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
    storage_options=None,
    **kw,
):
    r"""
    Write object to a comma-separated values (csv) file.

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
    index : bool, default True
        Write row names (index).
    index_label : str or sequence, or False, default None
        Column label for index column(s) if desired. If None is given, and
        `header` and `index` are True, then the index names are used. A
        sequence should be given if the object uses MultiIndex. If
        False do not print fields for index names. Use index_label=False
        for easier importing in R.
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
    Returns
    -------
    None or str
        If path_or_buf is None, returns the resulting csv format as a
        string. Otherwise returns None.

    See Also
    --------
    read_csv : Load a CSV file into a DataFrame.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'name': ['Raphael', 'Donatello'],
    ...                    'mask': ['red', 'purple'],
    ...                    'weapon': ['sai', 'bo staff']})
    >>> df.to_csv('out.csv', index=False).execute()
    """
    lineterminator = lineterminator or kw.pop("line_terminator", None)
    if kw:
        raise TypeError(
            f"to_csv() got an unexpected keyword argument '{next(iter(kw))}'"
        )

    if mode != "w":  # pragma: no cover
        raise NotImplementedError("only support to_csv with mode 'w' for now")
    op = DataFrameToCSV(
        path=path,
        sep=sep,
        na_rep=na_rep,
        float_format=float_format,
        columns=columns,
        header=header,
        index=index,
        index_label=index_label,
        mode=mode,
        encoding=encoding,
        compression=compression,
        quoting=quoting,
        quotechar=quotechar,
        line_terminator=lineterminator,
        chunksize=chunksize,
        date_format=date_format,
        doublequote=doublequote,
        escapechar=escapechar,
        decimal=decimal,
        storage_options=storage_options,
    )
    return op(df)
