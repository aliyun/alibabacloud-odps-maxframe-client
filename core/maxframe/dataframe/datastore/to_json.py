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

from typing import Any, Dict, Optional, Union

from ... import opcodes
from ...serialization.serializables import AnyField, BoolField, DictField, StringField
from ..utils import parse_index
from .core import LakeDataStore


class DataFrameToJSON(LakeDataStore):
    _op_type_ = opcodes.TO_JSON

    orient = StringField("orient", default=None)
    default_handler = AnyField("default_handler", default=None)
    lines = BoolField("lines", default=None)
    index = BoolField("index", default=None)
    json_kwargs = DictField("json_kwargs", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @property
    def one_file(self):
        # if wildcard in path, write json into multiple files
        return "*" not in self.path and not self.partition_cols

    def __call__(self, df):
        if self.one_file and (self.orient != "records" or not self.lines):
            raise ValueError(
                "You can only fix one single output file when orient='records' and lines is True"
            )
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


def to_json(
    df,
    path: Optional[str] = None,
    orient: Optional[str] = None,
    date_format: Optional[str] = None,
    double_precision: int = 10,
    force_ascii: bool = True,
    date_unit: Optional[str] = "ms",
    default_handler: Optional[callable] = None,
    lines: bool = False,
    compression: Union[str, Dict[str, Any], None] = "infer",
    index: Optional[bool] = None,
    indent: Optional[int] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    partition_cols: Optional[Union[str, list]] = None,
    **kwargs,
):
    r"""
    Convert the object to a JSON string.

    Note NaN's and None will be converted to null and datetime objects
    will be converted to UNIX timestamps.

    Parameters
    ----------
    path : str, path object, file-like object, or None, default None
        String, path object (implementing os.PathLike[str]), or file-like
        object implementing a write() function. If None, the result is
        returned as a string.
    orient : str
        Indication of expected JSON string format.

        * Series:

            - default is 'index'
            - allowed values are: {'split', 'records', 'index', 'table'}.

        * DataFrame:

            - default is 'columns'
            - allowed values are: {'split', 'records', 'index', 'columns',
              'values', 'table'}.

        * The format of the JSON string:

            - 'split' : dict like {'index' -> [index], 'columns' -> [columns],
              'data' -> [values]}
            - 'records' : list like [{column -> value}, ... , {column -> value}]
            - 'index' : dict like {index -> {column -> value}}
            - 'columns' : dict like {column -> {index -> value}}
            - 'values' : just the values array
            - 'table' : dict like {'schema': {schema}, 'data': {data}}

            Describing the data, where data component is like ``orient='records'``.

    date_format : {None, 'epoch', 'iso'}
        Type of date conversion. 'epoch' = epoch milliseconds,
        'iso' = ISO8601. The default depends on the `orient`. For
        ``orient='table'``, the default is 'iso'. For all other orients,
        the default is 'epoch'.
    double_precision : int, default 10
        The number of decimal places to use when encoding
        floating point numbers.
    force_ascii : bool, default True
        Force encoded string to be ASCII.
    date_unit : str, default 'ms' (milliseconds)
        The time unit to encode to, governs timestamp and ISO8601
        precision.  One of 's', 'ms', 'us', 'ns' for second, millisecond,
        microsecond, and nanosecond respectively.
    default_handler : callable, default None
        Handler to call if object cannot otherwise be converted to a
        suitable format for JSON. Should receive a single argument which is
        the object to convert and return a serializable object.
    lines : bool, default False
        If 'orient' is 'records' write out line-delimited json format. Will
        throw ValueError if incorrect 'orient' is used.
    compression : str or dict, default 'infer'
        For on-the-fly compression of the output data. If str, represents
        compression mode. If dict, value at 'method' is the compression mode.
        Compression mode may be any of the following possible
        values: {'infer', 'gzip', 'bz2', 'zip', 'xz', None}. If compression
        mode is 'infer' and `path_or_buf` is path-like, then detect
        compression mode from the following extensions: '.gz', '.bz2',
        '.zip' or '.xz'. (otherwise no compression). If dict given and
        mode is one of {'zip', 'xz'}, other entries passed as
        additional compression options.
    index : bool, default None
        Whether to include the index values in the JSON string. Not
        including the index (``index=False``) is only supported when
        orient is 'split' or 'table'.
    indent : int, optional
       Length of whitespace used to indent each record.
    partition_cols : list, optional, default None
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.

    See Also
    --------
    read_json : Convert a JSON string to pandas object.

    Notes
    -----
    The behavior of ``indent=0`` varies from the stdlib, which does not
    indent the output but does insert newlines. Currently, ``indent=0``
    and the default ``indent=None`` are equivalent in pandas, though this
    may change in a future release.

    ``orient='table'`` contains a 'pandas_version' field under 'schema'.
    This stores the version of `pandas` used in the latest revision of the
    schema.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([['a', 'b'], ['c', 'd']],
    ...                   index=['row 1', 'row 2'],
    ...                   columns=['col 1', 'col 2'])
    >>> df.to_json('data.json')
    >>> # Writing to a file with orient='records'
    >>> df.to_json('records.json', orient='records')  # doctest: +SKIP
    >>> # Writing in line-delimited json format
    >>> df.to_json('ldjson.json', orient='records', lines=True)  # doctest: +SKIP
    >>> # Write partitioned dataset
    >>> df.to_json('dataset', partition_cols=['col 1'])  # doctest: +SKIP
    """

    if kwargs:
        raise TypeError(
            f"to_json() got an unexpected keyword argument '{next(iter(kwargs))}'"
        )

    if path is None:
        raise NotImplementedError("Currently only support to_json with path specified")

    json_kwargs = dict(
        date_format=date_format,
        double_precision=double_precision,
        force_ascii=force_ascii,
        date_unit=date_unit,
        indent=indent,
    )
    op = DataFrameToJSON(
        path=path,
        orient=orient,
        default_handler=default_handler,
        lines=lines,
        compression=compression,
        index=index,
        storage_options=storage_options,
        partition_cols=partition_cols,
        json_kwargs=json_kwargs,
    )
    return op(df)
