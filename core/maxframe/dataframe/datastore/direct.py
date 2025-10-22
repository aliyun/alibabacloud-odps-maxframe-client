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

from ...utils import pd_release_version

_to_dict_has_index = pd_release_version[0] >= 2


def df_to_dict(
    df, orient="dict", into=dict, index=True, batch_size=10000, session=None
):
    """
    Convert the DataFrame to a dictionary.

    The type of the key-value pairs can be customized with the parameters
    (see below).

    Parameters
    ----------
    orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
        Determines the type of the values of the dictionary.

        - 'dict' (default) : dict like {column -> {index -> value}}
        - 'list' : dict like {column -> [values]}
        - 'series' : dict like {column -> Series(values)}
        - 'split' : dict like
          {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
        - 'tight' : dict like
          {'index' -> [index], 'columns' -> [columns], 'data' -> [values],
          'index_names' -> [index.names], 'column_names' -> [column.names]}
        - 'records' : list like
          [{column -> value}, ... , {column -> value}]
        - 'index' : dict like {index -> {column -> value}}

    into : class, default dict
        The collections.abc.MutableMapping subclass used for all Mappings
        in the return value.  Can be the actual class or an empty
        instance of the mapping type you want.  If you want a
        collections.defaultdict, you must pass it initialized.

    index : bool, default True
        Whether to include the index item (and index_names item if `orient`
        is 'tight') in the returned dictionary. Can only be ``False``
        when `orient` is 'split' or 'tight'.

    Returns
    -------
    dict, list or collections.abc.MutableMapping
        Return a collections.abc.MutableMapping object representing the
        DataFrame. The resulting transformation depends on the `orient`
        parameter.

    See Also
    --------
    DataFrame.from_dict: Create a DataFrame from a dictionary.
    DataFrame.to_json: Convert a DataFrame to JSON format.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'col1': [1, 2],
    ...                    'col2': [0.5, 0.75]},
    ...                   index=['row1', 'row2'])
    >>> df.execute()
          col1  col2
    row1     1  0.50
    row2     2  0.75
    >>> df.to_dict()
    {'col1': {'row1': 1, 'row2': 2}, 'col2': {'row1': 0.5, 'row2': 0.75}}

    You can specify the return orientation.

    >>> df.to_dict('series')
    {'col1': row1    1
             row2    2
    Name: col1, dtype: int64,
    'col2': row1    0.50
            row2    0.75
    Name: col2, dtype: float64}

    >>> df.to_dict('split')
    {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'],
     'data': [[1, 0.5], [2, 0.75]]}

    >>> df.to_dict('records')
    [{'col1': 1, 'col2': 0.5}, {'col1': 2, 'col2': 0.75}]

    >>> df.to_dict('index')
    {'row1': {'col1': 1, 'col2': 0.5}, 'row2': {'col1': 2, 'col2': 0.75}}

    >>> df.to_dict('tight')
    {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'],
     'data': [[1, 0.5], [2, 0.75]], 'index_names': [None], 'column_names': [None]}

    You can also specify the mapping type.

    >>> from collections import OrderedDict, defaultdict
    >>> df.to_dict(into=OrderedDict)
    OrderedDict([('col1', OrderedDict([('row1', 1), ('row2', 2)])),
                 ('col2', OrderedDict([('row1', 0.5), ('row2', 0.75)]))])

    If you want a `defaultdict`, you need to initialize it:

    >>> dd = defaultdict(list)
    >>> df.to_dict('records', into=dd)
    [defaultdict(<class 'list'>, {'col1': 1, 'col2': 0.5}),
     defaultdict(<class 'list'>, {'col1': 2, 'col2': 0.75})]
    """
    fetch_kwargs = dict(batch_size=batch_size)
    to_dict_kw = dict(orient=orient, into=into)
    if _to_dict_has_index:
        to_dict_kw["index"] = index
    return df.to_pandas(session=session, fetch_kwargs=fetch_kwargs).to_dict(
        **to_dict_kw
    )


def series_to_dict(series, into=dict, batch_size=10000, session=None):
    """
    Convert Series to {label -> value} dict or dict-like object.

    Parameters
    ----------
    into : class, default dict
        The collections.abc.Mapping subclass to use as the return
        object. Can be the actual class or an empty
        instance of the mapping type you want.  If you want a
        collections.defaultdict, you must pass it initialized.

    Returns
    -------
    collections.abc.Mapping
        Key-value representation of Series.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series([1, 2, 3, 4])
    >>> s.to_dict()
    {0: 1, 1: 2, 2: 3, 3: 4}
    >>> from collections import OrderedDict, defaultdict
    >>> s.to_dict(OrderedDict)
    OrderedDict([(0, 1), (1, 2), (2, 3), (3, 4)])
    >>> dd = defaultdict(list)
    >>> s.to_dict(dd)
    defaultdict(<class 'list'>, {0: 1, 1: 2, 2: 3, 3: 4})
    """
    fetch_kwargs = dict(batch_size=batch_size)
    return series.to_pandas(session=session, fetch_kwargs=fetch_kwargs).to_dict(
        into=into
    )


def series_to_list(series, batch_size=10000, session=None):
    """
    Return a list of the values.

    These are each a scalar type, which is a Python scalar
    (for str, int, float) or a pandas scalar
    (for Timestamp/Timedelta/Interval/Period)

    Returns
    -------
    list

    See Also
    --------
    numpy.ndarray.tolist : Return the array as an a.ndim-levels deep
        nested list of Python scalars.

    Examples
    --------
    For Series

    >>> import maxframe.dataframe as md
    >>> s = md.Series([1, 2, 3])
    >>> s.to_list()
    [1, 2, 3]

    For Index:

    >>> idx = md.Index([1, 2, 3])
    >>> idx.execute()
    Index([1, 2, 3], dtype='int64')

    >>> idx.to_list()
    [1, 2, 3]
    """
    fetch_kwargs = dict(batch_size=batch_size)
    return series.to_pandas(session=session, fetch_kwargs=fetch_kwargs).to_list()


def to_clipboard(
    obj, *, excel=True, sep=None, batch_size=10000, session=None, **kwargs
):
    """
    Copy object to the system clipboard.

    Write a text representation of object to the system clipboard.
    This can be pasted into Excel, for example.

    Parameters
    ----------
    excel : bool, default True
        Produce output in a csv format for easy pasting into excel.

        - True, use the provided separator for csv pasting.
        - False, write a string representation of the object to the clipboard.

    sep : str, default ``'\t'``
        Field delimiter.
    **kwargs
        These parameters will be passed to DataFrame.to_csv.

    See Also
    --------
    DataFrame.to_csv : Write a DataFrame to a comma-separated values
        (csv) file.
    read_clipboard : Read text from clipboard and pass to read_csv.

    Notes
    -----
    Requirements for your platform.

      - Linux : `xclip`, or `xsel` (with `PyQt4` modules)
      - Windows : none
      - macOS : none

    This method uses the processes developed for the package `pyperclip`. A
    solution to render any output string format is given in the examples.

    Examples
    --------
    Copy the contents of a DataFrame to the clipboard.

    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])

    >>> df.to_clipboard(sep=',')  # doctest: +SKIP
    ... # Wrote the following to the system clipboard:
    ... # ,A,B,C
    ... # 0,1,2,3
    ... # 1,4,5,6

    We can omit the index by passing the keyword `index` and setting
    it to false.

    >>> df.to_clipboard(sep=',', index=False)  # doctest: +SKIP
    ... # Wrote the following to the system clipboard:
    ... # A,B,C
    ... # 1,2,3
    ... # 4,5,6
    """
    fetch_kwargs = dict(batch_size=batch_size)
    return obj.to_pandas(session=session, fetch_kwargs=fetch_kwargs).to_clipboard(
        excel=excel, sep=sep, **kwargs
    )
