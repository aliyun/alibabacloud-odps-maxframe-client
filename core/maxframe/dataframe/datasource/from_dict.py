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

from ...core import ENTITY_TYPE
from ...utils import find_objects, no_default
from ..utils import validate_axis


def dataframe_from_dict(data, orient="columns", dtype=None, columns=None):
    """
    Construct DataFrame from dict of array-like or dicts.

    Creates DataFrame object from dictionary by columns or by index
    allowing dtype specification.

    Parameters
    ----------
    data : dict
        Of the form {field : array-like} or {field : dict}.
    orient : {'columns', 'index', 'tight'}, default 'columns'
        The "orientation" of the data. If the keys of the passed dict
        should be the columns of the resulting DataFrame, pass 'columns'
        (default). Otherwise if the keys should be rows, pass 'index'.
        If 'tight', assume a dict with keys ['index', 'columns', 'data',
        'index_names', 'column_names'].

    dtype : dtype, default None
        Data type to force after DataFrame construction, otherwise infer.
    columns : list, default None
        Column labels to use when ``orient='index'``. Raises a ValueError
        if used with ``orient='columns'`` or ``orient='tight'``.

    Returns
    -------
    DataFrame

    See Also
    --------
    DataFrame.from_records : DataFrame from structured ndarray, sequence
        of tuples or dicts, or DataFrame.
    DataFrame : DataFrame object creation using constructor.
    DataFrame.to_dict : Convert the DataFrame to a dictionary.

    Examples
    --------
    By default the keys of the dict become the DataFrame columns:

    >>> import maxframe.dataframe as md
    >>> data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}
    >>> md.DataFrame.from_dict(data).execute()
       col_1 col_2
    0      3     a
    1      2     b
    2      1     c
    3      0     d

    Specify ``orient='index'`` to create the DataFrame using dictionary
    keys as rows:

    >>> data = {'row_1': [3, 2, 1, 0], 'row_2': ['a', 'b', 'c', 'd']}
    >>> md.DataFrame.from_dict(data, orient='index').execute()
           0  1  2  3
    row_1  3  2  1  0
    row_2  a  b  c  d

    When using the 'index' orientation, the column names can be
    specified manually:

    >>> md.DataFrame.from_dict(data, orient='index',
    ...                        columns=['A', 'B', 'C', 'D']).execute()
           A  B  C  D
    row_1  3  2  1  0
    row_2  a  b  c  d

    Specify ``orient='tight'`` to create the DataFrame using a 'tight'
    format:

    >>> data = {'index': [('a', 'b'), ('a', 'c')],
    ...         'columns': [('x', 1), ('y', 2)],
    ...         'data': [[1, 3], [2, 4]],
    ...         'index_names': ['n1', 'n2'],
    ...         'column_names': ['z1', 'z2']}
    >>> md.DataFrame.from_dict(data, orient='tight').execute()
    z1     x  y
    z2     1  2
    n1 n2
    a  b   1  3
       c   2  4
    """
    from ..initializer import DataFrame as DataFrameInit
    from .from_tensor import dataframe_from_1d_tileables

    if orient != "tight" and not find_objects(data, ENTITY_TYPE):
        res = DataFrameInit(data)
    elif orient == "tight":
        # init directly
        init_kw = {
            "index": data.get("index"),
            "columns": data.get("columns"),
        }
        df = DataFrameInit(data["data"], **init_kw)
        rename_kw = {
            "index": data.get("index_names", no_default),
            "columns": data.get("column_names", no_default),
        }
        res = df.rename_axis(**rename_kw)
    else:
        axis = validate_axis(orient)
        res = dataframe_from_1d_tileables(data, columns=columns, axis=axis)

    if dtype is not None:
        res = res.astype(dtype)
    return res
