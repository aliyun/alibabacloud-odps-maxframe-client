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

from pandas.api.types import is_list_like

from ..utils import validate_axis


def xs(df_or_series, key, axis=0, level=None, drop_level=True):
    """
    Return cross-section from the Series/DataFrame.

    This method takes a `key` argument to select data at a particular
    level of a MultiIndex.

    Parameters
    ----------
    key : label or tuple of label
        Label contained in the index, or partially in a MultiIndex.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Axis to retrieve cross-section on.
    level : object, defaults to first n levels (n=1 or len(key))
        In case of a key partially contained in a MultiIndex, indicate
        which levels are used. Levels can be referred by label or position.
    drop_level : bool, default True
        If False, returns object with same levels as self.

    Returns
    -------
    Series or DataFrame
        Cross-section from the original Series or DataFrame
        corresponding to the selected index levels.

    See Also
    --------
    DataFrame.loc : Access a group of rows and columns
        by label(s) or a boolean array.
    DataFrame.iloc : Purely integer-location based indexing
        for selection by position.

    Notes
    -----
    `xs` can not be used to set values.

    MultiIndex Slicers is a generic way to get/set values on
    any level or levels.
    It is a superset of `xs` functionality, see
    :ref:`MultiIndex Slicers <advanced.mi_slicers>`.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> d = {'num_legs': [4, 4, 2, 2],
    ...      'num_wings': [0, 0, 2, 2],
    ...      'class': ['mammal', 'mammal', 'mammal', 'bird'],
    ...      'animal': ['cat', 'dog', 'bat', 'penguin'],
    ...      'locomotion': ['walks', 'walks', 'flies', 'walks']}
    >>> df = md.DataFrame(data=d)
    >>> df = df.set_index(['class', 'animal', 'locomotion'])
    >>> df.execute()
                                num_legs  num_wings
    class  animal  locomotion
    mammal cat     walks              4          0
            dog     walks              4          0
            bat     flies              2          2
    bird   penguin walks              2          2

    Get values at specified index

    >>> df.xs('mammal').execute()
                        num_legs  num_wings
    animal locomotion
    cat    walks              4          0
    dog    walks              4          0
    bat    flies              2          2

    Get values at several indexes

    >>> df.xs(('mammal', 'dog')).execute()
                num_legs  num_wings
    locomotion
    walks              4          0

    Get values at specified index and level

    >>> df.xs('cat', level=1).execute()
                        num_legs  num_wings
    class  locomotion
    mammal walks              4          0

    Get values at several indexes and levels

    >>> df.xs(('bird', 'walks'),
    ...       level=[0, 'locomotion']).execute()
                num_legs  num_wings
    animal
    penguin         2          2

    Get values at specified column and axis

    >>> df.xs('num_wings', axis=1).execute()
    class   animal   locomotion
    mammal  cat      walks         0
            dog      walks         0
            bat      flies         2
    bird    penguin  walks         2
    Name: num_wings, dtype: int64
    """
    axis = validate_axis(axis, df_or_series)
    if level is None:
        level = range(df_or_series.axes[axis].nlevels)
    elif not is_list_like(level):
        level = [level]

    slc = [slice(None)] * df_or_series.axes[axis].nlevels
    if not is_list_like(key):
        key = (key,)

    level_set = set()
    for k, level_ in zip(key, level):
        slc[level_] = k
        level_set.add(level_)
    left_levels = set(range(df_or_series.axes[axis].nlevels)) - level_set

    if len(slc) > 1:
        slc = tuple(slc)

    res = df_or_series.loc(axis=axis)[slc]
    if drop_level:
        if len(left_levels) == 0:
            if res.ndim > 1:
                res = res.iloc[0, :] if axis == 0 else res.iloc[:, 0]
            else:
                res = res.iloc[0]
        else:
            res = res.droplevel(list(level_set), axis=axis)
    return res
