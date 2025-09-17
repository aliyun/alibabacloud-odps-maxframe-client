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

from numbers import Integral

from .iloc import DataFrameIloc


class DataFrameIat:
    def __init__(self, obj):
        self._obj = obj
        self._iloc = DataFrameIloc(self._obj)

    def __getitem__(self, indexes):
        if not isinstance(indexes, tuple):
            indexes = (indexes,)

        for index in indexes:
            if not isinstance(index, Integral):
                raise ValueError("Invalid call for scalar access (getting)!")

        return self._iloc[indexes]


def iat(a):
    """
    Access a single value for a row/column pair by integer position.

    Similar to ``iloc``, in that both provide integer-based lookups. Use
    ``iat`` if you only need to get or set a single value in a DataFrame
    or Series.

    Raises
    ------
    IndexError
        When integer position is out of bounds.

    See Also
    --------
    DataFrame.at : Access a single value for a row/column label pair.
    DataFrame.loc : Access a group of rows and columns by label(s).
    DataFrame.iloc : Access a group of rows and columns by integer position(s).

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],
    ...                   columns=['A', 'B', 'C'])
    >>> df.execute()
        A   B   C
    0   0   2   3
    1   0   4   1
    2  10  20  30

    Get value at specified row/column pair

    >>> df.iat[1, 2].execute()
    1

    Set value at specified row/column pair

    >>> df.iat[1, 2] = 10
    >>> df.iat[1, 2].execute()
    10

    Get value within a series

    >>> df.loc[0].iat[1].execute()
    2
    """
    return DataFrameIat(a)
