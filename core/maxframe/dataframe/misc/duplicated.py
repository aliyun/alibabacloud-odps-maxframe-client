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

import numpy as np

from ... import opcodes
from ...core import OutputType
from ._duplicate import BaseDuplicateOp, validate_subset


class DataFrameDuplicated(BaseDuplicateOp):
    _op_type_ = opcodes.DUPLICATED

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _get_shape(cls, input_shape, op):
        return (input_shape[0],)

    @classmethod
    def _gen_tileable_params(cls, op: "DataFrameDuplicated", input_params):
        # duplicated() always returns a Series
        return {
            "shape": cls._get_shape(input_params["shape"], op),
            "index_value": input_params["index_value"],
            "dtype": np.dtype(bool),
            "name": input_params.get("name"),
        }

    def __call__(self, inp, inplace=False):
        self._output_types = [OutputType.series]
        params = self._gen_tileable_params(self, inp.params)

        return self.new_tileable([inp], kws=[params])


def df_duplicated(df, subset=None, keep="first", method="auto"):
    """
    Return boolean Series denoting duplicate rows.

    Considering certain columns is optional.

    Parameters
    ----------
    subset : column label or sequence of labels, optional
        Only consider certain columns for identifying duplicates, by
        default use all of the columns.
    keep : {'first', 'last', False}, default 'first'
        Determines which duplicates (if any) to mark.

        - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
        - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
        - False : Mark all duplicates as ``True``.

    Returns
    -------
    Series
        Boolean series for each duplicated rows.

    See Also
    --------
    Index.duplicated : Equivalent method on index.
    Series.duplicated : Equivalent method on Series.
    Series.drop_duplicates : Remove duplicate values from Series.
    DataFrame.drop_duplicates : Remove duplicate values from DataFrame.

    Examples
    --------
    Consider dataset containing ramen rating.

    >>> import maxframe.dataframe as md

    >>> df = md.DataFrame({
    ...     'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
    ...     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
    ...     'rating': [4, 4, 3.5, 15, 5]
    ... })
    >>> df.execute()
        brand style  rating
    0  Yum Yum   cup     4.0
    1  Yum Yum   cup     4.0
    2  Indomie   cup     3.5
    3  Indomie  pack    15.0
    4  Indomie  pack     5.0

    By default, for each set of duplicated values, the first occurrence
    is set on False and all others on True.

    >>> df.duplicated().execute()
    0    False
    1     True
    2    False
    3    False
    4    False
    dtype: bool

    By using 'last', the last occurrence of each set of duplicated values
    is set on False and all others on True.

    >>> df.duplicated(keep='last').execute()
    0     True
    1    False
    2    False
    3    False
    4    False
    dtype: bool

    By setting ``keep`` on False, all duplicates are True.

    >>> df.duplicated(keep=False).execute()
    0     True
    1     True
    2    False
    3    False
    4    False
    dtype: bool

    To find duplicates on specific column(s), use ``subset``.

    >>> df.duplicated(subset=['brand']).execute()
    0    False
    1     True
    2    False
    3     True
    4     True
    dtype: bool
    """

    if method not in ("auto", "tree", "subset_tree", "shuffle", None):
        raise ValueError(
            "method could only be one of "
            "'auto', 'tree', 'subset_tree', 'shuffle' or None"
        )
    subset = validate_subset(df, subset)
    op = DataFrameDuplicated(subset=subset, keep=keep, method=method)
    return op(df)


def series_duplicated(series, keep="first", method="auto"):
    """
    Indicate duplicate Series values.

    Duplicated values are indicated as ``True`` values in the resulting
    Series. Either all duplicates, all except the first or all except the
    last occurrence of duplicates can be indicated.

    Parameters
    ----------
    keep : {'first', 'last', False}, default 'first'
        Method to handle dropping duplicates:

        - 'first' : Mark duplicates as ``True`` except for the first
          occurrence.
        - 'last' : Mark duplicates as ``True`` except for the last
          occurrence.
        - ``False`` : Mark all duplicates as ``True``.

    Returns
    -------
    Series
        Series indicating whether each value has occurred in the
        preceding values.

    See Also
    --------
    Index.duplicated : Equivalent method on pandas.Index.
    DataFrame.duplicated : Equivalent method on pandas.DataFrame.
    Series.drop_duplicates : Remove duplicate values from Series.

    Examples
    --------
    By default, for each set of duplicated values, the first occurrence is
    set on False and all others on True:

    >>> import maxframe.dataframe as md

    >>> animals = md.Series(['lame', 'cow', 'lame', 'beetle', 'lame'])
    >>> animals.duplicated().execute()
    0    False
    1    False
    2     True
    3    False
    4     True
    dtype: bool

    which is equivalent to

    >>> animals.duplicated(keep='first').execute()
    0    False
    1    False
    2     True
    3    False
    4     True
    dtype: bool

    By using 'last', the last occurrence of each set of duplicated values
    is set on False and all others on True:

    >>> animals.duplicated(keep='last').execute()
    0     True
    1    False
    2     True
    3    False
    4    False
    dtype: bool

    By setting keep on ``False``, all duplicates are True:

    >>> animals.duplicated(keep=False).execute()
    0     True
    1    False
    2     True
    3    False
    4     True
    dtype: bool
    """
    if method not in ("auto", "tree", "shuffle", None):
        raise ValueError(
            "method could only be one of 'auto', 'tree', 'shuffle' or None"
        )
    op = DataFrameDuplicated(keep=keep, method=method)
    return op(series)


def index_duplicated(index, keep="first"):
    """
    Indicate duplicate index values.

    Duplicated values are indicated as ``True`` values in the resulting
    array. Either all duplicates, all except the first, or all except the
    last occurrence of duplicates can be indicated.

    Parameters
    ----------
    keep : {'first', 'last', False}, default 'first'
        The value or values in a set of duplicates to mark as missing.
        - 'first' : Mark duplicates as ``True`` except for the first
          occurrence.
        - 'last' : Mark duplicates as ``True`` except for the last
          occurrence.
        - ``False`` : Mark all duplicates as ``True``.

    Returns
    -------
    Tensor

    See Also
    --------
    Series.duplicated : Equivalent method on pandas.Series.
    DataFrame.duplicated : Equivalent method on pandas.DataFrame.
    Index.drop_duplicates : Remove duplicate values from Index.

    Examples
    --------
    By default, for each set of duplicated values, the first occurrence is
    set to False and all others to True:

    >>> import maxframe.dataframe as md

    >>> idx = md.Index(['lame', 'cow', 'lame', 'beetle', 'lame'])
    >>> idx.duplicated().execute()
    array([False, False,  True, False,  True])

    which is equivalent to

    >>> idx.duplicated(keep='first').execute()
    array([False, False,  True, False,  True])

    By using 'last', the last occurrence of each set of duplicated values
    is set on False and all others on True:

    >>> idx.duplicated(keep='last').execute()
    array([ True, False,  True, False, False])

    By setting keep on ``False``, all duplicates are True:

    >>> idx.duplicated(keep=False).execute()
    array([ True, False,  True, False,  True])
    """
    return index.to_series().duplicated(keep=keep).to_tensor()
