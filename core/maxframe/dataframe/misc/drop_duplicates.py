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
import pandas as pd

from ... import opcodes
from ...serialization.serializables import BoolField
from ..operators import OutputType
from ..utils import gen_unknown_index_value, parse_index
from ._duplicate import BaseDuplicateOp, validate_subset


class DataFrameDropDuplicates(BaseDuplicateOp):
    _op_type_ = opcodes.DROP_DUPLICATES

    ignore_index = BoolField("ignore_index", default=True)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _get_shape(cls, input_shape, op: "DataFrameDropDuplicates"):
        shape = (np.nan,) + input_shape[1:]
        if op.output_types[0] == OutputType.dataframe and len(shape) == 1:
            shape += (3,)
        return shape

    def _gen_tileable_params(self, op: "DataFrameDropDuplicates", input_params):
        params = input_params.copy()
        if op.ignore_index and self._output_types[0] != OutputType.index:
            params["index_value"] = parse_index(pd.RangeIndex(-1))
        else:
            params["index_value"] = gen_unknown_index_value(
                input_params["index_value"],
                op.keep,
                op.subset,
                type(op).__name__,
                normalize_range_index=True,
            )
        params["shape"] = self._get_shape(input_params["shape"], op)
        return params

    def __call__(self, inp, inplace=False):
        self._output_types = inp.op.output_types
        params = self._gen_tileable_params(self, inp.params)

        ret = self.new_tileable([inp], kws=[params])
        if inplace:
            inp.data = ret.data
        return ret


def df_drop_duplicates(
    df, subset=None, keep="first", inplace=False, ignore_index=False, method="auto"
):
    """
    Return DataFrame with duplicate rows removed.

    Considering certain columns is optional. Indexes, including time indexes
    are ignored.

    Parameters
    ----------
    subset : column label or sequence of labels, optional
        Only consider certain columns for identifying duplicates, by
        default use all of the columns.
    keep : {'first', 'last', False}, default 'first'
        Determines which duplicates (if any) to keep.
        - ``first`` : Drop duplicates except for the first occurrence.
        - ``last`` : Drop duplicates except for the last occurrence.
        - ``any`` : Drop duplicates except for a random occurrence.
        - False : Drop all duplicates.
    inplace : bool, default False
        Whether to drop duplicates in place or to return a copy.
    ignore_index : bool, default False
        If True, the resulting axis will be labeled 0, 1, …, n - 1.

    Returns
    -------
    DataFrame
        DataFrame with duplicates removed or None if ``inplace=True``.
    """
    if keep not in ("first", "last", "any", False):
        raise ValueError("keep could only be one of 'first', 'last' or False")
    if method not in ("auto", "tree", "subset_tree", "shuffle", None):
        raise ValueError(
            "method could only be one of "
            "'auto', 'tree', 'subset_tree', 'shuffle' or None"
        )
    subset = validate_subset(df, subset)
    op = DataFrameDropDuplicates(
        subset=subset, keep=keep, ignore_index=ignore_index, method=method
    )
    return op(df, inplace=inplace)


def series_drop_duplicates(
    series, keep="first", inplace=False, ignore_index=False, method="auto"
):
    """
    Return Series with duplicate values removed.

    Parameters
    ----------
    keep : {'first', 'last', ``False``}, default 'first'
        Method to handle dropping duplicates:

        - 'first' : Drop duplicates except for the first occurrence.
        - 'last' : Drop duplicates except for the last occurrence.
        - 'any' : Drop duplicates except for a random occurrence.
        - ``False`` : Drop all duplicates.

    inplace : bool, default ``False``
        If ``True``, performs operation inplace and returns None.

    Returns
    -------
    Series
        Series with duplicates dropped.

    See Also
    --------
    Index.drop_duplicates : Equivalent method on Index.
    DataFrame.drop_duplicates : Equivalent method on DataFrame.
    Series.duplicated : Related method on Series, indicating duplicate
        Series values.

    Examples
    --------
    Generate a Series with duplicated entries.

    >>> import maxframe.dataframe as md
    >>> s = md.Series(['lame', 'cow', 'lame', 'beetle', 'lame', 'hippo'],
    ...               name='animal')
    >>> s.execute()
    0      lame
    1       cow
    2      lame
    3    beetle
    4      lame
    5     hippo
    Name: animal, dtype: object

    With the 'keep' parameter, the selection behaviour of duplicated values
    can be changed. The value 'first' keeps the first occurrence for each
    set of duplicated entries. The default value of keep is 'first'.
    >>> s.drop_duplicates().execute()
    0      lame
    1       cow
    3    beetle
    5     hippo
    Name: animal, dtype: object
    The value 'last' for parameter 'keep' keeps the last occurrence for
    each set of duplicated entries.
    >>> s.drop_duplicates(keep='last').execute()
    1       cow
    3    beetle
    4      lame
    5     hippo
    Name: animal, dtype: object

    The value ``False`` for parameter 'keep' discards all sets of
    duplicated entries. Setting the value of 'inplace' to ``True`` performs
    the operation inplace and returns ``None``.

    >>> s.drop_duplicates(keep=False, inplace=True)
    >>> s.execute()
    1       cow
    3    beetle
    5     hippo
    Name: animal, dtype: object
    """
    if keep not in ("first", "last", "any", False):
        raise ValueError("keep could only be one of 'first', 'last' or False")
    if method not in ("auto", "tree", "shuffle", None):
        raise ValueError(
            "method could only be one of 'auto', 'tree', 'shuffle' or None"
        )
    op = DataFrameDropDuplicates(keep=keep, ignore_index=ignore_index, method=method)
    return op(series, inplace=inplace)


def index_drop_duplicates(index, keep="first", method="auto"):
    """
    Return Index with duplicate values removed.

    Parameters
    ----------
    keep : {'first', 'last', ``False``}, default 'first'
        - 'first' : Drop duplicates except for the first occurrence.
        - 'last' : Drop duplicates except for the last occurrence.
        - 'any' : Drop duplicates except for a random occurrence.
        - ``False`` : Drop all duplicates.

    Returns
    -------
    deduplicated : Index

    See Also
    --------
    Series.drop_duplicates : Equivalent method on Series.
    DataFrame.drop_duplicates : Equivalent method on DataFrame.
    Index.duplicated : Related method on Index, indicating duplicate
        Index values.

    Examples
    --------
    Generate a pandas.Index with duplicate values.

    >>> import maxframe.dataframe as md

    >>> idx = md.Index(['lame', 'cow', 'lame', 'beetle', 'lame', 'hippo'])

    The `keep` parameter controls  which duplicate values are removed.
    The value 'first' keeps the first occurrence for each
    set of duplicated entries. The default value of keep is 'first'.

    >>> idx.drop_duplicates(keep='first').execute()
    Index(['lame', 'cow', 'beetle', 'hippo'], dtype='object')

    The value 'last' keeps the last occurrence for each set of duplicated
    entries.

    >>> idx.drop_duplicates(keep='last').execute()
    Index(['cow', 'beetle', 'lame', 'hippo'], dtype='object')

    The value ``False`` discards all sets of duplicated entries.

    >>> idx.drop_duplicates(keep=False).execute()
    Index(['cow', 'beetle', 'hippo'], dtype='object')
    """
    if keep not in ("first", "last", "any", False):
        raise ValueError("keep could only be one of 'first', 'last' or False")
    if method not in ("auto", "tree", "shuffle", None):
        raise ValueError(
            "method could only be one of 'auto', 'tree', 'shuffle' or None"
        )
    op = DataFrameDropDuplicates(keep=keep, method=method)
    return op(index)
