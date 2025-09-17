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

from typing import List

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData
from ...serialization.serializables import AnyField, BoolField, Int32Field, StringField
from ...tensor.utils import filter_inputs
from ..core import DATAFRAME_TYPE, SERIES_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_df, build_series, validate_axis


class DataFrameWhere(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.WHERE

    _input = AnyField("input")
    cond = AnyField("cond", default=None)
    other = AnyField("other", default=None)
    axis = Int32Field("axis", default=None)
    level = AnyField("level", default=None)
    errors = StringField("errors", default=None)
    try_cast = BoolField("try_cast", default=None)
    replace_true = BoolField("replace_true", default=None)

    def __init__(self, input=None, **kw):
        super().__init__(_input=input, **kw)

    @property
    def input(self):
        return self._input

    def __call__(self, df_or_series):
        def _check_input_index(obj, axis=None):
            axis = axis if axis is not None else self.axis
            if isinstance(obj, DATAFRAME_TYPE) and (
                df_or_series.columns_value.key != obj.columns_value.key
                or df_or_series.index_value.key != obj.index_value.key
            ):
                raise NotImplementedError("Aligning different indices not supported")
            elif (
                isinstance(obj, SERIES_TYPE)
                and df_or_series.axes[axis].index_value.key != obj.index_value.key
            ):
                raise NotImplementedError("Aligning different indices not supported")

        _check_input_index(self.cond, axis=0)
        _check_input_index(self.other)

        if isinstance(df_or_series, DATAFRAME_TYPE):
            mock_obj = build_df(df_or_series)
        else:
            mock_obj = build_series(df_or_series)

        if isinstance(self.other, (pd.DataFrame, DATAFRAME_TYPE)):
            mock_other = build_df(self.other)
        elif isinstance(self.other, (pd.Series, SERIES_TYPE)):
            mock_other = build_series(self.other)
        else:
            mock_other = self.other

        result_df = mock_obj.where(
            np.zeros(mock_obj.shape).astype(bool),
            other=mock_other,
            axis=self.axis,
            level=self.level,
        )

        inputs = filter_inputs([df_or_series, self.cond, self.other])
        if isinstance(df_or_series, DATAFRAME_TYPE):
            return self.new_dataframe(
                inputs,
                shape=df_or_series.shape,
                dtypes=result_df.dtypes,
                index_value=df_or_series.index_value,
                columns_value=df_or_series.columns_value,
            )
        else:
            return self.new_series(
                inputs,
                shape=df_or_series.shape,
                name=df_or_series.name,
                dtype=result_df.dtype,
                index_value=df_or_series.index_value,
            )

    @classmethod
    def _set_inputs(cls, op: "DataFrameWhere", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs)
        op._input = next(inputs_iter)
        if isinstance(op.cond, ENTITY_TYPE):
            op.cond = next(inputs_iter)
        if isinstance(op.other, ENTITY_TYPE):
            op.other = next(inputs_iter)


_doc_template = """
Replace values where the condition is {replace_true}.

Parameters
----------
cond : bool Series/DataFrame, array-like, or callable
    Where `cond` is False, keep the original value. Where
    True, replace with corresponding value from `other`.
    If `cond` is callable, it is computed on the Series/DataFrame and
    should return boolean Series/DataFrame or array. The callable must
    not change input Series/DataFrame (though pandas doesn't check it).
other : scalar, Series/DataFrame, or callable
    Entries where `cond` is True are replaced with
    corresponding value from `other`.
    If other is callable, it is computed on the Series/DataFrame and
    should return scalar or Series/DataFrame. The callable must not
    change input Series/DataFrame (though pandas doesn't check it).
inplace : bool, default False
    Whether to perform the operation in place on the data.
axis : int, default None
    Alignment axis if needed.
level : int, default None
    Alignment level if needed.

Returns
-------
Same type as caller

See Also
--------
:func:`DataFrame.{opposite}` : Return an object of same shape as
    self.

Notes
-----
The mask method is an application of the if-then idiom. For each
element in the calling DataFrame, if ``cond`` is ``False`` the
element is used; otherwise the corresponding element from the DataFrame
``other`` is used.

The signature for :func:`DataFrame.where` differs from
:func:`numpy.where`. Roughly ``df1.where(m, df2)`` is equivalent to
``np.where(m, df1, df2)``.

For further details and examples see the ``mask`` documentation in
:ref:`indexing <indexing.where_mask>`.

Examples
--------
>>> import maxframe.tensor as mt
>>> import maxframe.dataframe as md
>>> s = md.Series(range(5))
>>> s.where(s > 0).execute()
0    NaN
1    1.0
2    2.0
3    3.0
4    4.0
dtype: float64

>>> s.mask(s > 0).execute()
0    0.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

>>> s.where(s > 1, 10).execute()
0    10
1    10
2    2
3    3
4    4
dtype: int64

>>> df = md.DataFrame(mt.arange(10).reshape(-1, 2), columns=['A', 'B'])
>>> df.execute()
   A  B
0  0  1
1  2  3
2  4  5
3  6  7
4  8  9
>>> m = df % 3 == 0
>>> df.where(m, -df).execute()
   A  B
0  0 -1
1 -2  3
2 -4 -5
3  6 -7
4 -8  9
>>> df.where(m, -df) == mt.where(m, df, -df).execute()
      A     B
0  True  True
1  True  True
2  True  True
3  True  True
4  True  True
>>> df.where(m, -df) == df.mask(~m, -df).execute()
      A     B
0  True  True
1  True  True
2  True  True
3  True  True
4  True  True
"""


def _where(
    df_or_series,
    cond,
    other=np.nan,
    inplace=False,
    axis=None,
    level=None,
    errors="raise",
    try_cast=False,
    replace_true=False,
):
    if df_or_series.ndim == 2 and getattr(other, "ndim", 2) == 1 and axis is None:
        raise ValueError("Must specify axis=0 or 1")

    axis = validate_axis(axis or 0, df_or_series)
    op = DataFrameWhere(
        cond=cond,
        other=other,
        axis=axis,
        level=level,
        errors=errors,
        try_cast=try_cast,
        replace_true=replace_true,
    )
    result = op(df_or_series)
    if inplace:
        df_or_series.data = result.data
    else:
        return result


def where(
    df_or_series,
    cond,
    other=np.nan,
    inplace=False,
    axis=None,
    level=None,
    errors="raise",
    try_cast=False,
):
    return _where(
        df_or_series,
        cond,
        other=other,
        inplace=inplace,
        axis=axis,
        level=level,
        errors=errors,
        try_cast=try_cast,
        replace_true=False,
    )


def mask(
    df_or_series,
    cond,
    other=np.nan,
    inplace=False,
    axis=None,
    level=None,
    errors="raise",
    try_cast=False,
):
    return _where(
        df_or_series,
        cond,
        other=other,
        inplace=inplace,
        axis=axis,
        level=level,
        errors=errors,
        try_cast=try_cast,
        replace_true=True,
    )


mask.__doc__ = _doc_template.format(replace_true=True, opposite="where")
where.__doc__ = _doc_template.format(replace_true=False, opposite="mask")
