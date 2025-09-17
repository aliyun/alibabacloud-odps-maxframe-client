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

from collections import OrderedDict
from typing import List

import pandas as pd

from ... import opcodes
from ...core import EntityData
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FieldTypes,
    Int32Field,
    Int64Field,
    KeyField,
    ListField,
    StringField,
    TupleField,
)
from ..core import DATAFRAME_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_empty_df, build_empty_series, parse_index, validate_axis
from .core import Window


class DataFrameRollingAgg(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.ROLLING_AGG

    input = KeyField("input")
    window = AnyField("window", default=None)
    min_periods = Int64Field("min_periods", default=None)
    center = BoolField("center", default=None)
    win_type = StringField("win_type", default=None)
    on = StringField("on", default=None)
    axis = Int32Field("axis", default=None)
    closed = StringField("closed", default=None)
    func = AnyField("func", default=None)
    func_args = TupleField("func_args", default=None)
    func_kwargs = DictField("func_kwargs", default=None)
    # for chunks
    preds = ListField("preds", FieldTypes.key, default=None)
    succs = ListField("succs", FieldTypes.key, default=None)

    def __init__(self, output_types=None, **kw):
        # suspend MF-specific args by now
        for key in Rolling._mf_specific_fields:
            kw.pop(key, None)
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _set_inputs(cls, op: "DataFrameRollingAgg", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        input_iter = iter(op._inputs)
        op.input = next(input_iter)
        if op.preds is not None:
            op.preds = [next(input_iter) for _ in op.preds]
        if op.succs is not None:
            op.succs = [next(input_iter) for _ in op.succs]

    def __call__(self, rolling):
        inp = rolling.input

        if isinstance(inp, DATAFRAME_TYPE):
            pd_index = inp.index_value.to_pandas()
            empty_df = build_empty_df(inp.dtypes, index=pd_index[:0])
            params = rolling.params.copy()
            if params["win_type"] == "freq":
                params["win_type"] = None
            if self.func != "count":
                empty_df = empty_df._get_numeric_data()
            for key in Rolling._mf_specific_fields:
                params.pop(key, None)
            test_df = empty_df.rolling(**params).agg(self.func)
            if self.axis == 0:
                index_value = inp.index_value
            else:
                index_value = parse_index(
                    test_df.index, rolling.params, inp, store_data=False
                )
            return self.new_dataframe(
                [inp],
                shape=(inp.shape[0], test_df.shape[1]),
                dtypes=test_df.dtypes,
                index_value=index_value,
                columns_value=parse_index(test_df.columns, store_data=True),
            )
        else:
            pd_index = inp.index_value.to_pandas()
            empty_series = build_empty_series(
                inp.dtype, index=pd_index[:0], name=inp.name
            )
            rolling_params = rolling.params.copy()
            for k in Rolling._mf_specific_fields:
                rolling_params.pop(k, None)
            test_obj = empty_series.rolling(**rolling_params).agg(self.func)
            if isinstance(test_obj, pd.DataFrame):
                return self.new_dataframe(
                    [inp],
                    shape=(inp.shape[0], test_obj.shape[1]),
                    dtypes=test_obj.dtypes,
                    index_value=inp.index_value,
                    columns_value=parse_index(test_obj.dtypes.index, store_data=True),
                )
            else:
                return self.new_series(
                    [inp],
                    shape=inp.shape,
                    dtype=test_obj.dtype,
                    index_value=inp.index_value,
                    name=test_obj.name,
                )


class Rolling(Window):
    _mf_specific_fields = Window._mf_specific_fields + ["shift"]

    window = AnyField("window", default=None)
    min_periods = Int64Field("min_periods", default=None)
    center = BoolField("center", default=None)
    win_type = StringField("win_type", default=None)
    on = StringField("on", default=None)
    axis = Int32Field("axis", default=None)
    closed = StringField("closed", default=None)
    # MF specific argument for position shift of window
    shift = Int64Field("shift", default=None)

    @property
    def params(self):
        p = OrderedDict()

        args = [
            "window",
            "min_periods",
            "center",
            "win_type",
            "axis",
            "on",
            "closed",
            "shift",
            "order_cols",
            "ascending",
        ]

        for attr in args:
            p[attr] = getattr(self, attr)
        return p

    def _repr_name(self):
        return "Rolling" if self.win_type is None else "Window"

    def validate(self):
        # leverage pandas itself to do validation
        pd_index = self.input.index_value.to_pandas()
        if isinstance(self.input, DATAFRAME_TYPE):
            empty_obj = build_empty_df(self.input.dtypes, index=pd_index[:0])
        else:
            empty_obj = build_empty_series(
                self.input.dtype, index=pd_index[:0], name=self.input.name
            )
        params = (self.params or dict()).copy()
        for key in self._mf_specific_fields:
            params.pop(key, None)
        pd_rolling = empty_obj.rolling(**params)
        for k in params:
            # update value according to pandas rolling
            setattr(self, k, getattr(pd_rolling, k))

    def aggregate(self, func, *args, **kwargs):
        op = DataFrameRollingAgg(
            func=func, func_args=args, func_kwargs=kwargs, **self.params
        )
        return op(self)

    def agg(self, func, *args, **kwargs):
        return self.aggregate(func, *args, **kwargs)

    def count(self):
        return self.aggregate("count")

    def sum(self, *args, **kwargs):
        return self.aggregate("sum", *args, **kwargs)

    def mean(self, *args, **kwargs):
        return self.aggregate("mean", *args, **kwargs)

    def median(self, **kwargs):
        return self.aggregate("median", **kwargs)

    def var(self, ddof=1, *args, **kwargs):
        return self.aggregate("var", ddof=ddof, *args, **kwargs)

    def std(self, ddof=1, *args, **kwargs):
        return self.aggregate("std", ddof=ddof, *args, **kwargs)

    def min(self, *args, **kwargs):
        return self.aggregate("min", *args, **kwargs)

    def max(self, *args, **kwargs):
        return self.aggregate("max", *args, **kwargs)

    def skew(self, **kwargs):
        return self.aggregate("skew", **kwargs)

    def kurt(self, **kwargs):
        return self.aggregate("kurt", **kwargs)


def rolling(
    obj,
    window,
    min_periods=None,
    center=False,
    win_type=None,
    on=None,
    axis=0,
    closed=None,
):
    """
    Provide rolling window calculations.

    Parameters
    ----------
    window : int, or offset
        Size of the moving window. This is the number of observations used for
        calculating the statistic. Each window will be a fixed size.
        If its an offset then this will be the time period of each window. Each
        window will be a variable sized based on the observations included in
        the time-period. This is only valid for datetimelike indexes. This is
        new in 0.19.0
    min_periods : int, default None
        Minimum number of observations in window required to have a value
        (otherwise result is NA). For a window that is specified by an offset,
        `min_periods` will default to 1. Otherwise, `min_periods` will default
        to the size of the window.
    center : bool, default False
        Set the labels at the center of the window.
    win_type : str, default None
        Provide a window type. If ``None``, all points are evenly weighted.
        See the notes below for further information.
    on : str, optional
        For a DataFrame, a datetime-like column on which to calculate the rolling
        window, rather than the DataFrame's index. Provided integer column is
        ignored and excluded from result since an integer index is not used to
        calculate the rolling window.
    axis : int or str, default 0
    closed : str, default None
        Make the interval closed on the 'right', 'left', 'both' or
        'neither' endpoints.
        For offset-based windows, it defaults to 'right'.
        For fixed windows, defaults to 'both'. Remaining cases not implemented
        for fixed windows.

    Returns
    -------
    a Window or Rolling sub-classed for the particular operation

    See Also
    --------
    expanding : Provides expanding transformations.
    ewm : Provides exponential weighted functions.

    Notes
    -----
    By default, the result is set to the right edge of the window. This can be
    changed to the center of the window by setting ``center=True``.
    To learn more about the offsets & frequency strings, please see `this link
    <http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    The recognized win_types are:
    * ``boxcar``
    * ``triang``
    * ``blackman``
    * ``hamming``
    * ``bartlett``
    * ``parzen``
    * ``bohman``
    * ``blackmanharris``
    * ``nuttall``
    * ``barthann``
    * ``kaiser`` (needs beta)
    * ``gaussian`` (needs std)
    * ``general_gaussian`` (needs power, width)
    * ``slepian`` (needs width)
    * ``exponential`` (needs tau), center is set to None.

    If ``win_type=None`` all points are evenly weighted. To learn more about
    different window types see `scipy.signal window functions
    <https://docs.scipy.org/doc/scipy/reference/signal.html#window-functions>`__.

    Examples
    --------
    >>> import numpy as np
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'B': [0, 1, 2, np.nan, 4]})
    >>> df.execute()
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    Rolling sum with a window length of 2, using the 'triang'
    window type.

    >>> df.rolling(2, win_type='triang').sum().execute()
         B
    0  NaN
    1  0.5
    2  1.5
    3  NaN
    4  NaN

    Rolling sum with a window length of 2, min_periods defaults
    to the window length.

    >>> df.rolling(2).sum().execute()
         B
    0  NaN
    1  1.0
    2  3.0
    3  NaN
    4  NaN

    Same as above, but explicitly set the min_periods

    >>> df.rolling(2, min_periods=1).sum().execute()
         B
    0  0.0
    1  1.0
    2  3.0
    3  2.0
    4  4.0

    A ragged (meaning not-a-regular frequency), time-indexed DataFrame

    >>> df = md.DataFrame({'B': [0, 1, 2, np.nan, 4]},
    >>>                   index = [md.Timestamp('20130101 09:00:00'),
    >>>                            md.Timestamp('20130101 09:00:02'),
    >>>                            md.Timestamp('20130101 09:00:03'),
    >>>                            md.Timestamp('20130101 09:00:05'),
    >>>                            md.Timestamp('20130101 09:00:06')])
    >>> df.execute()
                           B
    2013-01-01 09:00:00  0.0
    2013-01-01 09:00:02  1.0
    2013-01-01 09:00:03  2.0
    2013-01-01 09:00:05  NaN
    2013-01-01 09:00:06  4.0

    Contrasting to an integer rolling window, this will roll a variable
    length window corresponding to the time period.
    The default for min_periods is 1.

    >>> df.rolling('2s').sum().execute()
                           B
    2013-01-01 09:00:00  0.0
    2013-01-01 09:00:02  1.0
    2013-01-01 09:00:03  3.0
    2013-01-01 09:00:05  NaN
    2013-01-01 09:00:06  4.0
    """
    axis = validate_axis(axis, obj)
    r = Rolling(
        input=obj,
        window=window,
        min_periods=min_periods,
        center=center,
        win_type=win_type,
        on=on,
        axis=axis,
        closed=closed,
    )
    r.validate()
    return r
