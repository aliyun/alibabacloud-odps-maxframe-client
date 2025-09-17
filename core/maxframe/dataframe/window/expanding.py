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

from ... import opcodes
from ...serialization.serializables import BoolField, Int64Field
from .aggregation import BaseDataFrameExpandingAgg
from .core import Window


class DataFrameExpandingAgg(BaseDataFrameExpandingAgg):
    _op_type_ = opcodes.EXPANDING_AGG

    def __init__(self, *args, **kw):
        # suspend MF-specific args by now
        for key in Expanding._mf_specific_fields:
            kw.pop(key, None)
        super().__init__(*args, **kw)


class Expanding(Window):
    _mf_specific_fields = Window._mf_specific_fields + ["shift", "reverse_range"]

    min_periods = Int64Field("min_periods")
    # MF specific argument for position shift of window
    shift = Int64Field("shift", default=None)
    # MF specific argument for reversed window (sort of "narrowing")
    reverse_range = BoolField("reverse_range", default=False)

    def __call__(self, df):
        try:
            return df.expanding(**self.params)
        except TypeError:
            params = (self.params or dict()).copy()
            for key in self._mf_specific_fields:
                params.pop(key, None)
            return df.expanding(**params)

    @property
    def params(self):
        p = OrderedDict()

        args = [
            "min_periods",
            "shift",
            "reverse_range",
            "order_cols",
            "ascending",
        ]

        for k in args:
            p[k] = getattr(self, k)
        return p

    def aggregate(self, func, **kwargs):
        count_always_valid = kwargs.pop("_count_always_valid", False)

        op = DataFrameExpandingAgg(
            func=func, count_always_valid=count_always_valid, **self.params
        )
        return op(self)

    agg = aggregate

    def sum(self):
        return self.aggregate("sum")

    def prod(self):
        return self.aggregate("prod")

    def count(self):
        return self.aggregate("count")

    def min(self):
        return self.aggregate("min")

    def max(self):
        return self.aggregate("max")

    def mean(self):
        return self.aggregate("mean")

    def var(self, **kwargs):
        return self.aggregate("var", **kwargs)

    def std(self, **kwargs):
        return self.aggregate("std", **kwargs)


def expanding(obj, min_periods=1, shift=0, reverse_range=False):
    """
    Provide expanding transformations.

    Parameters
    ----------
    min_periods : int, default 1
    Minimum number of observations in window required to have a value
    (otherwise result is NA).
    center : bool, default False
    Set the labels at the center of the window.
    axis : int or str, default 0

    Returns
    -------
    a Window sub-classed for the particular operation

    See Also
    --------
    rolling : Provides rolling window calculations.
    ewm : Provides exponential weighted functions.

    Notes
    -----
    By default, the result is set to the right edge of the window. This can be
    changed to the center of the window by setting ``center=True``.

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
    >>> df.expanding(2).sum().execute()
         B
    0  NaN
    1  1.0
    2  3.0
    3  3.0
    4  7.0
    """
    return Expanding(
        input=obj, min_periods=min_periods, shift=shift, reverse_range=reverse_range
    )
