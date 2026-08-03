# Copyright 1999-2026 Alibaba Group Holding Ltd.
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


class LiteFrameGroupBy:
    """Intermediate object returned by LiteFrame.groupby().

    Supports aggregation via .agg() and shortcut methods like .sum(), .mean(), etc.
    Always uses column-based output (as_index=False equivalent), no multi-index.

    Parameters
    ----------
    liteframe : LiteFrame
        The LiteFrame to group.
    by : str or list of str
        Column name(s) to group by.
    sort : bool, default False
        Whether to sort group keys in the result. Currently not implemented;
        passing ``True`` raises ``NotImplementedError``. Group keys are
        returned in first-seen encounter order.
    dropna : bool, default True
        If True, rows with null group keys are dropped before aggregation.
        If False, null-key rows form their own group.
    """

    def __init__(self, liteframe, by, sort=False, dropna=True):
        if by is None:
            raise ValueError("'by' argument must be specified for groupby")
        if isinstance(by, str):
            if by not in liteframe.dtypes.index:
                raise KeyError(f"Column '{by}' not found in LiteFrame")
        elif isinstance(by, (list, tuple)):
            missing = [c for c in by if c not in liteframe.dtypes.index]
            if missing:
                raise KeyError(f"Columns {missing} not found in LiteFrame")
        else:
            raise TypeError(
                f"'by' must be a column name (str) or list of column names, "
                f"got {type(by).__name__}"
            )
        if sort:
            raise NotImplementedError(
                "sort=True for groupby is not yet implemented; "
                "group keys are returned in encounter order. "
                "Please sort the result manually if needed."
            )
        self._liteframe = liteframe
        self._by = by
        self._sort = sort
        self._dropna = dropna

    @property
    def groupby_params(self):
        return {
            "by": self._by,
            "sort": self._sort,
            "dropna": self._dropna,
        }

    def agg(self, func=None, method="auto", numeric_only=None, **kwargs):
        from maxframe.liteframe.operators.agg import LiteFrameAgg

        op = LiteFrameAgg(
            raw_func=func,
            raw_func_kw=kwargs if kwargs else None,
            groupby_params=self.groupby_params,
            method=method,
            numeric_only=numeric_only,
        )
        return op(self._liteframe)

    def sum(self, **kw):
        return self.agg("sum", **kw)

    def mean(self, **kw):
        return self.agg("mean", **kw)

    def min(self, **kw):
        return self.agg("min", **kw)

    def max(self, **kw):
        return self.agg("max", **kw)

    def prod(self, **kw):
        return self.agg("prod", **kw)

    def count(self, **kw):
        return self.agg("count", **kw)

    def size(self, **kw):
        return self.agg("size", **kw)

    def var(self, **kw):
        return self.agg("var", **kw)

    def std(self, **kw):
        return self.agg("std", **kw)

    def sem(self, **kw):
        return self.agg("sem", **kw)

    def skew(self, **kw):
        return self.agg("skew", **kw)

    def kurt(self, **kw):
        return self.agg("kurt", **kw)

    def kurtosis(self, **kw):
        return self.agg("kurtosis", **kw)

    def nunique(self, dropna=True, **kw):
        from maxframe.liteframe.reduction.compile import AggCall

        return self.agg(AggCall("nunique", dropna=dropna), **kw)

    def median(self, **kw):
        return self.agg("median", **kw)

    def all(self, **kw):
        return self.agg("all", **kw)

    def any(self, **kw):
        return self.agg("any", **kw)
