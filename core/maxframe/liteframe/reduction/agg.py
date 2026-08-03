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

from maxframe.liteframe.operators.agg import LiteFrameAgg
from maxframe.liteframe.reduction.compile import AggCall


def agg(self, func=None, method="auto", numeric_only=None, **kwargs):
    op = LiteFrameAgg(
        raw_func=func,
        raw_func_kw=kwargs if kwargs else None,
        groupby_params=None,
        method=method,
        numeric_only=numeric_only,
    )
    return op(self)


def sum(self, method="auto", numeric_only=None, **kw):
    return self.agg("sum", method=method, numeric_only=numeric_only, **kw)


def mean(self, method="auto", numeric_only=None, **kw):
    return self.agg("mean", method=method, numeric_only=numeric_only, **kw)


def min(self, method="auto", numeric_only=None, **kw):
    return self.agg("min", method=method, numeric_only=numeric_only, **kw)


def max(self, method="auto", numeric_only=None, **kw):
    return self.agg("max", method=method, numeric_only=numeric_only, **kw)


def prod(self, method="auto", numeric_only=None, **kw):
    return self.agg("prod", method=method, numeric_only=numeric_only, **kw)


def count(self, method="auto", **kw):
    return self.agg("count", method=method, **kw)


def size(self, method="auto", **kw):
    return self.agg("size", method=method, **kw)


def var(self, method="auto", numeric_only=None, **kw):
    return self.agg("var", method=method, numeric_only=numeric_only, **kw)


def std(self, method="auto", numeric_only=None, **kw):
    return self.agg("std", method=method, numeric_only=numeric_only, **kw)


def sem(self, method="auto", numeric_only=None, **kw):
    return self.agg("sem", method=method, numeric_only=numeric_only, **kw)


def skew(self, method="auto", numeric_only=None, **kw):
    return self.agg("skew", method=method, numeric_only=numeric_only, **kw)


def kurt(self, method="auto", numeric_only=None, **kw):
    return self.agg("kurt", method=method, numeric_only=numeric_only, **kw)


def kurtosis(self, method="auto", numeric_only=None, **kw):
    return self.agg("kurtosis", method=method, numeric_only=numeric_only, **kw)


def nunique(self, method="auto", numeric_only=None, dropna=True, **kw):
    return self.agg(
        AggCall("nunique", dropna=dropna),
        method=method,
        numeric_only=numeric_only,
        **kw,
    )


def median(self, method="auto", numeric_only=None, **kw):
    return self.agg("median", method=method, numeric_only=numeric_only, **kw)


def all(self, method="auto", numeric_only=None, **kw):
    return self.agg("all", method=method, numeric_only=numeric_only, **kw)


def any(self, method="auto", numeric_only=None, **kw):
    return self.agg("any", method=method, numeric_only=numeric_only, **kw)
