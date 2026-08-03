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


def _install():
    from maxframe.liteframe.core import LITEFRAME_TYPE
    from maxframe.liteframe.reduction.agg import (
        agg,
        all,
        any,
        count,
        kurt,
        kurtosis,
        max,
        mean,
        median,
        min,
        nunique,
        prod,
        sem,
        size,
        skew,
        std,
        sum,
        var,
    )

    for cls in LITEFRAME_TYPE:
        setattr(cls, "agg", agg)
        setattr(cls, "aggregate", agg)
        setattr(cls, "sum", sum)
        setattr(cls, "mean", mean)
        setattr(cls, "min", min)
        setattr(cls, "max", max)
        setattr(cls, "prod", prod)
        setattr(cls, "count", count)
        setattr(cls, "size", size)
        setattr(cls, "var", var)
        setattr(cls, "std", std)
        setattr(cls, "sem", sem)
        setattr(cls, "skew", skew)
        setattr(cls, "kurt", kurt)
        setattr(cls, "kurtosis", kurtosis)
        setattr(cls, "nunique", nunique)
        setattr(cls, "median", median)
        setattr(cls, "all", all)
        setattr(cls, "any", any)


_install()
del _install
