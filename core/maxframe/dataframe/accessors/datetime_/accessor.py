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

from functools import wraps
from typing import Iterable

import pandas as pd
from pandas.api.types import (
    is_datetime64_dtype,
    is_datetime64tz_dtype,
    is_period_dtype,
    is_timedelta64_dtype,
)

from ....utils import adapt_docstring
from .core import SeriesDatetimeMethod, datetime_method_to_handlers


class DatetimeAccessor:
    def __init__(self, series):
        if (
            not is_datetime64_dtype(series.dtype)
            and not is_datetime64tz_dtype(series.dtype)
            and not is_timedelta64_dtype(series.dtype)
            and not is_period_dtype(series.dtype)
        ):
            raise AttributeError("Can only use .dt accessor with datetimelike values")
        self._series = series

    @classmethod
    def _gen_func(cls, method, is_property):
        @wraps(getattr(pd.Series.dt, method))
        def _inner(self, *args, **kwargs):
            op = SeriesDatetimeMethod(
                method=method,
                is_property=is_property,
                method_args=args,
                method_kwargs=kwargs,
            )
            return op(self._series)

        _inner.__doc__ = adapt_docstring(getattr(pd.Series.dt, method).__doc__)
        return _inner

    @classmethod
    def _register(cls, method):
        is_property = not callable(getattr(pd.Series.dt, method))
        func = cls._gen_func(method, is_property)
        if is_property:
            func = property(func)
        setattr(cls, method, func)

    def __dir__(self) -> Iterable[str]:
        s = set(super().__dir__())
        s.update(datetime_method_to_handlers.keys())
        return list(s)
