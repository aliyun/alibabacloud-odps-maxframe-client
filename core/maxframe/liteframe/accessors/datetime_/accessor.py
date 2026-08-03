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

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_timedelta64_dtype

from maxframe.liteframe.accessors.datetime_.core import (
    DatetimeMultiColumnHandler,
    DatetimePropertyBaseHandler,
    LiteFrameDatetimeExpr,
    LiteFrameDatetimeMultiExpr,
    datetime_method_to_handlers,
    datetime_multi_method_to_handlers,
)
from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _resolve_column_expr,
)
from maxframe.liteframe.expressions import LiteFrameMultiColumnSubExpr


class DatetimeAccessor:
    """
    Vectorized datetime functions for LiteFrame.

    Provides pandas-style .dt accessor for single-column LiteFrame.
    Multi-column LiteFrames or non-datetime dtypes raise on accessor access.

    Examples
    --------
    >>> df = LiteFrame({"ts": pd.date_range("2020-01-01", periods=2)})
    >>> df.dt.year.execute()
       ts
    0  2020
    1  2020
    """

    def __init__(self, liteframe):
        if len(liteframe.columns) != 1:
            raise ValueError("Cannot apply .dt accessor to multi-column LiteFrame")
        dtype = liteframe.dtypes.iloc[0]
        if not self._is_datetime_like(dtype):
            raise AttributeError("Can only use .dt accessor with datetimelike values")
        self._liteframe = liteframe

    @staticmethod
    def _is_datetime_like(dtype):
        """Check if dtype is datetime-like."""
        return (
            is_datetime64_any_dtype(dtype)
            or isinstance(dtype, pd.DatetimeTZDtype)
            or isinstance(dtype, pd.PeriodDtype)
            or is_timedelta64_dtype(dtype)
        )

    @classmethod
    def _gen_func(cls, method, is_property=True):
        """Generate property/method wrapper with dtype inference."""

        def _inner(self):
            handler = datetime_method_to_handlers.get(
                method, DatetimePropertyBaseHandler
            )
            dtype = handler.infer_dtype(method, self._liteframe.dtypes.iloc[0])

            col_name = self._liteframe.columns[0]
            col_dtype = self._liteframe.dtypes.iloc[0]
            col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)

            expr = LiteFrameDatetimeExpr(
                operand=col_expr,
                method=method,
                is_property=is_property,
                dtype=dtype,
            )

            named_expr = expr.rename(self._liteframe.columns[0])

            return _build_fused_projection(self._liteframe, [named_expr])

        return _inner

    @classmethod
    def _register(cls, method, is_property=True):
        """Register method/property to accessor class."""
        func = cls._gen_func(method, is_property)
        if is_property:
            func = property(func)
        setattr(cls, method, func)

    @classmethod
    def _gen_multi_func(cls, method, is_property=False):
        """Generate method wrapper for multi-column datetime operations."""

        def _inner(self):
            handler = datetime_multi_method_to_handlers.get(
                method, DatetimeMultiColumnHandler
            )
            dtypes_map = handler.infer_dtypes(method, self._liteframe.dtypes.iloc[0])
            output_names = list(dtypes_map.keys())

            col_name = self._liteframe.columns[0]
            col_dtype = self._liteframe.dtypes.iloc[0]
            col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)

            multi_expr = LiteFrameDatetimeMultiExpr(
                operand=col_expr,
                method=method,
                is_property=is_property,
                output_names=output_names,
                dtypes_dict={k: str(v) for k, v in dtypes_map.items()},
            )

            projections = []
            for name in output_names:
                sub_expr = LiteFrameMultiColumnSubExpr(
                    multi_expr=multi_expr,
                    output_name=name,
                    dtype=dtypes_map[name],
                )
                named = sub_expr.rename(name)
                projections.append(named)

            return _build_fused_projection(self._liteframe, projections)

        return _inner

    @classmethod
    def _register_multi(cls, method, is_property=False):
        """Register multi-column method to accessor class."""
        func = cls._gen_multi_func(method, is_property)
        setattr(cls, method, func)
