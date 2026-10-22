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
import pyarrow as pa
from pandas.api.types import is_datetime64_any_dtype, is_timedelta64_dtype

try:
    from maxframe.lib.dtypes_extension import ArrowDtype
except ImportError:  # pragma: no cover
    ArrowDtype = None

from maxframe.liteframe.accessors._docstring import (
    _adapt_liteframe_docstring,
    _copy_signature,
)
from maxframe.liteframe.accessors.datetime_.core import (
    DatetimeMultiColumnHandler,
    DatetimePropertyBaseHandler,
    DatetimeStringMethodHandler,
    LiteFrameDatetimeExpr,
    LiteFrameDatetimeMultiExpr,
    TimedeltaPropertyHandler,
    TimedeltaTotalSecondsHandler,
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
            # pandas' is_timedelta64_dtype does not recognize ArrowDtype
            # wrapping a pyarrow duration type, so detect it explicitly.
            # Use the project's ArrowDtype (pd.ArrowDtype or FakeArrowDtype)
            # rather than pd.ArrowDtype directly: older pandas (py37-ray)
            # lacks pd.ArrowDtype entirely.
            or (
                ArrowDtype is not None
                and isinstance(dtype, ArrowDtype)
                and pa.types.is_duration(dtype.pyarrow_dtype)
            )
        )

    @staticmethod
    def _is_duration_dtype(dtype):
        """Check if dtype is a timedelta/duration type."""
        if is_timedelta64_dtype(dtype):
            return True
        # pandas' is_timedelta64_dtype does not recognize ArrowDtype
        # wrapping a pyarrow duration type, so detect it explicitly.
        return (
            ArrowDtype is not None
            and isinstance(dtype, ArrowDtype)
            and pa.types.is_duration(dtype.pyarrow_dtype)
        )

    # Methods that pandas only exposes on timedelta columns.  Derived from
    # the handler registry so the classification stays in sync with the
    # dtype-inference layer.  Multi-column ``components`` is also
    # timedelta-only; ``isocalendar`` is datetime-only.
    _DURATION_ONLY_METHODS = frozenset(
        m
        for m, h in datetime_method_to_handlers.items()
        if h in (TimedeltaPropertyHandler, TimedeltaTotalSecondsHandler)
    )
    _DURATION_ONLY_MULTI_METHODS = frozenset({"components"})

    # Methods pandas only exposes on datetime columns (year, month, …,
    # strftime/day_name/month_name, normalize).  ``round``/``floor``/``ceil``
    # are intentionally excluded: pandas supports them on both datetime and
    # timedelta, so they are shared and allowed on either dtype.
    _DATETIME_ONLY_METHODS = frozenset(
        m
        for m, h in datetime_method_to_handlers.items()
        if h in (DatetimePropertyBaseHandler, DatetimeStringMethodHandler)
    ) | frozenset({"normalize"})

    @classmethod
    def _gen_func(cls, method, is_property=True):
        """Generate property/method wrapper with dtype inference."""

        def _inner(self, *args, **kwargs):
            col_dtype = self._liteframe.dtypes.iloc[0]
            is_duration = self._is_duration_dtype(col_dtype)
            # pandas gates accessor methods by dtype in three groups:
            # duration-only (days, seconds, …, total_seconds) raise on
            # datetime; datetime-only (year, …, normalize) raise on
            # timedelta; shared (round/floor/ceil) work on both.  Reject
            # at graph-construction time rather than failing later in DPE.
            if method in self._DURATION_ONLY_METHODS and not is_duration:
                raise AttributeError(
                    f"Can only use .dt.{method} with timedelta-like values"
                )
            if method in self._DATETIME_ONLY_METHODS and is_duration:
                raise AttributeError(
                    f"Can only use .dt.{method} with datetimelike values"
                )
            handler = datetime_method_to_handlers.get(
                method, DatetimePropertyBaseHandler
            )
            dtype = handler.infer_dtype(method, col_dtype)

            col_name = self._liteframe.columns[0]
            col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)

            expr = LiteFrameDatetimeExpr(
                operand=col_expr,
                method=method,
                is_property=is_property,
                args=args,
                kwargs=kwargs,
                dtype=dtype,
            )

            named_expr = expr.rename(self._liteframe.columns[0])

            return _build_fused_projection(self._liteframe, [named_expr])

        pd_method = getattr(pd.Series.dt, method, None)
        if pd_method is not None:
            _inner.__doc__ = _adapt_liteframe_docstring(pd_method.__doc__)
        return _copy_signature(_inner, pd.Series.dt, method)

    @classmethod
    def _register(cls, method, is_property=None):
        """Register method/property to accessor class.

        ``is_property`` defaults to the handler's class attribute so that
        method-style accessors (e.g. .dt.total_seconds()) are registered
        as callables while property-style ones stay as properties.
        """
        if is_property is None:
            handler = datetime_method_to_handlers.get(
                method, DatetimePropertyBaseHandler
            )
            is_property = getattr(handler, "is_property", True)
        func = cls._gen_func(method, is_property)
        if is_property:
            func = property(func)
        setattr(cls, method, func)

    @classmethod
    def _gen_multi_func(cls, method, is_property=False):
        """Generate method wrapper for multi-column datetime operations."""

        def _inner(self):
            col_dtype = self._liteframe.dtypes.iloc[0]
            is_duration = self._is_duration_dtype(col_dtype)
            # Multi-column gating: ``components`` is timedelta-only,
            # ``isocalendar`` is datetime-only.  pandas raises AttributeError
            # for the mismatched dtype; reject at graph-construction time.
            if method in self._DURATION_ONLY_MULTI_METHODS and not is_duration:
                raise AttributeError(
                    f"Can only use .dt.{method} with timedelta-like values"
                )
            if method not in self._DURATION_ONLY_MULTI_METHODS and is_duration:
                raise AttributeError(
                    f"Can only use .dt.{method} with datetimelike values"
                )
            handler = datetime_multi_method_to_handlers.get(
                method, DatetimeMultiColumnHandler
            )
            dtypes_map = handler.infer_dtypes(method, col_dtype)
            output_names = list(dtypes_map.keys())

            col_name = self._liteframe.columns[0]
            col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)

            multi_expr = LiteFrameDatetimeMultiExpr(
                operands=[col_expr],
                method=method,
                is_property=is_property,
                output_names=output_names,
                dtypes={k: str(v) for k, v in dtypes_map.items()},
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

        pd_method = getattr(pd.Series.dt, method, None)
        if pd_method is not None:
            _inner.__doc__ = _adapt_liteframe_docstring(pd_method.__doc__)
        return _copy_signature(_inner, pd.Series.dt, method)

    @classmethod
    def _register_multi(cls, method, is_property=None):
        """Register multi-column method to accessor class.

        ``is_property`` defaults to the handler's class attribute so that
        property-style accessors (e.g. .dt.components) are installed as
        descriptors, while method-style ones (e.g. .dt.isocalendar()) stay
        callables.
        """
        if is_property is None:
            handler = datetime_multi_method_to_handlers.get(
                method, DatetimeMultiColumnHandler
            )
            is_property = getattr(handler, "is_property", False)
        func = cls._gen_multi_func(method, is_property)
        if is_property:
            func = property(func)
        setattr(cls, method, func)
