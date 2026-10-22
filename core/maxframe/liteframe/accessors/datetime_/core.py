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

import pyarrow as pa

from maxframe import opcodes
from maxframe.liteframe.expressions import LiteFrameMultiColumnExpr, LiteFrameUnaryExpr
from maxframe.serialization.serializables import (
    BoolField,
    DictField,
    StringField,
    TupleField,
)
from maxframe.utils import wrap_arrow_dtype

# Dtype mapping for datetime properties
DATETIME_PROPERTY_OUTPUT_DTYPE = {
    # Integer properties
    "year": wrap_arrow_dtype(pa.int32()),
    "month": wrap_arrow_dtype(pa.int32()),
    "day": wrap_arrow_dtype(pa.int32()),
    "hour": wrap_arrow_dtype(pa.int32()),
    "minute": wrap_arrow_dtype(pa.int32()),
    "second": wrap_arrow_dtype(pa.int32()),
    "microsecond": wrap_arrow_dtype(pa.int32()),
    "nanosecond": wrap_arrow_dtype(pa.int32()),
    "dayofweek": wrap_arrow_dtype(pa.int32()),
    "dayofyear": wrap_arrow_dtype(pa.int32()),
    "weekofyear": wrap_arrow_dtype(pa.int32()),
    "quarter": wrap_arrow_dtype(pa.int32()),
    "days_in_month": wrap_arrow_dtype(pa.int32()),
    "daysinmonth": wrap_arrow_dtype(pa.int32()),
    # Boolean properties
    "is_month_start": wrap_arrow_dtype(pa.bool_()),
    "is_month_end": wrap_arrow_dtype(pa.bool_()),
    "is_quarter_start": wrap_arrow_dtype(pa.bool_()),
    "is_quarter_end": wrap_arrow_dtype(pa.bool_()),
    "is_year_start": wrap_arrow_dtype(pa.bool_()),
    "is_year_end": wrap_arrow_dtype(pa.bool_()),
    "is_leap_year": wrap_arrow_dtype(pa.bool_()),
    # Arrow dtype properties - Polars .dt.date returns Date,
    # Polars .dt.time returns Time, both map to ArrowDtype.
    # For .dt.time, the time64 unit ("us") is not semantically
    # significant here: the codegen skips the cast for time64
    # because pl.Time takes no unit argument in Polars, and Polars
    # .dt.time() already returns pl.Time natively.
    "date": wrap_arrow_dtype(pa.date32()),
    "time": wrap_arrow_dtype(pa.time64("us")),
    # Method-style accessors (strftime, day_name, month_name) return
    # strings; round/floor/ceil/normalize preserve the input datetime
    # dtype.  These entries let the base dtype-inference path resolve
    # them without a dedicated handler.
    "strftime": wrap_arrow_dtype(pa.string()),
    "day_name": wrap_arrow_dtype(pa.string()),
    "month_name": wrap_arrow_dtype(pa.string()),
}

# Dtype mapping for timedelta properties.
# Polars Duration has no .dt.days/.seconds/etc.; the codegen computes
# them as floor-division/modulo of total_nanoseconds().  pandas returns
# int64 for `days` and int32 for the sub-day components.
TIMEDELTA_PROPERTY_OUTPUT_DTYPE = {
    "days": wrap_arrow_dtype(pa.int64()),
    "seconds": wrap_arrow_dtype(pa.int32()),
    "microseconds": wrap_arrow_dtype(pa.int32()),
    "nanoseconds": wrap_arrow_dtype(pa.int32()),
}
TIMEDELTA_TOTAL_SECONDS_OUTPUT_DTYPE = wrap_arrow_dtype(pa.float64())


class LiteFrameDatetimeExpr(LiteFrameUnaryExpr):
    """Expression for datetime property operations on LiteFrame columns."""

    _op_type_ = opcodes.DATETIME_METHOD  # Reuse existing opcode

    method = StringField("method")
    is_property = BoolField("is_property", default=True)
    args = TupleField("args", default=())
    kwargs = DictField("kwargs", default={})


class DatetimePropertyHandler:
    """Base handler for datetime/timedelta property dtype inference."""

    # Whether the pandas accessor exposes this as a property (True) or a
    # method (False).  Subclasses override for method-style accessors.
    is_property = True

    @staticmethod
    def infer_dtype(method: str, input_dtype):
        """Infer output dtype for this datetime property."""
        return DATETIME_PROPERTY_OUTPUT_DTYPE.get(method, object)


class DatetimePropertyBaseHandler(DatetimePropertyHandler):
    """Default handler for standard datetime properties with direct Polars mapping."""

    pass


class DatetimeStringMethodHandler(DatetimePropertyBaseHandler):
    """Handler for method-style string-returning accessors (strftime, day_name, …)."""

    is_property = False


class TimedeltaPropertyHandler(DatetimePropertyHandler):
    """Handler for timedelta-only properties (days, seconds, …)."""

    @staticmethod
    def infer_dtype(method: str, input_dtype):
        return TIMEDELTA_PROPERTY_OUTPUT_DTYPE.get(method, object)


class TimedeltaTotalSecondsHandler(DatetimePropertyHandler):
    """Handler for .dt.total_seconds() — a method returning float64."""

    is_property = False

    @staticmethod
    def infer_dtype(method: str, input_dtype):
        return TIMEDELTA_TOTAL_SECONDS_OUTPUT_DTYPE


class DatetimePreservingHandler(DatetimePropertyHandler):
    """Handler for methods that preserve the input datetime dtype.

    round/floor/ceil/normalize return the same datetime dtype as the
    input column.  The output dtype is taken from ``input_dtype`` so
    the unit (ns/us/ms/s) and Arrow wrapping are preserved.

    Polars ``Duration`` only accepts ``ns``/``us``/``ms``; ``timedelta64[s]``
    is canonicalized to ``duration[ms]`` here so the declared metadata
    matches the physical Polars schema the codegen produces (review
    follow-up: canonicalization lived only in the executor, so metadata
    said ``duration[s]`` while the physical result was ``Duration(ms)``).
    """

    is_property = False

    @staticmethod
    def infer_dtype(method: str, input_dtype):
        try:
            pa_type = input_dtype.pyarrow_dtype
        except AttributeError:
            pa_type = None
        if (
            pa_type is not None
            and pa.types.is_duration(pa_type)
            and pa_type.unit == "s"
        ):
            return wrap_arrow_dtype(pa.duration("ms"))
        return input_dtype


class LiteFrameDatetimeMultiExpr(LiteFrameMultiColumnExpr):
    """Expression for datetime operations that produce multiple columns."""

    _op_type_ = opcodes.DATETIME_MULTI_METHOD

    method = StringField("method")
    is_property = BoolField("is_property", default=True)


# Multi-column datetime method output dtype mapping
DATETIME_MULTI_METHOD_OUTPUT_DTYPES = {
    "isocalendar": {
        "year": wrap_arrow_dtype(pa.int32()),
        "week": wrap_arrow_dtype(pa.int32()),
        "day": wrap_arrow_dtype(pa.int32()),
    },
    # pandas .dt.components returns a DataFrame with 7 int64 columns.
    "components": {
        "days": wrap_arrow_dtype(pa.int64()),
        "hours": wrap_arrow_dtype(pa.int64()),
        "minutes": wrap_arrow_dtype(pa.int64()),
        "seconds": wrap_arrow_dtype(pa.int64()),
        "milliseconds": wrap_arrow_dtype(pa.int64()),
        "microseconds": wrap_arrow_dtype(pa.int64()),
        "nanoseconds": wrap_arrow_dtype(pa.int64()),
    },
}


class DatetimeMultiColumnHandler:
    """Handler for multi-column datetime operations."""

    is_property = False

    @staticmethod
    def infer_dtypes(method: str, input_dtype) -> dict:
        """Return {name: dtype} mapping for all output columns."""
        return DATETIME_MULTI_METHOD_OUTPUT_DTYPES.get(method, {})


class ComponentsHandler(DatetimeMultiColumnHandler):
    """Handler for .dt.components — a property (not a callable method) in pandas."""

    is_property = True


# Multi-column method handler registry
datetime_multi_method_to_handlers = {
    "isocalendar": DatetimeMultiColumnHandler,
    "components": ComponentsHandler,
}


# Handler registry - currently all properties use the base handler
datetime_method_to_handlers = {
    "year": DatetimePropertyBaseHandler,
    "month": DatetimePropertyBaseHandler,
    "day": DatetimePropertyBaseHandler,
    "hour": DatetimePropertyBaseHandler,
    "minute": DatetimePropertyBaseHandler,
    "second": DatetimePropertyBaseHandler,
    "microsecond": DatetimePropertyBaseHandler,
    "nanosecond": DatetimePropertyBaseHandler,
    "dayofweek": DatetimePropertyBaseHandler,
    "dayofyear": DatetimePropertyBaseHandler,
    "weekofyear": DatetimePropertyBaseHandler,
    "quarter": DatetimePropertyBaseHandler,
    "days_in_month": DatetimePropertyBaseHandler,
    "daysinmonth": DatetimePropertyBaseHandler,
    "is_month_start": DatetimePropertyBaseHandler,
    "is_month_end": DatetimePropertyBaseHandler,
    "is_quarter_start": DatetimePropertyBaseHandler,
    "is_quarter_end": DatetimePropertyBaseHandler,
    "is_year_start": DatetimePropertyBaseHandler,
    "is_year_end": DatetimePropertyBaseHandler,
    "is_leap_year": DatetimePropertyBaseHandler,
    "date": DatetimePropertyBaseHandler,
    "time": DatetimePropertyBaseHandler,
    "strftime": DatetimeStringMethodHandler,
    "day_name": DatetimeStringMethodHandler,
    "month_name": DatetimeStringMethodHandler,
    "normalize": DatetimePreservingHandler,
    "round": DatetimePreservingHandler,
    "floor": DatetimePreservingHandler,
    "ceil": DatetimePreservingHandler,
    "days": TimedeltaPropertyHandler,
    "seconds": TimedeltaPropertyHandler,
    "microseconds": TimedeltaPropertyHandler,
    "nanoseconds": TimedeltaPropertyHandler,
    "total_seconds": TimedeltaTotalSecondsHandler,
}
