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
from maxframe.serialization.serializables import BoolField, StringField
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
    # Arrow dtype properties - Polars .dt.date returns Date,
    # Polars .dt.time returns Time, both map to ArrowDtype.
    # For .dt.time, the time64 unit ("us") is not semantically
    # significant here: the codegen skips the cast for time64
    # because pl.Time takes no unit argument in Polars, and Polars
    # .dt.time() already returns pl.Time natively.
    "date": wrap_arrow_dtype(pa.date32()),
    "time": wrap_arrow_dtype(pa.time64("us")),
}


class LiteFrameDatetimeExpr(LiteFrameUnaryExpr):
    """Expression for datetime property operations on LiteFrame columns."""

    _op_type_ = opcodes.DATETIME_METHOD  # Reuse existing opcode

    method = StringField("method")
    is_property = BoolField("is_property", default=True)


class DatetimePropertyHandler:
    """Base handler for datetime property dtype inference."""

    @staticmethod
    def infer_dtype(method: str, input_dtype):
        """Infer output dtype for this datetime property."""
        return DATETIME_PROPERTY_OUTPUT_DTYPE.get(method, object)


class DatetimePropertyBaseHandler(DatetimePropertyHandler):
    """Default handler for standard datetime properties with direct Polars mapping."""

    pass


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
}


class DatetimeMultiColumnHandler:
    """Handler for multi-column datetime operations."""

    @staticmethod
    def infer_dtypes(method: str, input_dtype) -> dict:
        """Return {name: dtype} mapping for all output columns."""
        return DATETIME_MULTI_METHOD_OUTPUT_DTYPES.get(method, {})


# Multi-column method handler registry
datetime_multi_method_to_handlers = {
    "isocalendar": DatetimeMultiColumnHandler,
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
    "date": DatetimePropertyBaseHandler,
    "time": DatetimePropertyBaseHandler,
}
