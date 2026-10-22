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

import numbers

from maxframe.liteframe.core import RangeInfo
from maxframe.liteframe.expressions import (
    LiteFrameAdd,
    LiteFrameBinaryExpr,
    LiteFrameColumn,
    LiteFrameLiteral,
    LiteFrameMul,
    LiteFrameMultiColumnExpr,
    LiteFrameMultiColumnSubExpr,
    LiteFrameNaryExpr,
    LiteFrameNeg,
    LiteFrameSub,
    LiteFrameUnaryExpr,
    LiteFrameWhere,
    NamedExpr,
)
from maxframe.serialization.serializables import (
    BoolField,
    Int64Field,
    Serializable,
    StringField,
)


class RangeAnalysisResult(Serializable):
    """Result of analyzing whether an expression is an affine transform
    of a single range column.

    States:

    - ``is_affine=True, references_range=True``: affine transform, can
      stay virtual with updated metadata.
    - ``is_affine=False, references_range=True``: references a range
      column but NOT affine — must materialize.
    - ``is_affine=False, references_range=False``: no range column
      involvement — execute normally.
    """

    is_affine = BoolField("is_affine", default=False)
    references_range = BoolField("references_range", default=False)
    scale = Int64Field("scale", default=0)
    offset = Int64Field("offset", default=0)
    col_name = StringField("col_name", default=None)


def _non_affine(references_range=False):
    return RangeAnalysisResult(is_affine=False, references_range=references_range)


def analyze_column(name, range_columns):
    """Analyze a column reference for range-affine properties.

    If *name* is in *range_columns*, the column is a range column and
    the result is affine with scale=1, offset=0.
    """
    if name in range_columns:
        return RangeAnalysisResult(
            is_affine=True,
            references_range=True,
            scale=1,
            offset=0,
            col_name=name,
        )
    return _non_affine(references_range=False)


def analyze_literal(value):
    """Analyze a literal value for range-affine properties.

    Integer literals (Python int, numpy integers, etc.) are affine
    constants (scale=0, offset=value). Booleans are excluded to avoid
    treating True/False as 1/0 in arithmetic. Non-integer literals
    (float, str, etc.) are not affine.
    """
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return RangeAnalysisResult(
            is_affine=True,
            references_range=False,
            scale=0,
            offset=int(value),
        )
    return _non_affine(references_range=False)


def _resolve_col(left, right):
    """Return the col_name from L or R, ensuring they don't conflict."""
    if left.col_name is not None and right.col_name is not None:
        if left.col_name != right.col_name:
            return None  # two different range columns
        return left.col_name
    return left.col_name or right.col_name


def _validate_scale(scale, references_range):
    """If scale == 0, the result is a constant, not a range."""
    if scale == 0:
        return _non_affine(references_range=references_range)
    return None  # caller builds the full result


def combine_add(left, right):
    """Combine two RangeAnalysisResult values via addition."""
    if not left.is_affine or not right.is_affine:
        refs = left.references_range or right.references_range
        return _non_affine(references_range=refs)
    col_name = _resolve_col(left, right)
    if col_name is None:
        # col_name is None means either two different range columns
        # conflict, or neither operand references a range column
        # (literal-only arithmetic). Only flag references_range when
        # at least one operand actually references a range column.
        return _non_affine(
            references_range=left.references_range or right.references_range
        )
    scale = left.scale + right.scale
    offset = left.offset + right.offset
    refs = left.references_range or right.references_range
    validated = _validate_scale(scale, refs)
    if validated is not None:
        return validated
    return RangeAnalysisResult(
        is_affine=True,
        references_range=True,
        scale=scale,
        offset=offset,
        col_name=col_name,
    )


def combine_sub(left, right):
    """Combine two RangeAnalysisResult values via subtraction."""
    if not left.is_affine or not right.is_affine:
        refs = left.references_range or right.references_range
        return _non_affine(references_range=refs)
    col_name = _resolve_col(left, right)
    if col_name is None:
        # col_name is None means either two different range columns
        # conflict, or neither operand references a range column
        # (literal-only arithmetic). Only flag references_range when
        # at least one operand actually references a range column.
        return _non_affine(
            references_range=left.references_range or right.references_range
        )
    scale = left.scale - right.scale
    offset = left.offset - right.offset
    refs = left.references_range or right.references_range
    validated = _validate_scale(scale, refs)
    if validated is not None:
        return validated
    return RangeAnalysisResult(
        is_affine=True,
        references_range=True,
        scale=scale,
        offset=offset,
        col_name=col_name,
    )


def combine_mul(left, right):
    """Combine two RangeAnalysisResult values via multiplication."""
    if not left.is_affine or not right.is_affine:
        refs = left.references_range or right.references_range
        return _non_affine(references_range=refs)
    if left.scale == 0 and right.scale == 0:
        return _non_affine(references_range=False)
    if left.scale == 0:
        scale = left.offset * right.scale
        offset = left.offset * right.offset
        validated = _validate_scale(scale, True)
        if validated is not None:
            return validated
        return RangeAnalysisResult(
            is_affine=True,
            references_range=True,
            scale=scale,
            offset=offset,
            col_name=right.col_name,
        )
    if right.scale == 0:
        scale = right.offset * left.scale
        offset = right.offset * left.offset
        validated = _validate_scale(scale, True)
        if validated is not None:
            return validated
        return RangeAnalysisResult(
            is_affine=True,
            references_range=True,
            scale=scale,
            offset=offset,
            col_name=left.col_name,
        )
    return _non_affine(references_range=True)


def combine_neg(operand):
    """Combine a RangeAnalysisResult via negation."""
    if not operand.is_affine:
        return _non_affine(references_range=operand.references_range)
    if not operand.references_range:
        return RangeAnalysisResult(
            is_affine=True,
            references_range=False,
            scale=0,
            offset=-operand.offset,
        )
    return RangeAnalysisResult(
        is_affine=True,
        references_range=True,
        scale=-operand.scale,
        offset=-operand.offset,
        col_name=operand.col_name,
    )


def analyze_range_affine(expr, range_columns):
    """Recursively analyze an expression tree for range-affine transforms.

    Walks the expression tree and returns a :class:`RangeAnalysisResult`
    indicating whether the expression is an affine transform of a single
    range column.
    """
    if isinstance(expr, LiteFrameColumn):
        return analyze_column(expr.name, range_columns)
    if isinstance(expr, LiteFrameLiteral):
        return analyze_literal(expr.value)
    if isinstance(expr, LiteFrameAdd):
        return combine_add(
            analyze_range_affine(expr.left, range_columns),
            analyze_range_affine(expr.right, range_columns),
        )
    if isinstance(expr, LiteFrameSub):
        return combine_sub(
            analyze_range_affine(expr.left, range_columns),
            analyze_range_affine(expr.right, range_columns),
        )
    if isinstance(expr, LiteFrameMul):
        return combine_mul(
            analyze_range_affine(expr.left, range_columns),
            analyze_range_affine(expr.right, range_columns),
        )
    if isinstance(expr, LiteFrameNeg):
        return combine_neg(analyze_range_affine(expr.operand, range_columns))
    if isinstance(expr, LiteFrameUnaryExpr):
        operand_result = analyze_range_affine(expr.operand, range_columns)
        return _non_affine(references_range=operand_result.references_range)
    if isinstance(expr, LiteFrameBinaryExpr):
        left = analyze_range_affine(expr.left, range_columns)
        right = analyze_range_affine(expr.right, range_columns)
        return _non_affine(
            references_range=left.references_range or right.references_range
        )
    if isinstance(expr, LiteFrameWhere):
        cond = analyze_range_affine(expr.condition, range_columns)
        tv = analyze_range_affine(expr.true_value, range_columns)
        fv = analyze_range_affine(expr.false_value, range_columns)
        return _non_affine(
            references_range=cond.references_range
            or tv.references_range
            or fv.references_range
        )
    if isinstance(expr, LiteFrameNaryExpr):
        refs = any(
            analyze_range_affine(o, range_columns).references_range
            for o in expr.operands
        )
        return _non_affine(references_range=refs)
    if isinstance(expr, NamedExpr):
        return analyze_range_affine(expr.expr, range_columns)
    if isinstance(expr, LiteFrameMultiColumnSubExpr):
        sub = analyze_range_affine(expr.multi_expr, range_columns)
        return _non_affine(references_range=sub.references_range)
    if isinstance(expr, LiteFrameMultiColumnExpr):
        refs = any(
            analyze_range_affine(o, range_columns).references_range
            for o in expr.operands
        )
        return _non_affine(references_range=refs)
    return _non_affine(references_range=False)


def transform_range_info(info, scale, offset):
    """Transform a :class:`RangeInfo` by applying ``scale * range + offset``.

    - ``start``/``stop``: ``None`` propagates (unknown bounds).
    - ``step``: ``None`` treated as 1; result is always computable.
    """
    start = None if info.start is None else scale * info.start + offset
    stop = None if info.stop is None else scale * info.stop + offset
    step = scale * (info.step if info.step is not None else 1)
    return RangeInfo(start=start, stop=stop, step=step)
