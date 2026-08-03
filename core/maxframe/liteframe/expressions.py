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

import operator

import numpy as np
import pandas as pd
import pyarrow as pa

from maxframe import opcodes
from maxframe.liteframe.utils import normalize_col_name
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    DataTypeField,
    DictField,
    Int32Field,
    ListField,
    Serializable,
    StringField,
)
from maxframe.utils import classproperty, wrap_arrow_dtype


class SourceCompatibilityError(ValueError):
    """Raised when binary operation involves LiteFrame columns from different sources."""

    def __init__(self, left_source_key, right_source_key):
        super().__init__(
            f"Binary operation between LiteFrame columns from different sources: "
            f"left source={left_source_key}, right source={right_source_key}. "
            f"Both operands must come from the same LiteFrame."
        )


def _infer_result_dtype(left_dtype, right_dtype, op_cls):
    """Infer result dtype for binary operations.

    This is a basic fallback implementation. The full version with decimal
    precision rules is provided by maxframe.liteframe.arithmetic.core and
    will replace this function when that module is loaded.
    """
    # For comparison/logical ops, always return bool
    if issubclass(op_cls, _BOOL_RESULT_BINARY_OPS):
        return wrap_arrow_dtype(pa.bool_())
    # Numeric type promotion via numpy result_type
    try:
        left_np = pd.api.types.pandas_dtype(left_dtype).numpy_dtype
        right_np = pd.api.types.pandas_dtype(right_dtype).numpy_dtype
        result_np = np.result_type(left_np, right_np)
        return wrap_arrow_dtype(pa.from_numpy_dtype(result_np))
    except (TypeError, AttributeError, NotImplementedError):
        return left_dtype


def _infer_unary_result_dtype(operand_dtype, op_cls):
    """Infer result dtype for unary operations.

    Basic fallback — full version in maxframe.liteframe.arithmetic.core.
    """
    if issubclass(op_cls, (LiteFrameIsNA, LiteFrameInvert)):
        return wrap_arrow_dtype(pa.bool_())
    return operand_dtype


class LiteFrameBaseExpr(Serializable):
    _op_type_ = None

    gpu = BoolField("gpu", default=False)

    def __hash__(self):
        return super().__hash__()

    def bind(self, dtypes):  # noqa: ARG002
        """Deep-copy this expression and resolve all None dtypes.

        Walks the expression tree bottom-up, resolving leaf column dtypes
        from the schema and re-inferring intermediate node dtypes.

        Parameters
        ----------
        dtypes : pd.Series
            Mapping of column names to dtypes.

        Returns
        -------
        LiteFrameBaseExpr
            New expression tree with all dtypes resolved.
        """
        return self.copy()


class LiteFrameExpr(LiteFrameBaseExpr):
    _op_type_ = None

    dtype = DataTypeField("dtype")

    @classmethod
    def _wrap_operand(cls, other):
        if isinstance(other, LiteFrameExpr):
            return other
        return LiteFrameLiteral(value=other, dtype=None)

    def _binary_op(self, other, op_cls):
        right = self._wrap_operand(other)
        left_dtype = getattr(self, "dtype", None)
        right_dtype = getattr(right, "dtype", None)
        if left_dtype is not None and right_dtype is not None:
            result_dtype = _infer_result_dtype(left_dtype, right_dtype, op_cls)
        else:
            result_dtype = None
        return op_cls(left=self, right=right, dtype=result_dtype)

    def _rbinary_op(self, other, op_cls):
        left = self._wrap_operand(other)
        left_dtype = getattr(left, "dtype", None)
        right_dtype = getattr(self, "dtype", None)
        if left_dtype is not None and right_dtype is not None:
            result_dtype = _infer_result_dtype(left_dtype, right_dtype, op_cls)
        else:
            result_dtype = None
        return op_cls(left=left, right=self, dtype=result_dtype)

    # Arithmetic operators
    def __add__(self, other):
        return self._binary_op(other, LiteFrameAdd)

    def __radd__(self, other):
        return self._rbinary_op(other, LiteFrameAdd)

    def __sub__(self, other):
        return self._binary_op(other, LiteFrameSub)

    def __rsub__(self, other):
        return self._rbinary_op(other, LiteFrameSub)

    def __mul__(self, other):
        return self._binary_op(other, LiteFrameMul)

    def __rmul__(self, other):
        return self._rbinary_op(other, LiteFrameMul)

    def __truediv__(self, other):
        return self._binary_op(other, LiteFrameDiv)

    def __rtruediv__(self, other):
        return self._rbinary_op(other, LiteFrameDiv)

    def __pow__(self, other):
        return self._binary_op(other, LiteFramePow)

    def __rpow__(self, other):
        return self._rbinary_op(other, LiteFramePow)

    def __floordiv__(self, other):
        return self._binary_op(other, LiteFrameFloorDiv)

    def __rfloordiv__(self, other):
        return self._rbinary_op(other, LiteFrameFloorDiv)

    def __mod__(self, other):
        return self._binary_op(other, LiteFrameMod)

    def __rmod__(self, other):
        return self._rbinary_op(other, LiteFrameMod)

    def sqrt(self):
        return LiteFrameSqrt(operand=self, dtype=wrap_arrow_dtype(pa.float64()))

    # Comparison operators
    def __eq__(self, other):
        return self._binary_op(other, LiteFrameEq)

    def __ne__(self, other):
        return self._binary_op(other, LiteFrameNe)

    def __lt__(self, other):
        return self._binary_op(other, LiteFrameLt)

    def __le__(self, other):
        return self._binary_op(other, LiteFrameLe)

    def __gt__(self, other):
        return self._binary_op(other, LiteFrameGt)

    def __ge__(self, other):
        return self._binary_op(other, LiteFrameGe)

    # Logical / bitwise operators
    def __and__(self, other):
        return self._binary_op(other, LiteFrameAnd)

    def __or__(self, other):
        return self._binary_op(other, LiteFrameOr)

    def __xor__(self, other):
        return self._binary_op(other, LiteFrameXor)

    def rename(self, name):
        """Create a NamedExpr that assigns a name to this expression.

        Parameters
        ----------
        name : str
            The output column name.

        Returns
        -------
        NamedExpr
            A named expression wrapping this expression.
        """
        return NamedExpr(name=normalize_col_name(name), expr=self, dtype=self.dtype)

    def __invert__(self):
        return LiteFrameInvert(operand=self, dtype=self.dtype)

    def __neg__(self):
        return LiteFrameNeg(operand=self, dtype=self.dtype)


class LiteFrameColumn(LiteFrameExpr):
    _op_type_ = opcodes.LITEFRAME_COLUMN

    name = StringField("name")

    def bind(self, dtypes):
        name = normalize_col_name(self.name)
        if getattr(self, "dtype", None) is not None:
            return LiteFrameColumn(name=name, dtype=self.dtype)
        if name not in dtypes.index:
            raise KeyError(f"Column '{name}' not found in frame schema")
        return LiteFrameColumn(name=name, dtype=dtypes[name])


class LiteFrameLiteral(LiteFrameExpr):
    _op_type_ = opcodes.LITEFRAME_LITERAL

    value = AnyField("value")

    def bind(self, dtypes):  # noqa: ARG002
        if getattr(self, "dtype", None) is not None:
            return LiteFrameLiteral(value=self.value, dtype=self.dtype)
        return LiteFrameLiteral(
            value=self.value, dtype=_infer_literal_dtype(self.value)
        )


class LiteFrameUnaryExpr(LiteFrameExpr):
    _op_type_ = None

    operand = AnyField("operand")

    def bind(self, dtypes):
        bound_operand = self.operand.bind(dtypes)
        result_dtype = _infer_unary_result_dtype(bound_operand.dtype, type(self))
        new_expr = self.copy()
        new_expr.operand = bound_operand
        new_expr.dtype = result_dtype
        return new_expr


class LiteFrameIsNA(LiteFrameUnaryExpr):
    _op_type_ = opcodes.ISNAN


class LiteFrameNeg(LiteFrameUnaryExpr):
    _op_type_ = opcodes.NEGATIVE


class LiteFrameAbs(LiteFrameUnaryExpr):
    _op_type_ = opcodes.ABS


class LiteFrameInvert(LiteFrameUnaryExpr):
    _op_type_ = opcodes.INVERT


class LiteFrameRound(LiteFrameUnaryExpr):
    _op_type_ = opcodes.AROUND

    decimals = Int32Field("decimals", default=0)

    def bind(self, dtypes):
        bound_operand = self.operand.bind(dtypes)
        result_dtype = _infer_unary_result_dtype(bound_operand.dtype, type(self))
        return LiteFrameRound(
            operand=bound_operand, dtype=result_dtype, decimals=self.decimals
        )


class LiteFrameAstype(LiteFrameUnaryExpr):
    _op_type_ = opcodes.ASTYPE

    target_dtype = DataTypeField("target_dtype")
    errors = StringField("errors", default="raise")

    def bind(self, dtypes):
        bound_operand = self.operand.bind(dtypes)
        return LiteFrameAstype(
            operand=bound_operand,
            dtype=self.target_dtype,
            target_dtype=self.target_dtype,
            errors=self.errors,
        )


class LiteFrameBinaryExpr(LiteFrameExpr):
    _op_type_ = None
    left = AnyField("left")
    right = AnyField("right")

    def bind(self, dtypes):
        bound_left = self.left.bind(dtypes)
        bound_right = self.right.bind(dtypes)
        result_dtype = _infer_result_dtype(
            bound_left.dtype, bound_right.dtype, type(self)
        )
        return type(self)(left=bound_left, right=bound_right, dtype=result_dtype)


class LiteFrameAdd(LiteFrameBinaryExpr):
    _op_type_ = opcodes.ADD

    @classproperty
    def _operator(self):
        return operator.add


class LiteFrameSub(LiteFrameBinaryExpr):
    _op_type_ = opcodes.SUB

    @classproperty
    def _operator(self):
        return operator.sub


class LiteFrameMul(LiteFrameBinaryExpr):
    _op_type_ = opcodes.MUL

    @classproperty
    def _operator(self):
        return operator.mul


class LiteFrameDiv(LiteFrameBinaryExpr):
    _op_type_ = opcodes.DIV

    @classproperty
    def _operator(self):
        return operator.truediv


class LiteFrameEq(LiteFrameBinaryExpr):
    _op_type_ = opcodes.EQ

    @classproperty
    def _operator(self):
        return operator.eq


class LiteFrameNe(LiteFrameBinaryExpr):
    _op_type_ = opcodes.NE

    @classproperty
    def _operator(self):
        return operator.ne


class LiteFrameLt(LiteFrameBinaryExpr):
    _op_type_ = opcodes.LT

    @classproperty
    def _operator(self):
        return operator.lt


class LiteFrameLe(LiteFrameBinaryExpr):
    _op_type_ = opcodes.LE

    @classproperty
    def _operator(self):
        return operator.le


class LiteFrameGt(LiteFrameBinaryExpr):
    _op_type_ = opcodes.GT

    @classproperty
    def _operator(self):
        return operator.gt


class LiteFrameGe(LiteFrameBinaryExpr):
    _op_type_ = opcodes.GE

    @classproperty
    def _operator(self):
        return operator.ge


class LiteFrameAnd(LiteFrameBinaryExpr):
    _op_type_ = opcodes.AND

    @classproperty
    def _operator(self):
        return operator.and_


class LiteFrameOr(LiteFrameBinaryExpr):
    _op_type_ = opcodes.OR

    @classproperty
    def _operator(self):
        return operator.or_


class LiteFrameXor(LiteFrameBinaryExpr):
    _op_type_ = opcodes.XOR

    @classproperty
    def _operator(self):
        return operator.xor


class LiteFrameFillNA(LiteFrameBinaryExpr):
    _op_type_ = opcodes.FILL_NA

    @classproperty
    def _operator(self):
        # fillna is not a standard operator; use lambda that applies Series.fillna
        return lambda left, right: left.fillna(right)


# Binary ops that always produce a boolean result
_BOOL_RESULT_BINARY_OPS = (
    LiteFrameEq,
    LiteFrameNe,
    LiteFrameLt,
    LiteFrameLe,
    LiteFrameGt,
    LiteFrameGe,
    LiteFrameAnd,
    LiteFrameOr,
    LiteFrameXor,
)


class LiteFramePow(LiteFrameBinaryExpr):
    _op_type_ = opcodes.POW

    @classproperty
    def _operator(self):
        return operator.pow


class LiteFrameFloorDiv(LiteFrameBinaryExpr):
    _op_type_ = opcodes.FLOORDIV

    @classproperty
    def _operator(self):
        return operator.floordiv


class LiteFrameMod(LiteFrameBinaryExpr):
    _op_type_ = opcodes.MOD

    @classproperty
    def _operator(self):
        return operator.mod


class LiteFrameSqrt(LiteFrameUnaryExpr):
    _op_type_ = opcodes.SQRT


class LiteFrameWhere(LiteFrameExpr):
    """Conditional expression: where(condition, true_value, false_value)."""

    _op_type_ = opcodes.WHERE

    condition = AnyField("condition")
    true_value = AnyField("true_value")
    false_value = AnyField("false_value")

    def bind(self, dtypes):
        bound_cond = self.condition.bind(dtypes)
        bound_true = self.true_value.bind(dtypes)
        bound_false = self.false_value.bind(dtypes)
        # Result dtype comes from the true/false branches
        result_dtype = (
            _infer_result_dtype(bound_true.dtype, bound_false.dtype, type(self))
            if bound_true.dtype is not None and bound_false.dtype is not None
            else bound_true.dtype or bound_false.dtype
        )
        return LiteFrameWhere(
            condition=bound_cond,
            true_value=bound_true,
            false_value=bound_false,
            dtype=result_dtype,
        )


class LiteFrameNaryExpr(LiteFrameExpr):
    # Abstract base class for N-ary expressions; concrete subclasses must define _op_type_
    _op_type_ = None

    operands = ListField("operands")

    def bind(self, dtypes):
        bound_operands = [op.bind(dtypes) for op in self.operands]
        result_dtype = bound_operands[0].dtype if bound_operands else None
        return type(self)(operands=bound_operands, dtype=result_dtype)


class NamedExpr(LiteFrameExpr):
    _op_type_ = opcodes.LITEFRAME_NAMED_EXPR

    name = StringField("name")
    expr = AnyField("expr")

    def bind(self, dtypes):
        bound_inner = self.expr.bind(dtypes)
        name = normalize_col_name(self.name)
        return NamedExpr(name=name, expr=bound_inner, dtype=bound_inner.dtype)


class LiteFrameMultiColumnExpr(LiteFrameBaseExpr):
    _op_type_ = opcodes.LITEFRAME_MULTI_COLUMN_EXPR

    operand = AnyField("operand")
    output_names = ListField("output_names")
    dtypes_dict = DictField("dtypes_dict")

    def bind(self, dtypes):
        bound_operand = self.operand.bind(dtypes)
        kwargs = dict(
            operand=bound_operand,
            output_names=self.output_names,
            dtypes_dict=self.dtypes_dict,
        )
        # Copy subclass-specific fields if present (e.g., datetime accessor exprs)
        for attr in ("method", "is_property"):
            if hasattr(self, attr):
                kwargs[attr] = getattr(self, attr)
        return type(self)(**kwargs)


class LiteFrameMultiColumnSubExpr(LiteFrameExpr):
    _op_type_ = opcodes.LITEFRAME_MULTI_COLUMN_SUB_EXPR

    multi_expr = AnyField("multi_expr")
    output_name = StringField("output_name")

    def bind(self, dtypes):
        bound_multi = self.multi_expr.bind(dtypes)
        dtype_str = bound_multi.dtypes_dict[self.output_name]
        dtype = pd.api.types.pandas_dtype(dtype_str)
        return LiteFrameMultiColumnSubExpr(
            multi_expr=bound_multi,
            output_name=self.output_name,
            dtype=dtype,
        )


def col(name):
    """Create a LiteFrameColumn expression without binding dtype.

    The dtype will be resolved later when the expression is bound
    to a LiteFrame via :meth:`LiteFrameColumn.bind`.

    Parameters
    ----------
    name : str
        Column name.

    Returns
    -------
    LiteFrameColumn
        Column expression with dtype=None.
    """
    return LiteFrameColumn(name=normalize_col_name(name), dtype=None)


def _infer_literal_dtype(value):
    """Infer pyarrow-based ArrowDtype from a Python literal value."""
    if isinstance(value, bool):
        return wrap_arrow_dtype(pa.bool_())
    elif isinstance(value, int):
        return wrap_arrow_dtype(pa.int64())
    elif isinstance(value, float):
        return wrap_arrow_dtype(pa.float64())
    elif isinstance(value, str):
        return wrap_arrow_dtype(pa.string())
    else:
        return wrap_arrow_dtype(pa.string())
