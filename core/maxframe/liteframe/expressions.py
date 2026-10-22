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

import datetime
import decimal
import operator

import numpy as np
import pandas as pd
import pyarrow as pa

from maxframe import opcodes
from maxframe.core import TILEABLE_TYPE
from maxframe.core.operator.base import SchedulingHint
from maxframe.liteframe.datatypes import is_decimal_dtype
from maxframe.liteframe.utils import normalize_col_name
from maxframe.serialization.serializables import (
    AnyField,
    DataTypeField,
    DictField,
    FunctionField,
    Int32Field,
    ListField,
    ReferenceField,
    Serializable,
    StringField,
    TupleField,
)
from maxframe.udf import BuiltinFunction
from maxframe.utils import classproperty, to_pandas_dtype, wrap_arrow_dtype


class SourceCompatibilityError(ValueError):
    """Raised when binary operation involves LiteFrame columns from different sources."""

    def __init__(self, left_source_key, right_source_key):
        super().__init__(
            f"Binary operation between LiteFrame columns from different sources: "
            f"left source={left_source_key}, right source={right_source_key}. "
            f"Both operands must come from the same LiteFrame."
        )


def _is_bool_arrow_dtype(dtype):
    """Return True if *dtype* is an Arrow-backed boolean dtype."""
    pa_type = getattr(dtype, "pyarrow_dtype", None)
    try:
        return pa_type is not None and pa.types.is_boolean(pa_type)
    except Exception:
        return False


def _promote_numpy_dtypes(left_dtype, right_dtype, fallback):
    """Promote two dtypes via numpy ``result_type``, wrapped as Arrow dtype.

    Returns *fallback* when either dtype cannot be converted to a numpy
    dtype.
    """
    try:
        left_np = pd.api.types.pandas_dtype(left_dtype).numpy_dtype
        right_np = pd.api.types.pandas_dtype(right_dtype).numpy_dtype
        result_np = np.result_type(left_np, right_np)
        return wrap_arrow_dtype(pa.from_numpy_dtype(result_np))
    except (TypeError, AttributeError, NotImplementedError):
        return fallback


def _infer_result_dtype(left_dtype, right_dtype, op_cls):
    """Infer result dtype for binary operations.

    This is a basic fallback implementation. The full version with decimal
    precision rules is provided by maxframe.liteframe.arithmetic.core and
    will replace this function when that module is loaded.
    """
    # Comparison ops always return bool; bitwise And/Or/Xor are dtype-aware
    # (bool & bool -> bool, int & int -> int) via numpy result_type below.
    if issubclass(op_cls, _COMPARISON_BINARY_OPS):
        return wrap_arrow_dtype(pa.bool_())
    # True division always produces float (pandas/polars/numpy semantics):
    # int / int -> float64, not int64. Without this, the numpy result_type
    # fallback below would return int64 for int/int division, which is wrong
    # (e.g. groupby mean = sum/count would be planned as int instead of float).
    if issubclass(op_cls, LiteFrameDiv):
        # Decimal division does not produce float64 (polars/pandas return a
        # Decimal dtype); defer to the full promotion path in
        # arithmetic.core (Hive formula) so the declared dtype matches the
        # executing engine.  Returning None lets the caller fall through.
        if is_decimal_dtype(left_dtype) or is_decimal_dtype(right_dtype):
            return None
        return wrap_arrow_dtype(pa.float64())
    # Numeric type promotion via numpy result_type
    return _promote_numpy_dtypes(left_dtype, right_dtype, left_dtype)


def _infer_unary_result_dtype(operand_dtype, op_cls):
    """Infer result dtype for unary operations.

    Basic fallback — full version in maxframe.liteframe.arithmetic.core.
    """
    if issubclass(op_cls, LiteFrameIsNA):
        return wrap_arrow_dtype(pa.bool_())
    if issubclass(op_cls, LiteFrameLogicalNot):
        # np.logical_not always returns booleans for any input dtype.
        return wrap_arrow_dtype(pa.bool_())
    if issubclass(op_cls, LiteFrameInvert):
        # Bitwise NOT: bool -> bool (logical not), integer -> preserve dtype.
        if _is_bool_arrow_dtype(operand_dtype):
            return wrap_arrow_dtype(pa.bool_())
        return operand_dtype
    return operand_dtype


class LiteFrameBaseExpr(Serializable):
    __slots__ = ("_has_custom_code",)

    _op_type_ = None

    scheduling_hint = ReferenceField("scheduling_hint", SchedulingHint, default=None)

    def __hash__(self):
        return super().__hash__()

    @property
    def has_custom_code(self) -> bool:
        cached = getattr(self, "_has_custom_code", None)
        if cached is None:
            cached = self._compute_has_custom_code()
            self._has_custom_code = cached
        return cached

    def _compute_has_custom_code(self) -> bool:
        return False

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


_ISIN_REJECTED_MEMBER_TYPES = (
    list,
    tuple,
    dict,
    set,
    TILEABLE_TYPE,
    LiteFrameBaseExpr,
)

_ISIN_NOT_SCALAR_MSG = (
    "isin values must be a flat list of scalars; "
    "tileables, expressions, and nested containers are not supported"
)


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

    # Bitwise shift operators
    def __lshift__(self, other):
        return self._binary_op(other, LiteFrameLShift)

    def __rlshift__(self, other):
        return self._rbinary_op(other, LiteFrameLShift)

    def __rshift__(self, other):
        return self._binary_op(other, LiteFrameRShift)

    def __rrshift__(self, other):
        return self._rbinary_op(other, LiteFrameRShift)

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

    def __pos__(self):
        # Unary ``+`` is a numeric no-op (matching numpy / pandas); return the
        # expression unchanged so ``lf_eval('+A')`` parses via ast.UAdd.
        return self

    def __abs__(self):
        return LiteFrameAbs(operand=self, dtype=self.dtype)

    def isin(self, values):
        """Return a boolean expression testing membership in *values*.

        Only a constant list of scalars is supported.  Passing a tileable
        (DataFrame, Series, Tensor, LiteFrame, ...) or another LiteFrame
        expression raises ``NotImplementedError``.

        Parameters
        ----------
        values : list-like of scalars
            Constant values to test membership against.

        Returns
        -------
        LiteFrameIsInExpr
            Boolean expression.
        """
        # Reject bare tileables/expressions passed as *values*, then reject
        # any member that is a nested container, tileable, or expression.
        # The documented contract is a flat list of scalars; invalid members
        # would otherwise fail much later during codegen.
        if isinstance(values, (dict, set, TILEABLE_TYPE, LiteFrameBaseExpr)):
            raise NotImplementedError(_ISIN_NOT_SCALAR_MSG)
        if isinstance(values, (str, bytes)):
            values = [values]
        else:
            values = list(values)
        for v in values:
            if isinstance(v, _ISIN_REJECTED_MEMBER_TYPES):
                raise NotImplementedError(_ISIN_NOT_SCALAR_MSG)
        return LiteFrameIsInExpr(operand=self, values=values, dtype=None)


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
        # Preserve an already-resolved dtype (e.g. LiteFrameDatetimeExpr
        # infers int32/bool/string from the accessor method) instead of
        # re-inferring from the operand — _infer_unary_result_dtype only
        # knows about IsNA/LogicalNot/Invert and falls back to the input
        # dtype, which would overwrite a correctly inferred result type
        # (e.g. .dt.year returning timestamp instead of int32).
        if getattr(self, "dtype", None) is not None:
            result_dtype = self.dtype
        else:
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


class LiteFrameIsInExpr(LiteFrameUnaryExpr):
    """Membership test against a constant list of scalars (``col.isin([...])``).

    Always produces a boolean result.  Only constant scalar lists are
    supported; tileables and LiteFrame expressions are rejected at
    construction (see :meth:`LiteFrameExpr.isin`).
    """

    _op_type_ = opcodes.ISIN
    _fixed_result_type = np.dtype(bool)

    values = ListField("values", default=None)

    def bind(self, dtypes):
        bound_operand = self.operand.bind(dtypes)
        return LiteFrameIsInExpr(
            operand=bound_operand,
            values=list(self.values) if self.values is not None else None,
            dtype=wrap_arrow_dtype(pa.bool_()),
        )


class LiteFrameLogicalNot(LiteFrameUnaryExpr):
    """Logical NOT (always boolean), for ``np.logical_not``.

    Distinct from :class:`LiteFrameInvert` (the ``~`` operator), which is
    dtype-aware bitwise NOT (bool -> bool, integer -> integer). NumPy's
    ``logical_not`` must return booleans for *any* input dtype, so this
    expression always yields ``bool``.
    """

    _op_type_ = opcodes.NOT
    _fixed_result_type = np.dtype(bool)


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
    _fixed_result_type = np.dtype(bool)

    @classproperty
    def _operator(self):
        return operator.eq


class LiteFrameNe(LiteFrameBinaryExpr):
    _op_type_ = opcodes.NE
    _fixed_result_type = np.dtype(bool)

    @classproperty
    def _operator(self):
        return operator.ne


class LiteFrameLt(LiteFrameBinaryExpr):
    _op_type_ = opcodes.LT
    _fixed_result_type = np.dtype(bool)

    @classproperty
    def _operator(self):
        return operator.lt


class LiteFrameLe(LiteFrameBinaryExpr):
    _op_type_ = opcodes.LE
    _fixed_result_type = np.dtype(bool)

    @classproperty
    def _operator(self):
        return operator.le


class LiteFrameGt(LiteFrameBinaryExpr):
    _op_type_ = opcodes.GT
    _fixed_result_type = np.dtype(bool)

    @classproperty
    def _operator(self):
        return operator.gt


class LiteFrameGe(LiteFrameBinaryExpr):
    _op_type_ = opcodes.GE
    _fixed_result_type = np.dtype(bool)

    @classproperty
    def _operator(self):
        return operator.ge


class LiteFrameAnd(LiteFrameBinaryExpr):
    _op_type_ = opcodes.AND
    _fixed_result_type = np.dtype(bool)

    @classproperty
    def _operator(self):
        return operator.and_


class LiteFrameOr(LiteFrameBinaryExpr):
    _op_type_ = opcodes.OR
    _fixed_result_type = np.dtype(bool)

    @classproperty
    def _operator(self):
        return operator.or_


class LiteFrameXor(LiteFrameBinaryExpr):
    _op_type_ = opcodes.XOR

    @classproperty
    def _operator(self):
        return operator.xor


class LiteFrameLShift(LiteFrameBinaryExpr):
    _op_type_ = opcodes.LSHIFT

    @classproperty
    def _operator(self):
        return operator.lshift


class LiteFrameRShift(LiteFrameBinaryExpr):
    _op_type_ = opcodes.RSHIFT

    @classproperty
    def _operator(self):
        return operator.rshift


class LiteFrameFillNA(LiteFrameBinaryExpr):
    _op_type_ = opcodes.FILL_NA

    @classproperty
    def _operator(self):
        # fillna is not a standard operator; use lambda that applies Series.fillna
        return lambda left, right: left.fillna(right)


# Comparison ops always produce a boolean result. Bitwise And/Or/Xor are
# NOT included here: they are dtype-aware (bool & bool -> bool, int & int -> int).
_BOOL_RESULT_BINARY_OPS = (
    LiteFrameEq,
    LiteFrameNe,
    LiteFrameLt,
    LiteFrameLe,
    LiteFrameGt,
    LiteFrameGe,
)
# Alias kept for clarity in the fallback inference path.
_COMPARISON_BINARY_OPS = _BOOL_RESULT_BINARY_OPS


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


# Ufuncs that produce boolean results
_BOOL_UFUNCS = frozenset({"isfinite", "isinf", "isnan", "signbit"})

# Binary ufuncs that preserve/promote the operand dtype (e.g. np.maximum on
# integer columns returns integer, not float64).
_DTYPE_PRESERVING_BINARY_UFUNCS = frozenset({"maximum", "minimum", "fmax", "fmin"})

# Unary ufuncs whose int->float64 promotion is the classic numpy contract.
# numpy >= 2.0 changed floor/ceil/trunc to preserve the integer dtype, but
# MaxFrame follows the long-standing numpy 1.x behavior (int -> float64) so
# the advertised and executor result dtypes are deterministic across numpy
# versions. rint already returns float64 on both versions; included for
# uniformity so the rounding family is treated together.
_INT_TO_FLOAT_UFUNCS = frozenset({"floor", "ceil", "trunc", "rint"})


def _is_integer_arrow_dtype(dtype):
    """Return True if *dtype* is an Arrow-backed integer dtype."""
    pa_type = getattr(dtype, "pyarrow_dtype", None)
    try:
        return pa_type is not None and pa.types.is_integer(pa_type)
    except Exception:
        return False


def _infer_unary_ufunc_dtype_from_numpy(ufunc_name, input_dtype):
    """Infer a unary numpy ufunc's result dtype by probing numpy.

    Calls the real ``np.<ufunc>`` on a zero-element array of *input_dtype*
    and returns the result dtype wrapped as an Arrow dtype. This matches
    numpy's ufunc dtype contract exactly for every ufunc and input dtype
    (e.g. ``np.floor`` on ``int64`` -> ``float64`` while ``np.square`` on
    ``int64`` -> ``int64``), instead of guessing from a static table.

    Falls back to ``float64`` when the ufunc is not a real ``np.ufunc``
    (e.g. ``np.spacing``) or when *input_dtype* cannot be probed.

    Parameters
    ----------
    ufunc_name : str
        Name of the numpy ufunc (e.g. ``"floor"``, ``"square"``).
    input_dtype : dtype
        The resolved operand dtype.

    Returns
    -------
    dtype
        Inferred result dtype, wrapped as an Arrow dtype.
    """
    np_ufunc = getattr(np, ufunc_name, None)
    if not isinstance(np_ufunc, np.ufunc):
        return wrap_arrow_dtype(pa.float64())
    try:
        np_dtype = pd.api.types.pandas_dtype(input_dtype).numpy_dtype
    except (TypeError, AttributeError, NotImplementedError, ValueError):
        return wrap_arrow_dtype(pa.float64())
    try:
        result_np = np_ufunc(np.empty(0, dtype=np_dtype)).dtype
    except (TypeError, ValueError):
        return wrap_arrow_dtype(pa.float64())
    try:
        return wrap_arrow_dtype(pa.from_numpy_dtype(result_np))
    except (TypeError, ValueError):
        return wrap_arrow_dtype(pa.float64())


def _infer_ufunc_result_dtype(ufunc_name, input_dtype, right_dtype=None):
    """Infer result dtype for a numpy ufunc.

    For unary ufuncs only *input_dtype* is used: the result dtype is probed
    from numpy (see :func:`_infer_unary_ufunc_dtype_from_numpy`) so it
    matches numpy's ufunc contract exactly. For binary ufuncs
    *right_dtype* is the second operand's dtype and the result is inferred
    from both operands, mirroring :func:`_infer_result_dtype`: boolean ufuncs
    return bool, dtype-preserving ufuncs (maximum/minimum/fmax/fmin) use
    numpy ``result_type`` promotion so integer operands keep an integer
    dtype, and the remaining ufuncs default to ``float64`` (the numpy result
    for transcendental binary ufuncs such as arctan2/hypot).

    Parameters
    ----------
    ufunc_name : str
        Name of the numpy ufunc (e.g. ``"exp"``, ``"maximum"``).
    input_dtype : dtype
        The (left) operand dtype.
    right_dtype : dtype, optional
        The right operand dtype for binary ufuncs.

    Returns
    -------
    dtype
        Inferred result dtype.
    """
    if ufunc_name in _BOOL_UFUNCS:
        return wrap_arrow_dtype(pa.bool_())
    if right_dtype is None:
        # floor/ceil/trunc/rint on an integer column deterministically
        # produce float64: numpy >= 2.0 preserves the int dtype, but
        # MaxFrame follows the classic numpy 1.x contract so the advertised
        # dtype is version-independent.
        if ufunc_name in _INT_TO_FLOAT_UFUNCS and _is_integer_arrow_dtype(input_dtype):
            return wrap_arrow_dtype(pa.float64())
        # Unary path: probe numpy for the actual result dtype so the
        # advertised dtype matches numpy's ufunc contract (e.g. np.floor on
        # float64 -> float64, np.square on int64 -> int64).
        return _infer_unary_ufunc_dtype_from_numpy(ufunc_name, input_dtype)
    # Binary path: infer from both operand dtypes.
    if ufunc_name in _DTYPE_PRESERVING_BINARY_UFUNCS:
        return _promote_numpy_dtypes(input_dtype, right_dtype, input_dtype)
    # Transcendental / float-producing binary ufuncs (arctan2, hypot,
    # copysign, nextafter, ldexp, logaddexp, logaddexp2).
    return wrap_arrow_dtype(pa.float64())


class LiteFrameUnaryUfuncExpr(LiteFrameUnaryExpr):
    """Generic unary ufunc expression. Stores ufunc name for codegen dispatch."""

    _op_type_ = opcodes.LITEFRAME_UNARY_UFUNC
    ufunc_name = StringField("ufunc_name")

    def bind(self, dtypes):
        bound_operand = self.operand.bind(dtypes)
        result_dtype = _infer_ufunc_result_dtype(self.ufunc_name, bound_operand.dtype)
        new_expr = self.copy()
        new_expr.operand = bound_operand
        new_expr.dtype = result_dtype
        return new_expr


class LiteFrameBinUfuncExpr(LiteFrameBinaryExpr):
    """Generic binary ufunc expression. Stores ufunc name for codegen dispatch."""

    _op_type_ = opcodes.LITEFRAME_BIN_UFUNC
    ufunc_name = StringField("ufunc_name")

    def bind(self, dtypes):
        bound_left = self.left.bind(dtypes)
        bound_right = self.right.bind(dtypes)
        result_dtype = _infer_ufunc_result_dtype(
            self.ufunc_name, bound_left.dtype, bound_right.dtype
        )
        return type(self)(
            left=bound_left,
            right=bound_right,
            dtype=result_dtype,
            ufunc_name=self.ufunc_name,
        )


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


class LiteFrameTreeAnd(LiteFrameNaryExpr):
    """N-ary AND expression for combining multiple boolean conjuncts."""

    _op_type_ = opcodes.TREE_AND

    def bind(self, dtypes):
        bound_operands = [op.bind(dtypes) for op in self.operands]
        return type(self)(operands=bound_operands, dtype=wrap_arrow_dtype(pa.bool_()))


class LiteFrameTreeOr(LiteFrameNaryExpr):
    """N-ary OR expression for combining multiple boolean disjuncts."""

    _op_type_ = opcodes.TREE_OR

    def bind(self, dtypes):
        bound_operands = [op.bind(dtypes) for op in self.operands]
        return type(self)(operands=bound_operands, dtype=wrap_arrow_dtype(pa.bool_()))


class NamedExpr(LiteFrameExpr):
    _op_type_ = opcodes.LITEFRAME_NAMED_EXPR

    name = StringField("name")
    expr = AnyField("expr")

    def _compute_has_custom_code(self) -> bool:
        if isinstance(self.expr, LiteFrameBaseExpr):
            return self.expr.has_custom_code
        return False

    def bind(self, dtypes):
        bound_inner = self.expr.bind(dtypes)
        name = normalize_col_name(self.name)
        return NamedExpr(name=name, expr=bound_inner, dtype=bound_inner.dtype)


def is_always_boolean_expr(expr) -> bool:
    """Return True if *expr* always produces a boolean result.

    Single source of truth for boolean-producing expression detection.
    Classes declaring ``_fixed_result_type = np.dtype(bool)`` (the six
    comparisons, ``LiteFrameIsInExpr``, ``LiteFrameLogicalNot``,
    ``LiteFrameAnd``, ``LiteFrameOr``) are always boolean. ``LiteFrameInvert``
    (``~``) is *not* marked: it is dtype-aware (bool -> bool, int -> int),
    so callers must handle it separately.

    Recurses into :class:`NamedExpr` wrappers.
    """
    if isinstance(expr, NamedExpr):
        return is_always_boolean_expr(expr.expr)
    return getattr(type(expr), "_fixed_result_type", None) is np.dtype(bool)


class LiteFrameMultiColumnExpr(LiteFrameBaseExpr):
    _op_type_ = opcodes.LITEFRAME_MULTI_COLUMN_EXPR

    # Input expression(s) the multi-column operation reads from.  For UDF
    # exprs these are the input LiteFrameColumn nodes (replacing the former
    # single ``operand`` placeholder + ``input_columns`` field); for datetime
    # multi-column exprs it is the single source column.
    operands = ListField("operands")
    output_names = ListField("output_names")
    dtypes = DictField("dtypes")

    def __init__(self, *args, **kwargs):
        # Backward compatibility: the input field was renamed from a single
        # ``operand`` (AnyField) to ``operands`` (ListField).  Mixed-version
        # serialized plans (and any code still constructing with ``operand=``)
        # would otherwise raise ``KeyError: 'operand'`` because this class no
        # longer has an ``operand`` field.  Map a legacy single ``operand`` to
        # ``[operand]`` so old plans deserialize/construct unchanged.  An
        # explicit ``operands`` always takes precedence.
        # An explicit ``operands`` always takes precedence; the legacy
        # ``operand`` is otherwise mapped to ``[operand]``.  Either way the
        # legacy key is consumed here so it never reaches ``super().__init__``
        # (which would raise ``KeyError: 'operand'`` -- no such field).
        if "operand" in kwargs:
            legacy = kwargs.pop("operand")
            kwargs.setdefault("operands", [legacy])
        super().__init__(*args, **kwargs)

    def bind(self, dtypes):
        bound_operands = [op.bind(dtypes) for op in self.operands]
        kwargs = dict(
            operands=bound_operands,
            output_names=self.output_names,
            dtypes=self.dtypes,
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

    def _compute_has_custom_code(self) -> bool:
        if isinstance(self.multi_expr, LiteFrameBaseExpr):
            return self.multi_expr.has_custom_code
        return False

    def bind(self, dtypes):
        bound_multi = self.multi_expr.bind(dtypes)
        dtype_str = bound_multi.dtypes[self.output_name]
        dtype = to_pandas_dtype(dtype_str)
        return LiteFrameMultiColumnSubExpr(
            multi_expr=bound_multi,
            output_name=self.output_name,
            dtype=dtype,
        )


class LiteFrameUDFExpr(LiteFrameMultiColumnExpr):
    """Expression node for UDF-based column transformations.

    Embeds a user function (or BuiltinFunction) in a LiteFrameProjection.
    Used for map, transform_chunk, and transform(axis=1).
    """

    _op_type_ = opcodes.LITEFRAME_UDF_EXPR

    func = FunctionField("func")
    mode = StringField("mode")  # "map", "transform_chunk", "transform_row"
    args = TupleField("args", default=())
    kwargs = DictField("kwargs", default={})
    input_frame_type = StringField("input_frame_type", default=None)
    check_output_dtypes = StringField("check_output_dtypes", default="raises")

    def _compute_has_custom_code(self) -> bool:
        return not isinstance(self.func, BuiltinFunction)

    @property
    def dtype(self):
        # map() uses a bare LiteFrameUDFExpr directly as a projection (one
        # output column), unlike transform modes which wrap each output in a
        # LiteFrameMultiColumnSubExpr that carries a .dtype.  Expose the
        # single output dtype here so dtype extraction (e.g. during
        # rename()/assign() fusion) succeeds for map-mode projections.
        # For other modes the property is not accessed (the UDF expr is
        # wrapped in LiteFrameMultiColumnSubExpr); preserve the prior
        # "no dtype" behavior by raising AttributeError.
        if self.mode == "map" and self.output_names:
            return to_pandas_dtype(self.dtypes[self.output_names[0]])
        raise AttributeError(
            f"{type(self).__name__!r} object has no dtype in " f"{self.mode!r} mode"
        )

    def bind(self, dtypes):
        bound_operands = [op.bind(dtypes) for op in self.operands]
        return LiteFrameUDFExpr(
            func=self.func,
            operands=bound_operands,
            output_names=self.output_names,
            dtypes=self.dtypes,
            mode=self.mode,
            args=self.args,
            kwargs=self.kwargs,
            input_frame_type=self.input_frame_type,
            check_output_dtypes=self.check_output_dtypes,
            scheduling_hint=self.scheduling_hint,
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

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> from maxframe.liteframe import col
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    Combine columns in an expression:

    >>> result = lf.select(col('a') + col('b'))
    >>> result.execute()
       _expr_0
    0        5
    1        7
    2        9

    Compare columns:

    >>> result = lf.select(col('a') > col('b'))
    >>> result.execute()
       _expr_0
    0    False
    1    False
    2    False
    """
    return LiteFrameColumn(name=normalize_col_name(name), dtype=None)


# numpy datetime64/timedelta64 units: {ms,us,ns} are Polars-native; coarse
# units (D/h/m/s, including 's' which Polars cannot cast to) map to 'us';
# Y/M are ambiguous and rejected; sub-ns maps to 'ns'.
_PA_TIME_UNITS = frozenset({"ms", "us", "ns"})


def _np_time_unit_to_pa(unit: str) -> str:
    if unit in _PA_TIME_UNITS:
        return unit
    if unit in ("Y", "M"):
        raise TypeError(
            f"LiteFrameLiteral does not support numpy temporal unit {unit!r} "
            f"(ambiguous calendar unit); use 'D', 'h', 's', 'ms', 'us', or 'ns'"
        )
    if unit in ("D", "h", "m", "s"):
        return "us"
    return "ns"


def _infer_literal_dtype(value):
    """Infer pyarrow-based ArrowDtype from a Python literal value."""
    # np.timedelta64 subclasses np.integer and .item() returns an epoch int
    # for ns units; infer the Arrow dtype from the numpy unit instead.
    if isinstance(value, np.datetime64):
        unit = _np_time_unit_to_pa(np.datetime_data(value.dtype)[0])
        return wrap_arrow_dtype(pa.timestamp(unit))
    if isinstance(value, np.timedelta64):
        unit = _np_time_unit_to_pa(np.datetime_data(value.dtype)[0])
        return wrap_arrow_dtype(pa.duration(unit))
    if isinstance(value, (bool, np.bool_)):
        return wrap_arrow_dtype(pa.bool_())
    if isinstance(value, (int, np.integer)):
        return wrap_arrow_dtype(pa.int64())
    if isinstance(value, (float, np.floating)):
        return wrap_arrow_dtype(pa.float64())
    if isinstance(value, str):
        return wrap_arrow_dtype(pa.string())
    if isinstance(value, (bytes, bytearray)):
        return wrap_arrow_dtype(pa.binary())
    # Temporal / decimal types.  Ordering matters: pd.Timestamp subclasses
    # datetime.datetime, which subclasses datetime.date.  pd.NA / None are
    # null sentinels with no inherent dtype.
    if isinstance(value, pd.Timestamp):
        tz = value.tzinfo
        if tz is None:
            return wrap_arrow_dtype(pa.timestamp("ns"))
        # Pass the timezone as a string, not the tzinfo object: pyarrow < 3
        # stores a fixed offset for datetime.timezone instances (e.g.
        # ``tz=+00:00`` for ``datetime.timezone.utc``) while later versions
        # normalise to the zone name.  Using the name string (``"UTC"``)
        # produces the same ``timestamp[ns, tz=UTC]`` across versions.
        tz_name = getattr(tz, "zone", str(tz))
        return wrap_arrow_dtype(pa.timestamp("ns", tz=tz_name))
    if isinstance(value, datetime.datetime):
        tz = value.tzinfo
        if tz is not None:
            # Mirror the pd.Timestamp branch: pass the timezone as a name
            # string so pyarrow produces a stable ``timestamp[us, tz=...]``
            # across versions (see the pd.Timestamp comment above).
            tz_name = getattr(tz, "zone", str(tz))
            return wrap_arrow_dtype(pa.timestamp("us", tz=tz_name))
        return wrap_arrow_dtype(pa.timestamp("us"))
    if isinstance(value, datetime.date):
        return wrap_arrow_dtype(pa.date32())
    if isinstance(value, datetime.time):
        return wrap_arrow_dtype(pa.time64("us"))
    if isinstance(value, datetime.timedelta):
        return wrap_arrow_dtype(pa.duration("us"))
    if isinstance(value, decimal.Decimal):
        t = value.as_tuple()
        scale = -t.exponent
        precision = max(len(t.digits), scale if scale > 0 else 1)
        return wrap_arrow_dtype(pa.decimal128(precision, scale))
    if value is pd.NA or value is None:
        return wrap_arrow_dtype(pa.null())
    # Value already carries an Arrow dtype (pyarrow scalar, or a scalar
    # fetched from an ArrowDtype-backed DataFrame like ArrowExtensionScalar):
    # trust it instead of falling back to string.
    pa_type = getattr(getattr(value, "dtype", None), "pyarrow_dtype", None)
    if pa_type is None and isinstance(value, pa.Scalar):
        pa_type = value.type
    if pa_type is not None:
        # Close the inference/codegen boundary for nested Arrow scalars:
        # _literal_value_to_code serialises only primitive scalars, so a
        # list/struct/map scalar accepted here would fail later at codegen
        # with a confusing TypeError.  Reject it up front with the same
        # message the np.datetime64 rejection uses.
        if isinstance(value, pa.Scalar) and pa.types.is_nested(pa_type):
            raise TypeError(
                f"LiteFrameLiteral does not support nested Arrow scalar "
                f"of type {pa_type}: {value!r}"
            )
        return wrap_arrow_dtype(pa_type)
    return wrap_arrow_dtype(pa.string())
