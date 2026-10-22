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
from functools import reduce
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from maxframe.liteframe.accessors.datetime_.core import (
    LiteFrameDatetimeExpr,
    LiteFrameDatetimeMultiExpr,
)
from maxframe.liteframe.accessors.dict_.core import LiteFrameDictExpr
from maxframe.liteframe.accessors.list_.core import LiteFrameListExpr
from maxframe.liteframe.accessors.string_.core import (
    LiteFrameStrExpr,
    LiteFrameStrMultiExpr,
)
from maxframe.liteframe.accessors.struct_.core import LiteFrameStructExpr
from maxframe.liteframe.expressions import (
    LiteFrameAbs,
    LiteFrameAdd,
    LiteFrameAnd,
    LiteFrameAstype,
    LiteFrameBaseExpr,
    LiteFrameBinUfuncExpr,
    LiteFrameColumn,
    LiteFrameDiv,
    LiteFrameEq,
    LiteFrameFillNA,
    LiteFrameFloorDiv,
    LiteFrameGe,
    LiteFrameGt,
    LiteFrameInvert,
    LiteFrameIsNA,
    LiteFrameLe,
    LiteFrameLiteral,
    LiteFrameLogicalNot,
    LiteFrameLShift,
    LiteFrameLt,
    LiteFrameMod,
    LiteFrameMul,
    LiteFrameMultiColumnSubExpr,
    LiteFrameNe,
    LiteFrameNeg,
    LiteFrameOr,
    LiteFramePow,
    LiteFrameRound,
    LiteFrameRShift,
    LiteFrameSqrt,
    LiteFrameSub,
    LiteFrameTreeAnd,
    LiteFrameTreeOr,
    LiteFrameUnaryUfuncExpr,
    LiteFrameWhere,
    LiteFrameXor,
    NamedExpr,
)
from maxframe.utils import TypeDispatcher

_DISPATCHER = TypeDispatcher()

# Cache attribute for multi-column expression results.
_MULTI_CACHE_ATTR = "_multi_cache"


class LiteFrameExprEvaluator:
    """Evaluates LiteFrame expression trees against a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def eval(self, expr: LiteFrameBaseExpr) -> Any:
        """Evaluate an expression tree and return a pandas Series or scalar."""
        handler = _DISPATCHER.get_handler(type(expr))
        return handler(self, expr)


def _register(expr_cls):
    """Decorator to register an expression handler."""

    def decorator(func):
        _DISPATCHER.register(expr_cls, func)
        return func

    return decorator


# --- Leaf expressions ---


@_register(LiteFrameColumn)
def _eval_column(evaluator, expr):
    return evaluator._df[expr.name]


@_register(LiteFrameLiteral)
def _eval_literal(evaluator, expr):
    return expr.value


# --- Binary expressions ---


_BINARY_OPS = {
    LiteFrameAdd: operator.add,
    LiteFrameSub: operator.sub,
    LiteFrameMul: operator.mul,
    LiteFrameDiv: operator.truediv,
    LiteFrameEq: operator.eq,
    LiteFrameNe: operator.ne,
    LiteFrameLt: operator.lt,
    LiteFrameLe: operator.le,
    LiteFrameGt: operator.gt,
    LiteFrameGe: operator.ge,
    LiteFrameAnd: operator.and_,
    LiteFrameOr: operator.or_,
    LiteFrameXor: operator.xor,
    LiteFramePow: operator.pow,
    LiteFrameFloorDiv: operator.floordiv,
    LiteFrameMod: operator.mod,
    LiteFrameLShift: np.left_shift,
    LiteFrameRShift: np.right_shift,
}


def _make_binary_handler(op_func):
    """Create a binary expression handler applying *op_func* to left/right."""

    def _handler(evaluator, expr):
        return op_func(evaluator.eval(expr.left), evaluator.eval(expr.right))

    return _handler


for _cls, _op_func in _BINARY_OPS.items():
    _DISPATCHER.register(_cls, _make_binary_handler(_op_func))


@_register(LiteFrameFillNA)
def _eval_fillna(evaluator, expr):
    return evaluator.eval(expr.left).fillna(evaluator.eval(expr.right))


# --- N-ary boolean expressions ---


@_register(LiteFrameTreeAnd)
def _eval_tree_and(evaluator, expr):
    return reduce(operator.and_, (evaluator.eval(op) for op in expr.operands))


@_register(LiteFrameTreeOr)
def _eval_tree_or(evaluator, expr):
    return reduce(operator.or_, (evaluator.eval(op) for op in expr.operands))


# --- Unary expressions ---


@_register(LiteFrameIsNA)
def _eval_isna(evaluator, expr):
    return evaluator.eval(expr.operand).isna()


@_register(LiteFrameNeg)
def _eval_neg(evaluator, expr):
    return -evaluator.eval(expr.operand)


@_register(LiteFrameAbs)
def _eval_abs(evaluator, expr):
    return evaluator.eval(expr.operand).abs()


@_register(LiteFrameInvert)
def _eval_invert(evaluator, expr):
    return ~evaluator.eval(expr.operand)


@_register(LiteFrameLogicalNot)
def _eval_logical_not(evaluator, expr):
    # np.logical_not: always returns booleans for any input dtype.
    # Non-boolean operands are cast to bool first (0 / 0.0 -> False),
    # matching NumPy's logical_not contract.
    return ~evaluator.eval(expr.operand).astype(bool)


@_register(LiteFrameRound)
def _eval_round(evaluator, expr):
    return evaluator.eval(expr.operand).round(expr.decimals)


@_register(LiteFrameAstype)
def _eval_astype(evaluator, expr):
    target = expr.target_dtype
    if hasattr(target, "pyarrow_dtype"):
        pa_dtype = target.pyarrow_dtype
        np_dtype = pa_dtype.to_pandas_dtype()
        pd_dtype = np.dtype(np_dtype)
    else:
        pd_dtype = str(target)
    return evaluator.eval(expr.operand).astype(pd_dtype, errors=expr.errors)


@_register(LiteFrameSqrt)
def _eval_sqrt(evaluator, expr):
    return np.sqrt(evaluator.eval(expr.operand))


# --- Generic ufunc expressions ---


@_register(LiteFrameUnaryUfuncExpr)
def _eval_unary_ufunc(evaluator, expr):
    ufunc = getattr(np, expr.ufunc_name)
    result = ufunc(evaluator.eval(expr.operand))
    # The advertised dtype (expr.dtype) may differ from the ufunc's native
    # result on this numpy version (e.g. floor/ceil/trunc on int64: numpy
    # >= 2.0 returns int64, but MaxFrame advertises float64). Cast so the
    # SPE result matches the advertised dtype deterministically.
    target = getattr(expr.dtype, "numpy_dtype", None)
    if target is not None and not isinstance(result, (int, float, bool)):
        result = result.astype(target, copy=False)
    return result


@_register(LiteFrameBinUfuncExpr)
def _eval_bin_ufunc(evaluator, expr):
    ufunc = getattr(np, expr.ufunc_name)
    return ufunc(evaluator.eval(expr.left), evaluator.eval(expr.right))


# --- Where expression ---


@_register(LiteFrameWhere)
def _eval_where(evaluator, expr):
    cond = evaluator.eval(expr.condition)
    true_val = evaluator.eval(expr.true_value)
    false_val = evaluator.eval(expr.false_value)
    return pd.Series(np.where(cond, true_val, false_val), index=cond.index)


# --- Named expression ---


@_register(NamedExpr)
def _eval_named(evaluator, expr):
    return evaluator.eval(expr.expr)


# --- Multi-column sub-expression ---


@_register(LiteFrameMultiColumnSubExpr)
def _eval_multi_sub(evaluator, expr):
    multi_expr = expr.multi_expr
    cache = getattr(evaluator, _MULTI_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(evaluator, _MULTI_CACHE_ATTR, cache)
    cache_key = id(multi_expr)
    if cache_key not in cache:
        cache[cache_key] = evaluator.eval(multi_expr)
    result_df = cache[cache_key]
    return result_df[expr.output_name]


# --- Accessor expressions ---


@_register(LiteFrameStrExpr)
def _eval_str_expr(evaluator, expr):
    series = evaluator.eval(expr.operand)
    method = expr.method
    args = expr.args or ()
    kwargs = expr.kwargs or {}
    str_accessor = series.str
    return getattr(str_accessor, method)(*args, **kwargs)


@_register(LiteFrameStrMultiExpr)
def _eval_str_multi_expr(evaluator, expr):
    series = evaluator.eval(expr.operands[0])
    method = expr.method
    args = expr.args or ()
    kwargs = expr.kwargs or {}
    result = getattr(series.str, method)(*args, **kwargs)
    result.columns = expr.output_names
    return result


@_register(LiteFrameDatetimeExpr)
def _eval_datetime_expr(evaluator, expr):
    series = evaluator.eval(expr.operand)
    method = expr.method
    dt_accessor = series.dt
    if expr.is_property:
        return getattr(dt_accessor, method)
    args = getattr(expr, "args", ()) or ()
    kwargs = getattr(expr, "kwargs", {}) or {}
    return getattr(dt_accessor, method)(*args, **kwargs)


@_register(LiteFrameDatetimeMultiExpr)
def _eval_datetime_multi_expr(evaluator, expr):
    series = evaluator.eval(expr.operands[0])
    method = expr.method
    if method == "isocalendar":
        result = series.dt.isocalendar()
        result.columns = expr.output_names
        return result
    if method == "components":
        result = series.dt.components
        result.columns = expr.output_names
        return result
    raise ValueError(f"Unsupported datetime multi method: {method}")


@_register(LiteFrameListExpr)
def _eval_list_expr(evaluator, expr):
    series = evaluator.eval(expr.operand)
    method = expr.method
    args = expr.args or ()
    if method == "len":
        return series.apply(len)
    elif method == "__getitem__":
        idx = args[0]
        return series.apply(lambda v: v[idx])
    elif method == "get":
        idx = args[0]
        return series.apply(lambda v: v[idx] if idx < len(v) else None)
    raise ValueError(f"Unsupported list method: {method}")


@_register(LiteFrameDictExpr)
def _eval_dict_expr(evaluator, expr):
    series = evaluator.eval(expr.operand)
    method = expr.method
    args = expr.args or ()
    kwargs = expr.kwargs or {}
    if method == "len":
        return series.apply(len)
    elif method == "__getitem__":
        key = args[0]
        return series.apply(lambda v: v[key])
    elif method == "get":
        key = args[0]
        default = args[1] if len(args) > 1 else kwargs.get("default_value")
        return series.apply(lambda v: v.get(key, default))
    elif method == "contains":
        key = args[0]
        return series.apply(lambda v: key in v)
    elif method == "remove":
        key = args[0]

        def _remove(v):
            if key not in v:
                raise KeyError(key)
            return {k: val for k, val in v.items() if k != key}

        return series.apply(_remove)
    raise ValueError(f"Unsupported dict method: {method}")


@_register(LiteFrameStructExpr)
def _eval_struct_expr(evaluator, expr):
    series = evaluator.eval(expr.operand)
    method = expr.method
    args = expr.args or ()
    kwargs = expr.kwargs or {}
    if method == "field":
        name = args[0] if args else kwargs.get("name_or_index")
        # Normalize a scalar name/index into a one-element path; the public
        # API stores a list for nested access (e.g. ["inner", "x"]).
        names = name if isinstance(name, list) else [name]
        # Resolve integer struct-field indices to field names and descend the
        # pyarrow struct type one segment at a time, mirroring the DPE
        # codegen.  Each segment is resolved against the current level's type
        # so a nested int index (e.g. ["inner", 0]) also works.
        pa_type = pa.array(series).type
        resolved = []
        for n in names:
            if isinstance(n, int):
                n = pa_type[n].name
            resolved.append(n)
            pa_type = pa_type[n].type if isinstance(n, str) else pa_type[n].type

        def _walk(v):
            for seg in resolved:
                v = v[seg] if isinstance(v, dict) else getattr(v, seg)
            return v

        return series.apply(_walk)
    raise ValueError(f"Unsupported struct method: {method}")


def _materialize_range_columns(df, range_columns):
    """Materialize virtual range columns into *df* and return the frame.

    Each entry in *range_columns* is ``(start, stop, step)``; only *start*
    and *step* are used to reconstruct the integer sequence whose length
    matches the DataFrame.
    """
    if not range_columns:
        return df
    for rc_name, rc_info in range_columns.items():
        start = rc_info[0]
        step = rc_info[2]
        length = len(df)
        df = df.assign(**{rc_name: list(range(start, start + length * step, step))})
    return df
