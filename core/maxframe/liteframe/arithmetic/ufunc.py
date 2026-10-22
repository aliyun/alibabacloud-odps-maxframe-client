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

from numbers import Number

import numpy as np

from maxframe.liteframe.arithmetic.binary import (
    add,
    div,
    floordiv,
    lshift,
    mod,
    mul,
    pow,
    rdiv,
    rfloordiv,
    rlshift,
    rmod,
    rpow,
    rrshift,
    rshift,
    rsub,
    sub,
)
from maxframe.liteframe.arithmetic.comparison import eq, ge, gt, le, lt, ne
from maxframe.liteframe.arithmetic.core import (
    _build_binary_projection,
    _build_unary_projection,
)
from maxframe.liteframe.arithmetic.logical import and_, logical_not, or_, xor
from maxframe.liteframe.arithmetic.unary import abs_, neg, pos, round_
from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.expressions import (
    LiteFrameBinUfuncExpr,
    LiteFrameUnaryUfuncExpr,
)

# ---------------------------------------------------------------------------
# Lookup tables — direct Polars methods
# ---------------------------------------------------------------------------

_DIRECT_UNARY_POLARS = {
    "exp": ".exp()",
    "log": ".log()",
    "log10": ".log10()",
    "log1p": ".log1p()",
    "sqrt": ".sqrt()",
    "cbrt": ".cbrt()",
    "sin": ".sin()",
    "cos": ".cos()",
    "tan": ".tan()",
    "arcsin": ".arcsin()",
    "arccos": ".arccos()",
    "arctan": ".arctan()",
    "sinh": ".sinh()",
    "cosh": ".cosh()",
    "tanh": ".tanh()",
    "arcsinh": ".arcsinh()",
    "arccosh": ".arccosh()",
    "arctanh": ".arctanh()",
    "sign": ".sign()",
    "degrees": ".degrees()",
    "radians": ".radians()",
    "isfinite": ".is_finite()",
    "isinf": ".is_infinite()",
    "isnan": ".is_nan()",
}

_DIRECT_BINARY_POLARS = {
    "arctan2": "pl.arctan2({left}, {right})",
}

# Unary ufuncs that fall back to map_batches
_MAP_BATCH_UNARY = frozenset(
    {
        "exp2",
        "log2",
        "expm1",
        "square",
        # floor/ceil route through map_batches(np.<ufunc>) rather than the
        # polars .floor()/.ceil() methods: polars preserves the integer
        # dtype for integer inputs, whereas numpy's floor/ceil promote int
        # to float64. Using map_batches keeps the executor result dtype
        # consistent with the (numpy-probed) advertised dtype.
        "floor",
        "ceil",
        "reciprocal",
        "rint",
        "trunc",
        "signbit",
        "conj",
        "conjugate",
        "spacing",
    }
)

# Binary ufuncs that fall back to map_batches
_MAP_BATCH_BINARY = frozenset(
    {
        "hypot",
        "maximum",
        "fmax",
        "minimum",
        "fmin",
        "logaddexp",
        "logaddexp2",
        "copysign",
        "nextafter",
        "ldexp",
    }
)

# Excluded ufuncs (return multiple arrays)
_EXCLUDED_UFUNCS = frozenset({"modf", "frexp"})

# Already-handled unary ufuncs with dedicated expression classes
_EXISTING_UNARY_HANDLERS = {
    "absolute": abs_,
    "fabs": abs_,
    "negative": neg,
    "positive": pos,
    "round": round_,
    "around": round_,
    "logical_not": logical_not,
}

# Already-handled binary ufuncs with dedicated expression classes
_EXISTING_BINARY_HANDLERS = {
    "add": add,
    "subtract": sub,
    "multiply": mul,
    "left_shift": lshift,
    "right_shift": rshift,
    "divide": div,
    "true_divide": div,
    "power": pow,
    "floor_divide": floordiv,
    "mod": mod,
    "remainder": mod,
    "equal": eq,
    "not_equal": ne,
    "less": lt,
    "less_equal": le,
    "greater": gt,
    "greater_equal": ge,
    "logical_and": and_,
    "logical_or": or_,
    "logical_xor": xor,
}

# Binary ufuncs that support reverse operations
_REVERSE_BINARY = {
    "subtract": rsub,
    "divide": rdiv,
    "true_divide": rdiv,
    "floor_divide": rfloordiv,
    "mod": rmod,
    "remainder": rmod,
    # Non-commutative comparisons: a scalar on the left maps to the opposite
    # comparison so that np.less(2, lf) produces (lf > 2), not (lf < 2).
    "less": gt,
    "greater": lt,
    "less_equal": ge,
    "greater_equal": le,
    # Power: a scalar on the left uses the reverse power handler so that
    # np.power(2, lf) produces (2 ** lf), not (lf ** 2).
    "power": rpow,
    # Bit shifts are non-commutative: a scalar on the left maps to the
    # reverse shift handler so np.left_shift(2, lf) produces (2 << lf),
    # not (lf << 2).
    "left_shift": rlshift,
    "right_shift": rrshift,
}


def _find_lf_input(inputs):
    """Find the first LiteFrame in inputs."""
    for x in inputs:
        if isinstance(x, LITEFRAME_TYPE):
            return x
    return None


def _find_other_operand(lf_input, inputs):
    """Return the non-LiteFrame operand by argument position.

    Unlike an identity check, this preserves the operand when a LiteFrame is
    used as both inputs (e.g. ``np.add(lf, lf)``).
    """
    if inputs[0] is lf_input:
        return inputs[1] if len(inputs) > 1 else None
    return inputs[0]


def __array_ufunc__(lf, ufunc, method, *inputs, **kwargs):
    """NumPy ufunc dispatch for LiteFrame.

    Supports all numpy unary and binary ufuncs. Ufuncs with dedicated
    expression classes (absolute, negative, add, etc.) dispatch to the
    existing handlers. Remaining ufuncs use generic
    :class:`LiteFrameUnaryUfuncExpr` / :class:`LiteFrameBinUfuncExpr`.
    """
    if method != "__call__":
        return NotImplemented

    # Reject ufuncs that return multiple outputs
    if ufunc.__name__ in _EXCLUDED_UFUNCS:
        return NotImplemented

    # Reject ufuncs with signatures (gufuncs)
    if getattr(ufunc, "signature", None) is not None:
        return NotImplemented

    for x in inputs:
        if not isinstance(x, (Number,)) and not isinstance(x, LITEFRAME_TYPE):
            try:
                np.asarray(x)
            except Exception:
                return NotImplemented

    ufunc_name = ufunc.__name__

    # --- Existing unary handlers ---
    if ufunc_name in _EXISTING_UNARY_HANDLERS:
        lf_input = _find_lf_input(inputs)
        if lf_input is None:
            return NotImplemented
        handler = _EXISTING_UNARY_HANDLERS[ufunc_name]
        return handler(lf_input, **kwargs)

    # --- Existing binary handlers ---
    if ufunc_name in _EXISTING_BINARY_HANDLERS:
        lf_input = _find_lf_input(inputs)
        if lf_input is None:
            return NotImplemented

        other = _find_other_operand(lf_input, inputs)

        # Handle reverse operations (e.g. np.subtract(1, lf))
        if (
            other is not None
            and not isinstance(other, LITEFRAME_TYPE)
            and ufunc_name in _REVERSE_BINARY
            and inputs[0] is not lf_input
        ):
            return _REVERSE_BINARY[ufunc_name](lf_input, other)

        handler = _EXISTING_BINARY_HANDLERS[ufunc_name]
        return handler(lf_input, other)

    # --- Generic unary ufuncs ---
    if ufunc_name in _DIRECT_UNARY_POLARS or ufunc_name in _MAP_BATCH_UNARY:
        lf_input = _find_lf_input(inputs)
        if lf_input is None:
            return NotImplemented
        return _build_unary_projection(
            lf_input, LiteFrameUnaryUfuncExpr, ufunc_name=ufunc_name
        )

    # --- Generic binary ufuncs ---
    if ufunc_name in _DIRECT_BINARY_POLARS or ufunc_name in _MAP_BATCH_BINARY:
        lf_input = _find_lf_input(inputs)
        if lf_input is None:
            return NotImplemented

        other = _find_other_operand(lf_input, inputs)

        # When the LiteFrame is the second argument (e.g. np.arctan2(1.0, lf)),
        # build the expression with the literal on the left so non-commutative
        # ufuncs preserve operand order. This position-aware construction
        # extends naturally to LiteFrame/LiteFrame operands.
        reverse = (
            other is not None
            and not isinstance(other, LITEFRAME_TYPE)
            and inputs[0] is not lf_input
        )

        return _build_binary_projection(
            lf_input,
            LiteFrameBinUfuncExpr,
            other,
            reverse=reverse,
            ufunc_name=ufunc_name,
        )

    return NotImplemented
