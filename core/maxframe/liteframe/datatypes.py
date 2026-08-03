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

import functools
import operator

import numpy as np
import pandas as pd

from maxframe.liteframe.utils import build_empty_series

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

try:
    from maxframe.lib.dtypes_extension import ArrowDtype
except ImportError:  # pragma: no cover
    ArrowDtype = None

try:
    from maxframe.utils import wrap_arrow_dtype
except ImportError:  # pragma: no cover
    wrap_arrow_dtype = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DECIMAL128_PRECISION = 38
MAX_DECIMAL128_SCALE = 18


# ---------------------------------------------------------------------------
# Dtype inference utilities (moved from utils.py)
# ---------------------------------------------------------------------------


def to_arrow_dtypes(dtypes):
    """Convert dtypes to ArrowDtype equivalents where possible."""
    if ArrowDtype is None:
        return dtypes

    new_dtypes = {}
    for col, dtype in dtypes.items():
        try:
            pa_type = pa.from_numpy_dtype(dtype)
            new_dtypes[col] = ArrowDtype(pa_type)
        except Exception:
            new_dtypes[col] = dtype
    return pd.Series(new_dtypes)


def _is_fake_arrow_dtype(dtype):
    """Check whether *dtype* is a FakeArrowDtype that cannot perform arithmetic."""
    return (
        ArrowDtype is not None
        and isinstance(dtype, ArrowDtype)
        and getattr(ArrowDtype, "_is_fake", False)
    )


def _to_numpy_dtype(dtype):
    """Convert a FakeArrowDtype to its numpy equivalent for mock-based inference."""
    try:
        return np.dtype(dtype.pyarrow_dtype.to_pandas_dtype())
    except Exception:
        return dtype


def _infer_dtype_via_numpy(left_dtype, right_dtype, opr):
    """Infer result dtype by converting ArrowDtype to numpy, then wrapping back.

    Used as a fallback when ArrowDtype is fake (doesn't support arithmetic).
    """
    np_left = _to_numpy_dtype(left_dtype)
    np_right = _to_numpy_dtype(right_dtype)
    left = build_empty_series(np_left)
    right = build_empty_series(np_right)
    try:
        result_dtype = opr(left, right).dtype
        if wrap_arrow_dtype is not None:
            try:
                pa_type = pa.from_numpy_dtype(result_dtype)
                return wrap_arrow_dtype(pa_type)
            except Exception:
                pass
        return result_dtype
    except TypeError:
        if any(
            t.dtype == "O" or isinstance(t.dtype, pd.api.extensions.ExtensionDtype)
            for t in (left, right)
        ):
            if wrap_arrow_dtype is not None:
                return wrap_arrow_dtype(pa.string())
            return np.dtype("O")
        raise


def infer_dtype(left_dtype, right_dtype, opr):
    """Infer the result dtype of applying *opr* to two dtypes.

    This uses a mock-based approach: it creates two empty pandas Series
    with the given dtypes, applies the operator, and reads the result dtype.

    Parameters
    ----------
    left_dtype : dtype
        Left operand dtype.
    right_dtype : dtype
        Right operand dtype.
    opr : callable
        A binary operator function (e.g. ``operator.add``).

    Returns
    -------
    dtype
        The inferred result dtype.
    """
    if _is_fake_arrow_dtype(left_dtype) or _is_fake_arrow_dtype(right_dtype):
        return _infer_dtype_via_numpy(left_dtype, right_dtype, opr)

    left = build_empty_series(left_dtype)
    right = build_empty_series(right_dtype)
    try:
        return opr(left, right).dtype
    except TypeError:
        if any(
            t.dtype == "O" or isinstance(t.dtype, pd.api.extensions.ExtensionDtype)
            for t in (left, right)
        ):
            if wrap_arrow_dtype is not None:
                return wrap_arrow_dtype(pa.string())
            return np.dtype("O")
        raise


def infer_dtype_cached(left_dtype, right_dtype, opr):
    """Cached version of :func:`infer_dtype`.

    Falls back to uncached inference when *opr* is not hashable
    (e.g. a lambda).
    """
    try:
        return _infer_dtype_cached_impl(left_dtype, right_dtype, opr)
    except TypeError:
        # opr is unhashable (e.g. lambda); fall back to uncached
        return infer_dtype(left_dtype, right_dtype, opr)


@functools.lru_cache(100)
def _infer_dtype_cached_impl(left_dtype, right_dtype, opr):
    return infer_dtype(left_dtype, right_dtype, opr)


# ---------------------------------------------------------------------------
# Decimal type helpers
# ---------------------------------------------------------------------------


def is_decimal_dtype(dtype):
    """Check if a dtype is a decimal type (ArrowDtype with Decimal128 or
    pandas DecimalDtype).
    """
    if ArrowDtype is not None and isinstance(dtype, ArrowDtype):
        if pa is not None and pa.types.is_decimal(dtype.pyarrow_dtype):
            return True
    if (
        isinstance(dtype, pd.api.extensions.ExtensionDtype)
        and hasattr(dtype, "precision")
        and hasattr(dtype, "scale")
        and isinstance(dtype.precision, int)
        and isinstance(dtype.scale, int)
    ):
        return True
    return False


def _estimate_integer_precision(dtype):
    """For integer types, precision is the number of decimal digits needed."""
    try:
        if hasattr(dtype, "itemsize") and np.issubdtype(dtype, np.integer):
            max_val = np.iinfo(dtype).max
            return min(MAX_DECIMAL128_PRECISION, len(str(max_val)))
    except (ValueError, TypeError):
        pass
    return MAX_DECIMAL128_PRECISION


def get_decimal_precision_scale(dtype):
    """Extract (precision, scale) from a decimal or integer dtype.

    For decimal dtypes, returns the stored precision and scale.
    For integer dtypes, estimates precision from the type width and returns scale=0.
    For other dtypes, raises TypeError.
    """
    # Try ArrowDtype decimal first
    if ArrowDtype is not None and isinstance(dtype, ArrowDtype):
        if pa is not None and pa.types.is_decimal(dtype.pyarrow_dtype):
            return dtype.pyarrow_dtype.precision, dtype.pyarrow_dtype.scale
    # Try other ExtensionDtype with precision/scale (e.g. pd.DecimalDtype)
    if (
        isinstance(dtype, pd.api.extensions.ExtensionDtype)
        and hasattr(dtype, "precision")
        and hasattr(dtype, "scale")
        and isinstance(dtype.precision, int)
        and isinstance(dtype.scale, int)
    ):
        return dtype.precision, dtype.scale
    # Integer types: estimate precision, scale=0
    if isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.integer):
        return _estimate_integer_precision(dtype), 0
    raise TypeError(f"Cannot extract decimal precision/scale from {dtype}")


def _wrap_decimal_dtype(precision, scale):
    """Create an ArrowDtype(decimal128(p, s)).

    When *precision* exceeds MAX_DECIMAL128_PRECISION, the result is
    capped and *scale* is reduced by the same amount so that integer
    digit capacity is preserved rather than silently truncating both
    dimensions independently.
    """
    if ArrowDtype is None or pa is None:
        raise RuntimeError(
            "ArrowDtype or pyarrow not available; cannot create decimal dtype"
        )
    if precision > MAX_DECIMAL128_PRECISION:
        # Reduce scale to keep integer digit count, floor at 0
        scale = max(scale - (precision - MAX_DECIMAL128_PRECISION), 0)
        precision = MAX_DECIMAL128_PRECISION
    scale = min(scale, MAX_DECIMAL128_SCALE)
    return ArrowDtype(pa.decimal128(precision, scale))


# ---------------------------------------------------------------------------
# Arithmetic precision/scale formulas (Hive rules)
# ---------------------------------------------------------------------------


def _calc_add_precision_scale(p1, s1, p2, s2):
    result_scale = max(s1, s2)
    result_precision = max(p1 - s1, p2 - s2) + result_scale + 1
    return result_precision, result_scale


def _calc_mul_precision_scale(p1, s1, p2, s2):
    result_scale = s1 + s2
    result_precision = p1 + p2 + 1
    return result_precision, result_scale


def _calc_div_precision_scale(p1, s1, p2, s2):
    result_scale = max(6, s1 + p2 + 1)
    result_precision = p1 - s1 + s2 + result_scale
    return result_precision, result_scale


_DECIMAL_CALC_MAP = {
    operator.add: _calc_add_precision_scale,
    operator.sub: _calc_add_precision_scale,
    operator.mul: _calc_mul_precision_scale,
    operator.truediv: _calc_div_precision_scale,
}


def infer_decimal_arithmetic_dtype(left_dtype, right_dtype, opr):
    """Infer result dtype for decimal arithmetic using Hive precision rules.

    Returns ArrowDtype(decimal128(p,s)) if at least one operand is decimal
    and the operator is supported. Returns None otherwise.
    """
    if not (is_decimal_dtype(left_dtype) or is_decimal_dtype(right_dtype)):
        return None

    if opr not in _DECIMAL_CALC_MAP:
        return None

    # If one side is not decimal but is integer, we can still compute
    try:
        p1, s1 = get_decimal_precision_scale(left_dtype)
        p2, s2 = get_decimal_precision_scale(right_dtype)
    except TypeError:
        return None

    calc_func = _DECIMAL_CALC_MAP[opr]
    result_precision, result_scale = calc_func(p1, s1, p2, s2)
    return _wrap_decimal_dtype(result_precision, result_scale)


# ---------------------------------------------------------------------------
# Aggregation precision/scale formulas (Hive rules)
# ---------------------------------------------------------------------------


def _calc_sum_precision_scale(p, s):
    result_precision = min(p + 10, MAX_DECIMAL128_PRECISION)
    return result_precision, s


def _calc_mean_precision_scale(p, s):
    result_precision = min(p + 4, MAX_DECIMAL128_PRECISION)
    result_scale = min(s + 4, MAX_DECIMAL128_SCALE)
    return result_precision, result_scale


def _calc_var_std_precision_scale(p, s):
    return MAX_DECIMAL128_PRECISION, min(s * 2, MAX_DECIMAL128_SCALE)


_AGGREGATION_CALC_MAP = {
    "sum": _calc_sum_precision_scale,
    "prod": _calc_sum_precision_scale,
    "mean": _calc_mean_precision_scale,
    "var": _calc_var_std_precision_scale,
    "std": _calc_var_std_precision_scale,
    # min, max preserve input type
}

# Aggregation functions that require decimal precision widening.
DECIMAL_WIDENING_FUNCS = frozenset({"sum", "prod", "mean", "var", "std"})


def infer_decimal_agg_dtype(input_dtype, func_name):
    """Infer result dtype for decimal aggregation using Hive precision rules.

    For min/max, preserves input type. For unknown funcs, returns input_dtype.
    """
    if not is_decimal_dtype(input_dtype):
        return input_dtype

    if func_name in ("min", "max"):
        return input_dtype

    p, s = get_decimal_precision_scale(input_dtype)
    calc_func = _AGGREGATION_CALC_MAP.get(func_name)
    if calc_func is None:
        return input_dtype

    result_precision, result_scale = calc_func(p, s)
    return _wrap_decimal_dtype(result_precision, result_scale)
