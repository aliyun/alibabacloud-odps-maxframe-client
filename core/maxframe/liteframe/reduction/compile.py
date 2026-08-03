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
import logging
from collections import OrderedDict
from typing import Any, NamedTuple, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

from maxframe.lib.dtypes_extension import ArrowDtype
from maxframe.liteframe.datatypes import (
    DECIMAL_WIDENING_FUNCS,
    infer_decimal_agg_dtype,
    is_decimal_dtype,
)
from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameLiteral,
    LiteFramePow,
    NamedExpr,
)
from maxframe.utils import wrap_arrow_dtype

logger = logging.getLogger(__name__)


class AggCall(str):
    """String subclass carrying per-call kwargs for an aggregation function.

    Behaves as a plain string (the function name) for all code paths that
    only need the name, but exposes ``.kwds`` so the compile step can inject
    function-specific arguments (e.g. ``dropna`` for ``nunique``) into the
    corresponding ``agg_funcs`` descriptor.
    """

    __slots__ = ("kwds",)

    def __new__(cls, name, **kwds):
        obj = super().__new__(cls, name)
        obj.kwds = kwds
        return obj


# For count: map phase uses "count", combine/agg phase uses "sum"
_COMBINE_FUNC_MAP = {
    "count": "sum",
    "size": "sum",
    "sum_sq": "sum",
    "sum_cube": "sum",
    "sum_fourth": "sum",
}

# Functions that can be directly re-aggregated across chunks:
# combine/agg stages use the same (or mapped via _COMBINE_FUNC_MAP) function.
_DIRECTLY_DECOMPOSABLE_FUNCS = frozenset(
    {"sum", "min", "max", "count", "size", "prod", "all", "any"}
)

# Decomposition recipes: derived function -> list of required primitives.
# Each primitive is emitted at the map stage and re-summed at combine/agg.
# A post-aggregation LiteFrameProjection reconstructs the final value.
_DECOMPOSITION_RECIPES = {
    "mean": ["sum", "count"],
    "var": ["sum", "sum_sq", "count"],
    "std": ["sum", "sum_sq", "count"],
    "sem": ["sum", "sum_sq", "count"],
    "skew": ["sum", "sum_sq", "sum_cube", "count"],
    "kurtosis": ["sum", "sum_sq", "sum_cube", "sum_fourth", "count"],
    "kurt": ["sum", "sum_sq", "sum_cube", "sum_fourth", "count"],
}

# Functions that CANNOT be pre-aggregated across chunks.
# Tree reduction cannot be used for these; the map/combine stages must
# pass raw data through and only the final agg stage computes the result.
_NON_DECOMPOSABLE_FUNCS = frozenset({"median", "nunique"})


class CompileAggResult(NamedTuple):
    """Result of compile_agg: holds all compiled aggregation metadata."""

    agg_funcs: list
    agg_output_columns: list
    output_column_names: list
    agg_output_dtypes: Any
    out_dtypes: Any
    out_shape: Tuple[int, int]
    projection_exprs: list
    pre_agg_projection_exprs: list


# Map power-sum primitive names to their exponents for pre-agg projection.
_POWER_SUM_EXPONENTS = {
    "sum_sq": 2,
    "sum_cube": 3,
    "sum_fourth": 4,
}


def _is_numeric_dtype(dt):
    """Check if a dtype is numeric, handling ArrowDtype from both pandas and maxframe."""
    if isinstance(dt, ArrowDtype):
        pa_type = dt.pyarrow_dtype
        return (
            pa.types.is_integer(pa_type)
            or pa.types.is_floating(pa_type)
            or pa.types.is_decimal(pa_type)
        )
    try:
        return np.issubdtype(dt, np.number)
    except TypeError:
        return False


def _normalize_func(raw_func):
    """Normalize raw_func into a standard form.

    Returns (func_dict, func_rename) where:
    - func_dict: OrderedDict mapping column_name -> list of func names
      If column_name is None, the func applies to all columns.
    - func_rename: list of output column names (for named agg), or None
    """
    if raw_func is None:
        # raw_func may be None when only named agg kwargs are provided
        func_dict = OrderedDict()
        func_rename = None
        return func_dict, func_rename

    func_dict = OrderedDict()
    func_rename = None

    if isinstance(raw_func, str):
        # e.g. "sum" -> apply to all columns
        func_dict[None] = [raw_func]
    elif isinstance(raw_func, list):
        # e.g. ["sum", "mean"] -> apply all to all columns
        func_dict[None] = list(raw_func)
    elif isinstance(raw_func, dict):
        # e.g. {"A": "sum"} or {"A": ["sum", "mean"]}
        for col, funcs in raw_func.items():
            if isinstance(funcs, (list, tuple)):
                func_dict[col] = list(funcs)
            else:
                func_dict[col] = [funcs]
    elif callable(raw_func):
        func_dict[None] = [raw_func]
    else:
        raise TypeError(f"Unsupported aggregation function type: {type(raw_func)}")

    return func_dict, func_rename


def _normalize_func_kw(raw_func_kw):
    """Normalize named aggregation kwargs like total=("A", "sum").

    Returns (func_dict, func_rename).
    """
    if not raw_func_kw:
        return None, None

    func_dict = OrderedDict()
    func_rename = []
    for name, spec in raw_func_kw.items():
        if isinstance(spec, tuple) and len(spec) == 2:
            col, func = spec
            func_dict.setdefault(col, []).append(func)
            func_rename.append(name)
        else:
            raise ValueError(
                f"Named aggregation requires (column, func) tuple, got {spec}"
            )
    return func_dict, func_rename


def _infer_output_column_names(func_dict, func_rename, columns, groupby_params):
    """Compute output column names following LiteFrame naming rules.

    Rules:
    - Single func per column: keep original column name
    - Multiple funcs per column: use {col}_{func} suffix
    - Named agg: use provided name
    - With groupby: prepend group key columns

    Column ordering follows pandas convention: column-first then func-first,
    e.g. ["a_sum", "a_mean", "b_sum", "b_mean"] rather than
    ["a_sum", "b_sum", "a_mean", "b_mean"].
    """
    group_key_cols = []
    if groupby_params:
        by = groupby_params["by"]
        if isinstance(by, str):
            group_key_cols = [by]
        else:
            group_key_cols = list(by)

    # Check if any column has multiple funcs
    has_multiple = any(len(funcs) > 1 for funcs in func_dict.values())

    output_cols = list(group_key_cols)

    # When col is None (apply to all columns), we iterate column-first then
    # func-first to match pandas convention. For specific columns, we keep
    # the natural func-first order since there's only one target column.
    # Track named agg names if available
    rename_idx = 0

    for col, funcs in func_dict.items():
        target_cols = [col] if col is not None else list(columns)

        if col is None and len(target_cols) > 1:
            # Column-first ordering for multi-column, multi-func case
            for tc in target_cols:
                for func in funcs:
                    if func_rename and rename_idx < len(func_rename):
                        output_cols.append(func_rename[rename_idx])
                        rename_idx += 1
                    elif isinstance(func, str):
                        if has_multiple or len(funcs) > 1:
                            output_cols.append(f"{tc}_{func}")
                        else:
                            output_cols.append(tc)
                    else:
                        # Callable -- single column name
                        output_cols.append(tc)
        else:
            # Single target column or single func: natural func-first order
            for func in funcs:
                if func_rename and rename_idx < len(func_rename):
                    # Named aggregation -- use the provided name
                    output_cols.append(func_rename[rename_idx])
                    rename_idx += 1
                elif isinstance(func, str):
                    for tc in target_cols:
                        if has_multiple or len(funcs) > 1:
                            output_cols.append(f"{tc}_{func}")
                        else:
                            output_cols.append(tc)
                else:
                    # Callable -- single column name
                    for tc in target_cols:
                        output_cols.append(tc)

    return output_cols


def _build_agg_funcs(func_dict, columns, groupby_params):
    """Build the agg_funcs descriptor list with intermediate deduplication.

    Each entry is a dict with:
    - raw_func_name: original function name (or None for callables)
    - map_func_name: function name to use in map stage
    - agg_func_name: function name to use in combine/agg stages
    - cols: list of column names this agg applies to
    - output_key: unique key for this agg output
    - kwds: dict of keyword args
    - passthrough: bool (True for non-decomposable functions)
    - pre_agg_col: str (optional, present for power-sum primitives like
      sum_sq/sum_cube/sum_fourth) — names the pre-projected column that
      the executor should sum instead of looking up map_func_name

    Uses intermediate deduplication: same (primitive, column) pair emits only
    one descriptor, shared across functions that need it.

    Returns (agg_funcs, projection_needed, registry) where:
    - agg_funcs: list of descriptors
    - projection_needed: bool, True if any recipe-based function is present
    - registry: dict mapping (primitive, col) -> output_key
    """
    agg_funcs = []
    registry = {}  # (primitive, col) -> output_key
    key_counter = [0]  # mutable counter for nested helper
    projection_needed = False
    direct_requests = {}  # (func_name, col) -> count

    def _register_primitive(primitive, col):
        """Register a (primitive, col) pair, deduplicating across calls."""
        reg_key = (primitive, col)
        if reg_key in registry:
            return
        agg_func = _COMBINE_FUNC_MAP.get(primitive, primitive)
        output_key = f"agg_{key_counter[0]}"
        desc = {
            "raw_func_name": primitive,
            "map_func_name": primitive,
            "agg_func_name": agg_func,
            "cols": [col],
            "output_key": output_key,
            "kwds": {},
        }
        # Power-sum primitives get a pre_agg_col pointing to a projected column
        if primitive in _POWER_SUM_EXPONENTS:
            desc["pre_agg_col"] = f"{output_key}__{col}_{primitive}_input"
        agg_funcs.append(desc)
        registry[reg_key] = output_key
        key_counter[0] += 1

    for col, funcs in func_dict.items():
        target_cols = list(columns) if col is None else [col]

        for func in funcs:
            func_name = _resolve_func_name(func)

            if func_name in _DECOMPOSITION_RECIPES:
                projection_needed = True
                for primitive in _DECOMPOSITION_RECIPES[func_name]:
                    for tc in target_cols:
                        _register_primitive(primitive, tc)

            elif func_name in _DIRECTLY_DECOMPOSABLE_FUNCS:
                for tc in target_cols:
                    direct_requests[(func_name, tc)] = (
                        direct_requests.get((func_name, tc), 0) + 1
                    )
                    _register_primitive(func_name, tc)

            elif func_name in _NON_DECOMPOSABLE_FUNCS or callable(func):
                output_key = f"agg_{key_counter[0]}"
                kwds = getattr(func, "kwds", {})
                agg_funcs.append(
                    {
                        "raw_func_name": func_name,
                        "map_func_name": func_name,
                        "agg_func_name": func_name,
                        "cols": target_cols,
                        "output_key": output_key,
                        "kwds": dict(kwds),
                        "passthrough": True,
                    }
                )
                key_counter[0] += 1

            else:
                raise ValueError(f"Unknown aggregation function: {func_name}")

    if any(count > 1 for count in direct_requests.values()):
        projection_needed = True

    return agg_funcs, projection_needed, registry


def _resolve_func_name(func):
    """Resolve a function spec to its string name."""
    if isinstance(func, str):
        return func
    if callable(func):
        return getattr(func, "__name__", repr(func))
    raise TypeError(f"Unsupported function type: {type(func)}")


# Known output dtypes for functions that don't depend on input dtype.
_FIXED_DTYPE_FUNCS = {
    "size": wrap_arrow_dtype(pa.int64()),
    "count": wrap_arrow_dtype(pa.int64()),
}

# Map Python scalar types to arrow dtypes for scalar result inference.
_PYTHON_TYPE_TO_ARROW = {
    int: pa.int64(),
    float: pa.float64(),
    bool: pa.bool_(),
    str: pa.string(),
}


@functools.lru_cache(maxsize=256)
def _infer_func_dtype(func_name, input_dtype):
    """Infer the output dtype for func_name applied to input_dtype.

    Globally cached (LRU, max 256 entries) so repeated calls with the
    same (func_name, input_dtype) pair skip the mock aggregation.
    """
    if func_name in _FIXED_DTYPE_FUNCS:
        return _FIXED_DTYPE_FUNCS[func_name]

    # Run a mock aggregation on a single-column empty Series
    try:
        s = pd.Series(dtype=input_dtype)
        result = getattr(s, func_name)()
    except Exception:
        logger.warning(
            "Mock aggregation %r on dtype %r failed, using fallback dtype",
            func_name,
            input_dtype,
        )
        # Fallback: float64 for numeric, preserve otherwise
        if _is_numeric_dtype(input_dtype):
            return wrap_arrow_dtype(pa.float64())
        return input_dtype

    if isinstance(result, pd.Series):
        return result.dtype
    # Scalar result
    pa_type = _PYTHON_TYPE_TO_ARROW.get(type(result), pa.float64())
    return wrap_arrow_dtype(pa_type)


def _mock_agg_to_infer_dtypes(func_dict, input_dtypes, groupby_params):
    """Infer output dtypes for each (func, column) pair.

    Returns (agg_output_columns, agg_output_dtypes) where:
    - agg_output_columns: list of column names from raw agg output
    - agg_output_dtypes: pd.Series of dtypes for each agg output column

    Dtype inference is globally cached via @lru_cache on _infer_func_dtype,
    so repeated (func_name, input_dtype) pairs across calls are free.
    """
    agg_result_dtypes = OrderedDict()

    for col, funcs in func_dict.items():
        target_cols = list(input_dtypes.index) if col is None else [col]
        pairs = _iter_col_func_pairs(col, funcs, target_cols)

        for col_key, func, tc in pairs:
            result_key = (col_key, func, tc)
            if result_key in agg_result_dtypes:
                continue
            dt = _infer_dtype_for_pair(func, tc, input_dtypes)
            agg_result_dtypes[result_key] = dt

    # Build output columns list
    agg_output_columns = []
    agg_output_dtypes_map = {}
    for (_col_key, _func, c), dt in agg_result_dtypes.items():
        name = c if c is not None else f"__agg_{len(agg_output_columns)}"
        agg_output_columns.append(name)
        agg_output_dtypes_map[name] = dt

    return agg_output_columns, pd.Series(agg_output_dtypes_map)


def _iter_col_func_pairs(col, funcs, target_cols):
    """Yield (col_key, func, target_col) in the correct iteration order.

    Column-first when col is None with multiple target columns,
    func-first otherwise.
    """
    col_key = col or None
    if col is None and len(target_cols) > 1:
        for tc in target_cols:
            for func in funcs:
                yield (col_key, func, tc)
    else:
        for func in funcs:
            for tc in target_cols:
                yield (col_key, func, tc)


def _infer_dtype_for_pair(func, target_col, input_dtypes):
    """Infer output dtype for a single (func, target_col) pair."""
    if not isinstance(func, str):
        return wrap_arrow_dtype(pa.float64())

    if func == "size":
        return _FIXED_DTYPE_FUNCS["size"]

    col_dtype = input_dtypes.get(target_col, wrap_arrow_dtype(pa.float64()))
    return _infer_func_dtype(func, col_dtype)


def _widen_decimal_agg_dtypes(
    func_dict, agg_output_columns, agg_output_dtypes, input_dtypes
):
    """Widen decimal precision in agg_output_dtypes for functions that need it.

    Uses Hive rules via infer_decimal_agg_dtype. This ensures ODPS tables
    are created with sufficient precision to hold aggregation results
    without overflow or truncation.

    The mapping from output column name to function name is built by
    replaying the *exact* same iteration order as
    ``_mock_agg_to_infer_dtypes``, including column-first vs func-first
    ordering, so that each agg output column is associated with the
    correct widening function even when a column appears in multiple
    aggregation functions.
    """
    # Build a mapping: output column name -> func_name
    # by replaying the same iteration order as _mock_agg_to_infer_dtypes
    columns = (
        list(input_dtypes.index)
        if isinstance(input_dtypes, pd.Series)
        else list(input_dtypes.keys())
    )

    # Replay iteration to produce an ordered list of (column_name, func)
    # that is aligned 1:1 with agg_output_columns
    col_func_pairs = []
    for col_key, funcs in func_dict.items():
        target_cols = columns if col_key is None else [col_key]

        if col_key is None and len(target_cols) > 1:
            # Column-first ordering (matches _mock_agg_to_infer_dtypes)
            for tc in target_cols:
                for func in funcs:
                    col_func_pairs.append((tc, func))
        else:
            # Func-first ordering
            for func in funcs:
                for tc in target_cols:
                    col_func_pairs.append((tc, func))

    # Build map: agg_output_column_name -> func_name
    col_func_map = {}
    pair_idx = 0
    for col_name in agg_output_columns:
        if pair_idx < len(col_func_pairs):
            _, func = col_func_pairs[pair_idx]
            col_func_map[col_name] = func
            pair_idx += 1

    widened = agg_output_dtypes.copy()
    for col_name in agg_output_columns:
        dt = widened.get(col_name)
        if dt is None:
            continue
        if not is_decimal_dtype(dt):
            continue
        func_name = col_func_map.get(col_name)
        if func_name is None:
            continue
        if func_name not in DECIMAL_WIDENING_FUNCS:
            continue
        new_dt = infer_decimal_agg_dtype(dt, func_name)
        if new_dt is not dt:
            widened[col_name] = new_dt

    return widened


def _build_projection_exprs(
    func_dict,
    columns,
    groupby_params,
    registry,
    agg_funcs,
    input_dtypes,
    output_column_names,
):
    """Build LiteFrameExpr projection expressions from decomposition results.

    When projection is needed (any recipe-based function), ALL outputs go
    through projection. Direct decomposable results become simple rename
    expressions.

    ``output_column_names`` is the pre-computed list of final output column
    names (including any named-agg renames).  This function consumes them
    in the same iteration order as ``_infer_output_column_names`` so that
    each projection expression gets the correct output name.

    Returns list of NamedExpr for each output column, or empty list if no
    projection is needed.
    """
    group_key_cols = []
    if groupby_params:
        by = groupby_params["by"]
        group_key_cols = [by] if isinstance(by, str) else list(by)

    projection_exprs = []

    # Pass through group keys
    for gk in group_key_cols:
        gk_dtype = input_dtypes.get(gk) if hasattr(input_dtypes, "get") else None
        projection_exprs.append(LiteFrameColumn(name=gk, dtype=gk_dtype).rename(gk))

    # Build lookup for passthrough (func_name, col) -> output_key
    passthrough_lookup = {}
    for desc in agg_funcs:
        if desc.get("passthrough"):
            for c in desc["cols"]:
                passthrough_lookup[(desc["raw_func_name"], c)] = desc["output_key"]

    def _inter_name(primitive, col):
        """Get the intermediate column name for a (primitive, col) pair."""
        output_key = registry[(primitive, col)]
        return f"{output_key}__{col}_{primitive}"

    def _ref(primitive, col):
        """Create a LiteFrameColumn referencing an intermediate."""
        return LiteFrameColumn(name=_inter_name(primitive, col), dtype=None)

    def _recipe_expr(func_name, col):
        """Build arithmetic expression for a recipe-based function."""
        s = _ref("sum", col)
        c = _ref("count", col)
        one = LiteFrameLiteral(value=1, dtype=None)
        two = LiteFrameLiteral(value=2, dtype=None)
        three = LiteFrameLiteral(value=3, dtype=None)

        if func_name == "mean":
            return s / c
        elif func_name in ("var", "std", "sem"):
            sq = _ref("sum_sq", col)
            # variance = (sum_sq - sum^2/count) / (count - 1)
            variance = (sq - s**two / c) / (c - one)
            if func_name == "var":
                return variance
            elif func_name == "std":
                return variance.sqrt()
            else:  # sem
                return (variance / c).sqrt()
        elif func_name == "skew":
            sq = _ref("sum_sq", col)
            cu = _ref("sum_cube", col)
            n = c
            mean = s / n
            m2 = sq / n - mean**two
            m3 = cu / n - three * mean * sq / n + two * mean**three
            skew_pop = m3 / (m2 * m2.sqrt())
            adj = (n * (n - one)).sqrt() / (n - two)
            return skew_pop * adj
        elif func_name in ("kurtosis", "kurt"):
            sq = _ref("sum_sq", col)
            cu = _ref("sum_cube", col)
            sm4 = _ref("sum_fourth", col)
            four = LiteFrameLiteral(value=4, dtype=None)
            six = LiteFrameLiteral(value=6, dtype=None)
            n = c
            mean = s / n
            m2 = sq / n - mean**two
            m4 = (
                sm4 / n
                - four * mean * cu / n
                + six * mean**two * sq / n
                - three * mean**four
            )
            kurt_pop = m4 / (m2**two) - three
            adj_num = (n - one) * ((n + one) * kurt_pop + six)
            adj_den = (n - two) * (n - three)
            return adj_num / adj_den
        else:
            raise ValueError(f"No recipe expression for: {func_name}")

    # Iterate in same order as _infer_output_column_names, consuming
    # output_column_names by index so named-agg renames are respected.
    out_name_idx = len(group_key_cols)

    for col_key, funcs in func_dict.items():
        target_cols = list(columns) if col_key is None else [col_key]

        items = []
        if col_key is None and len(target_cols) > 1:
            for tc in target_cols:
                for func in funcs:
                    items.append((tc, func))
        else:
            for func in funcs:
                for tc in target_cols:
                    items.append((tc, func))

        for tc, func in items:
            if isinstance(func, str):
                func_name = func
            elif callable(func):
                func_name = getattr(func, "__name__", repr(func))
            else:
                func_name = str(func)

            # Use the pre-computed output column name (handles named agg,
            # multi-func suffixes, etc.) instead of recomputing here.
            out_name = output_column_names[out_name_idx]
            out_name_idx += 1

            if func_name in _DECOMPOSITION_RECIPES:
                expr = _recipe_expr(func_name, tc)
                projection_exprs.append(expr.rename(out_name))
            elif func_name in _DIRECTLY_DECOMPOSABLE_FUNCS:
                inter_name = _inter_name(func_name, tc)
                projection_exprs.append(
                    LiteFrameColumn(name=inter_name, dtype=None).rename(out_name)
                )
            elif func_name in _NON_DECOMPOSABLE_FUNCS or callable(func):
                # Passthrough: include in projection as a column reference
                # so it is not silently dropped when recipe-based functions
                # are present.
                output_key = passthrough_lookup[(func_name, tc)]
                inter_name = f"{output_key}__{tc}_{func_name}"
                projection_exprs.append(
                    LiteFrameColumn(name=inter_name, dtype=None).rename(out_name)
                )

    return projection_exprs


def _build_pre_agg_projection_exprs(agg_funcs, input_dtypes):
    """Build LiteFrameProjection expressions for pre-aggregation power transformations.

    For each power-sum primitive (sum_sq, sum_cube, sum_fourth), creates a
    NamedExpr that applies LiteFramePow(col, exponent) with float64 dtype
    to avoid integer overflow.

    Returns a list of NamedExpr objects, or empty list if no power-sum
    primitives are present.
    """
    float64_dtype = wrap_arrow_dtype(pa.float64())
    pre_agg_exprs = []

    for desc in agg_funcs:
        primitive = desc["raw_func_name"]
        if primitive not in _POWER_SUM_EXPONENTS:
            continue
        col = desc["cols"][0]
        exponent = _POWER_SUM_EXPONENTS[primitive]
        pre_agg_col = desc["pre_agg_col"]

        col_expr = LiteFrameColumn(name=col, dtype=float64_dtype)
        lit_expr = LiteFrameLiteral(value=exponent, dtype=float64_dtype)
        pow_expr = LiteFramePow(left=col_expr, right=lit_expr, dtype=float64_dtype)
        named = NamedExpr(name=pre_agg_col, expr=pow_expr, dtype=float64_dtype)
        pre_agg_exprs.append(named)

    return pre_agg_exprs


def compile_agg(op, liteframe):
    """Compile aggregation: normalize func, build agg_funcs, compute output info.

    Returns a CompileAggResult namedtuple with fields:
    - agg_funcs: list of agg descriptors
    - agg_output_columns: column names from raw agg output
    - output_column_names: final output column names (with suffix rules)
    - agg_output_dtypes: dtypes for each agg output column
    - out_dtypes: pd.Series of final output dtypes
    - out_shape: tuple (n_rows, n_cols)
    - projection_exprs: list of NamedExpr for post-agg projection (or empty)
    - pre_agg_projection_exprs: list of NamedExpr for pre-agg power transforms (or empty)
    """
    raw_func = op.raw_func
    raw_func_kw = op.raw_func_kw
    groupby_params = op.groupby_params
    input_dtypes = liteframe.dtypes
    all_input_dtypes = input_dtypes  # preserve unfiltered dtypes for group key lookup
    numeric_only = op.numeric_only

    # Compute group key columns (before filtering)
    group_key_cols = []
    if groupby_params:
        by = groupby_params["by"]
        group_key_cols = [by] if isinstance(by, str) else list(by)

    # Filter columns if numeric_only
    if numeric_only:
        input_cols = [
            c for c in input_dtypes.index if _is_numeric_dtype(input_dtypes[c])
        ]
        input_dtypes = input_dtypes[input_cols]
    else:
        input_cols = list(input_dtypes.index)

    # For groupby: exclude group key columns from the agg target columns
    # so that the aggregation is only applied to non-key columns.
    if group_key_cols:
        agg_cols = [c for c in input_cols if c not in group_key_cols]
    else:
        agg_cols = input_cols

    # Normalize function spec
    func_dict, _ = _normalize_func(raw_func)

    # Handle named agg kwargs separately to avoid rename misalignment.
    # kw renames are appended to output_column_names after the non-kw
    # entries, rather than merged into a shared rename list that would
    # be consumed by the wrong func_dict entries during iteration.
    #
    # Mixing positional func and named-agg kwargs that target the same
    # column is not supported. When func_dict has a None key (apply to
    # all columns), any kw_dict entry would overlap with it. For explicit
    # column keys, a direct collision would cause output_column_names
    # (computed from func_dict only) to be shorter than the projection
    # iteration (which uses merged_func_dict), leading to index misalign.
    kw_rename = None
    if raw_func_kw:
        kw_dict, kw_rename = _normalize_func_kw(raw_func_kw)
        if kw_dict:
            # Reject overlapping column targets between func_dict and kw_dict.
            # None in func_dict means "all columns", which overlaps any kw entry.
            if None in func_dict:
                raise ValueError(
                    "Cannot combine positional aggregation (applying to all "
                    "columns) with named aggregation kwargs targeting specific "
                    "columns. Pass explicit column->func mappings instead."
                )
            overlap = set(func_dict) & set(kw_dict)
            if overlap:
                raise ValueError(
                    f"Columns {sorted(overlap)} appear in both positional func "
                    f"and named aggregation kwargs. Specify each column only once."
                )
            # Build merged dict for agg_funcs, dtype inference, etc.
            merged_func_dict = OrderedDict(func_dict)
            for k, v in kw_dict.items():
                merged_func_dict[k] = list(v)
        else:
            merged_func_dict = func_dict
    else:
        merged_func_dict = func_dict

    # Build agg_funcs descriptors (needs all functions)
    agg_funcs, projection_needed, registry = _build_agg_funcs(
        merged_func_dict, agg_cols, groupby_params
    )

    # When projection is needed (recipe-based functions present), reject
    # mixed positional func + named-agg kwargs. The projection iterates
    # func_dict only, so kw entries would be silently dropped. Users should
    # use one style or the other until kw entries are included in projection.
    if projection_needed and raw_func_kw and func_dict and kw_rename:
        raise ValueError(
            "Mixing positional aggregation functions with named aggregation "
            "kwargs is not supported when recipe-based functions (mean, var, "
            "std, etc.) are present. Please use either all positional or all "
            "named-agg kwargs."
        )

    # Compute output column names:
    # Non-kw entries use default naming rules (no renames)
    output_column_names = _infer_output_column_names(
        func_dict,
        None,
        agg_cols,
        groupby_params,
    )
    # Kw entries use their explicit rename
    if kw_rename:
        output_column_names.extend(kw_rename)

    # Infer dtypes via mock aggregation
    # Build a dtypes subset for mock: exclude group keys from agg target
    agg_dtypes = (
        input_dtypes[[c for c in input_dtypes.index if c in agg_cols]]
        if group_key_cols
        else input_dtypes
    )
    agg_output_columns, agg_output_dtypes = _mock_agg_to_infer_dtypes(
        merged_func_dict,
        agg_dtypes,
        groupby_params,
    )

    # Widen decimal precision for aggregation results to prevent overflow
    agg_output_dtypes = _widen_decimal_agg_dtypes(
        merged_func_dict, agg_output_columns, agg_output_dtypes, agg_dtypes
    )

    # Build final out_dtypes from output_column_names.
    # Group key columns get their original input dtypes.
    # Non-key columns are mapped 1:1 to agg_output_columns by iteration
    # order (both produced by the same func_dict iteration pattern).
    out_dtypes_dict = OrderedDict()
    group_key_set = set(group_key_cols)

    for col_name in output_column_names:
        if col_name in group_key_set:
            out_dtypes_dict[col_name] = all_input_dtypes.get(
                col_name, wrap_arrow_dtype(pa.string())
            )

    non_key_names = [c for c in output_column_names if c not in group_key_set]
    for i, out_name in enumerate(non_key_names):
        if i < len(agg_output_columns):
            agg_col = agg_output_columns[i]
            if agg_col in agg_output_dtypes.index:
                out_dtypes_dict[out_name] = agg_output_dtypes[agg_col]
            else:
                out_dtypes_dict[out_name] = wrap_arrow_dtype(pa.float64())
        else:
            out_dtypes_dict[out_name] = wrap_arrow_dtype(pa.float64())

    out_dtypes = pd.Series(out_dtypes_dict)

    # Compute output shape
    if groupby_params:
        out_shape = (np.nan, len(out_dtypes))
    else:
        out_shape = (1, len(out_dtypes))

    # Build projection expressions if any recipe-based function is present
    if projection_needed:
        projection_exprs = _build_projection_exprs(
            merged_func_dict,
            agg_cols,
            groupby_params,
            registry,
            agg_funcs,
            input_dtypes,
            output_column_names,
        )
    else:
        projection_exprs = []

    # Build pre-agg projection expressions for power-sum primitives
    pre_agg_projection_exprs = _build_pre_agg_projection_exprs(agg_funcs, input_dtypes)

    return CompileAggResult(
        agg_funcs=agg_funcs,
        agg_output_columns=agg_output_columns,
        output_column_names=output_column_names,
        agg_output_dtypes=agg_output_dtypes,
        out_dtypes=out_dtypes,
        out_shape=out_shape,
        projection_exprs=projection_exprs,
        pre_agg_projection_exprs=pre_agg_projection_exprs,
    )
