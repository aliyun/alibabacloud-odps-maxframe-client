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
    LiteFrameAnd,
    LiteFrameColumn,
    LiteFrameInvert,
    LiteFrameIsNA,
    LiteFrameLiteral,
    LiteFrameMul,
    LiteFramePow,
    LiteFrameWhere,
    NamedExpr,
)
from maxframe.liteframe.utils import is_datetime_dtype, resolve_unique_col_name
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
    "sum_xy": "sum",
    "sum_x_pair": "sum",
    "sum_y_pair": "sum",
    "sum_x_pair_sq": "sum",
    "sum_y_pair_sq": "sum",
    "count_pair": "sum",
}

# Functions that can be directly re-aggregated across chunks:
# combine/agg stages use the same (or mapped via _COMBINE_FUNC_MAP) function.
#
# NOTE: "first" and "last" are position-dependent, unlike the other entries.
# The reduction framework must preserve per-group row order through both
# tree and shuffle (hash-partition) reduction for these to be correct.
_DIRECTLY_DECOMPOSABLE_FUNCS = frozenset(
    {"sum", "min", "max", "count", "size", "prod", "all", "any", "first", "last"}
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
    "kurt": ["sum", "sum_sq", "sum_cube", "sum_fourth", "count"],
}

# Functions that CANNOT be pre-aggregated across chunks.
# Tree reduction cannot be used for these; the map/combine stages must
# pass raw data through and only the final agg stage computes the result.
_NON_DECOMPOSABLE_FUNCS = frozenset({"median", "nunique", "quantile"})
_QUANTILE_PRESERVING_INTERP = frozenset({"lower", "higher", "nearest"})
_SKETCH_MAP_FUNC = "quantile_sketch"

# Functions that support a min_count guard (where-guard on count).
_MIN_COUNT_FUNCS = frozenset({"sum", "prod", "min", "max", "first", "last"})

# min_count=1 is a no-op for these because the count of non-null values
# is 0 if and only if all values are null, in which case the unguarded
# result is already null — independent of skipna.
_MIN_COUNT_ONE_NOOP = frozenset({"min", "max", "first", "last"})


def _reject_unsupported_min_count(func, func_name):
    """Raise ValueError when min_count is explicitly requested on a
    function outside _MIN_COUNT_FUNCS.
    """
    mc = getattr(func, "kwds", {}).get("min_count", 0)
    if mc > 0 and func_name not in _MIN_COUNT_FUNCS:
        raise ValueError(
            f"min_count is not supported for {func_name}(); "
            f"supported functions: {sorted(_MIN_COUNT_FUNCS)}"
        )


def _effective_min_count(func, func_name):
    """Return effective min_count (0 if it doesn't affect the result).

    Negative values are treated as no guard, matching pandas.
    """
    mc = getattr(func, "kwds", {}).get("min_count", 0)
    if mc <= 0:
        return 0
    if func_name in _MIN_COUNT_ONE_NOOP and mc == 1:
        return 0
    return mc


def _skipna_of(func):
    """Return skipna kwarg (default True) for first/last."""
    return getattr(func, "kwds", {}).get("skipna", True)


_SKETCH_AGG_FUNC = "quantile_sketch_merge"


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
    # Custom agg: pre/post projection exprs (empty for non-custom aggs)
    custom_pre_projection: list = []
    custom_post_projection: list = []


# Map power-sum primitive names to their exponents for pre-agg projection.
_POWER_SUM_EXPONENTS = {
    "sum_sq": 2,
    "sum_cube": 3,
    "sum_fourth": 4,
}

# Cross-product primitive: sum of element-wise product of two columns.
# Pre-agg projection uses LiteFrameMul instead of LiteFramePow.
_CROSS_PRODUCT_PRIMITIVE = "sum_xy"

# Pairwise mask primitives: computed over the both-non-null observation set.
# sum_x_pair = sum(col_x * mask), sum_y_pair = sum(col_y * mask),
# count_pair = sum(mask).  Used in cov/corr reconstruction so that
# sum_x, sum_y, and n all use the same pairwise-complete observation set
# as sum_xy (which null-propagates via LiteFrameMul), matching pandas.
_PAIRWISE_MASK_PRIMITIVES = frozenset(
    {
        "sum_x_pair",
        "sum_y_pair",
        "sum_x_pair_sq",
        "sum_y_pair_sq",
        "count_pair",
    }
)


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


def _check_pairwise_in_list(funcs):
    """Raise a targeted error if cov/corr appear in a list/dict agg.

    Pairwise functions produce N×N matrix output and cannot be composed
    with per-column aggregations.  Without this check, they fall through
    to ``_FUNC_TO_POLARS`` and raise a misleading "Unknown aggregation
    function: cov".
    """
    pairwise = [f for f in funcs if f in ("cov", "corr")]
    if pairwise:
        raise ValueError(
            f"{', '.join(repr(f) for f in pairwise)} are pairwise functions "
            f"that produce N×N matrix output; call lf.cov() or lf.corr() "
            f"directly instead of agg()"
        )


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

    # Pairwise functions (cov/corr) are only valid as bare-string
    # calls: lf.cov(), lf.corr().  They cannot be composed in a list
    # or dict agg (lf.agg(['cov','sum'])) because they produce N×N
    # matrix output, not per-column scalars.  Raise a clear error rather
    # than falling through to "Unknown aggregation function: cov".
    if isinstance(raw_func, list):
        _check_pairwise_in_list(raw_func)
    elif isinstance(raw_func, dict):
        for _col, funcs in raw_func.items():
            _check_pairwise_in_list(
                funcs if isinstance(funcs, (list, tuple)) else [funcs]
            )

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


def _get_callable_name(func):
    """Return the name of a callable for column-naming purposes.

    Mirrors ``pandas.core.common.get_callable_name``: prefer ``__name__``,
    fall back to the class name for callable instances, then ``"<custom>"``.
    """
    if hasattr(func, "__name__"):
        return func.__name__
    return type(func).__name__ if callable(func) else "<custom>"


def _resolve_callable_names(funcs):
    """Return unique display names for a list of callable aggs.

    Disambiguates callables that share the same name, mirroring the
    intent of pandas' ``_managle_lambda_list`` but with one uniform
    rule: when the same base name appears more than once, every
    occurrence is numbered with a zero-based index — inside the angle
    brackets for angle-bracket names (``<lambda_0>``, ``<lambda_1>``,
    ``<custom_0>``, …) or as an ``_N`` suffix for plain names
    (``my_func_0``, ``my_func_1``). A single callable keeps its
    original name unchanged.
    """
    counts: dict[str, int] = {}
    for f in funcs:
        if callable(f):
            base = _get_callable_name(f)
            counts[base] = counts.get(base, 0) + 1
    idx: dict[str, int] = {}
    names = []
    for f in funcs:
        if not callable(f):
            names.append(None)
            continue
        base = _get_callable_name(f)
        if counts[base] > 1:
            n = idx.get(base, 0)
            idx[base] = n + 1
            if base.startswith("<") and base.endswith(">"):
                names.append(f"<{base[1:-1]}_{n}>")
            else:
                names.append(f"{base}_{n}")
        else:
            names.append(base)
    return names


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

    def _str_func_name(tc, func, funcs_list):
        suffix = getattr(func, "kwds", {}).get("output_suffix")
        if suffix:
            return f"{tc}_{suffix}"
        if has_multiple or len(funcs_list) > 1:
            return f"{tc}_{func}"
        return tc

    output_cols = list(group_key_cols)
    # Track all names in use (group keys + generated output cols) to
    # resolve collisions: a generated name like "a_count" (from col "a"
    # + func "count") can collide with a group key named "a_count", which
    # would cause a polars DuplicateError at the projection/rename step.
    used_names = set(group_key_cols)

    def _emit(name):
        """Append *name* to output_cols, resolving collisions with used names."""
        resolved = resolve_unique_col_name(name, used_names)
        output_cols.append(resolved)
        used_names.add(resolved)

    # When col is None (apply to all columns), we iterate column-first then
    # func-first to match pandas convention. For specific columns, we keep
    # the natural func-first order since there's only one target column.
    # Track named agg names if available
    rename_idx = 0

    for col, funcs in func_dict.items():
        target_cols = [col] if col is not None else list(columns)
        # Pre-resolve unique names for callable funcs so that two
        # callables sharing the same __name__ (e.g. two lambdas, or two
        # callable instances) produce distinct columns. Mirrors pandas'
        # _managle_lambda_list.
        callable_names = _resolve_callable_names(funcs)

        if col is None and len(target_cols) > 1:
            # Column-first ordering for multi-column, multi-func case.
            # size is frame-level (one count per group / whole frame), not
            # per-column; it is emitted once for the single agg descriptor
            # (see _build_agg_funcs) regardless of target column count.
            size_emitted = False
            for tc in target_cols:
                for i, func in enumerate(funcs):
                    if func_rename and rename_idx < len(func_rename):
                        _emit(func_rename[rename_idx])
                        rename_idx += 1
                    elif isinstance(func, str) and func == "size":
                        if not size_emitted:
                            _emit("size")
                            size_emitted = True
                    elif isinstance(func, str):
                        _emit(_str_func_name(tc, func, funcs))
                    else:
                        func_name = callable_names[i]
                        _emit(
                            f"{tc}_{func_name}"
                            if (has_multiple or len(funcs) > 1)
                            else tc
                        )
        else:
            # Single target column or single func: natural func-first order
            for i, func in enumerate(funcs):
                if func_rename and rename_idx < len(func_rename):
                    # Named aggregation -- use the provided name
                    _emit(func_rename[rename_idx])
                    rename_idx += 1
                elif isinstance(func, str) and func == "size":
                    # size is frame-level; pandas names the result "size".
                    _emit("size")
                elif isinstance(func, str):
                    for tc in target_cols:
                        _emit(_str_func_name(tc, func, funcs))
                else:
                    func_name = callable_names[i]
                    for tc in target_cols:
                        _emit(
                            f"{tc}_{func_name}"
                            if (has_multiple or len(funcs) > 1)
                            else tc
                        )

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

    def _register_primitive(primitive, col, kwds=None, reg_key=None):
        """Register a (primitive, col) pair, deduplicating across calls."""
        kwds = kwds or {}
        reg_key = reg_key or (primitive, col)
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
            "kwds": dict(kwds),
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
            _reject_unsupported_min_count(func, func_name)

            if func_name in _DECOMPOSITION_RECIPES:
                projection_needed = True
                for primitive in _DECOMPOSITION_RECIPES[func_name]:
                    for tc in target_cols:
                        _register_primitive(primitive, tc)

            elif func_name in _DIRECTLY_DECOMPOSABLE_FUNCS:
                # min_count guard: register func + count companion, force
                # projection so a where-guard can null insufficient groups.
                min_count = _effective_min_count(func, func_name)
                if min_count > 0:
                    if func_name in _MIN_COUNT_ONE_NOOP and not groupby_params:
                        raise TypeError(
                            f"{func_name}() does not support min_count "
                            f"for whole-frame aggregation. Use "
                            f"groupby().{func_name}(min_count=...) for "
                            f"per-group min_count guard."
                        )
                    for tc in target_cols:
                        if func_name in ("first", "last"):
                            skipna = _skipna_of(func)
                            _register_primitive(
                                func_name,
                                tc,
                                kwds={"skipna": skipna},
                                reg_key=(func_name, tc, skipna),
                            )
                        else:
                            _register_primitive(func_name, tc)
                        _register_primitive("count", tc)
                    projection_needed = True
                elif func_name == "size":
                    # size is frame-level (one count per group / whole frame),
                    # not per-column. Register a single descriptor carrying
                    # all target columns so the executor emits one pl.len()
                    # column, and name the output "size" (see
                    # _infer_output_column_names).
                    if not any(d["raw_func_name"] == "size" for d in agg_funcs):
                        output_key = f"agg_{key_counter[0]}"
                        agg_func = _COMBINE_FUNC_MAP.get(func_name, func_name)
                        agg_funcs.append(
                            {
                                "raw_func_name": func_name,
                                "map_func_name": func_name,
                                "agg_func_name": agg_func,
                                "cols": list(target_cols),
                                "output_key": output_key,
                                "kwds": {},
                            }
                        )
                        registry[(func_name, None)] = output_key
                        key_counter[0] += 1
                elif func_name in ("first", "last"):
                    # Dedup by (func_name, col, skipna) so first(True) and
                    # first(False) on the same column produce separate
                    # descriptors.
                    skipna = _skipna_of(func)
                    for tc in target_cols:
                        rk = (func_name, tc, skipna)
                        direct_requests[rk] = direct_requests.get(rk, 0) + 1
                        _register_primitive(
                            func_name, tc, kwds={"skipna": skipna}, reg_key=rk
                        )
                else:
                    for tc in target_cols:
                        direct_requests[(func_name, tc)] = (
                            direct_requests.get((func_name, tc), 0) + 1
                        )
                        _register_primitive(func_name, tc)

            elif func_name in _NON_DECOMPOSABLE_FUNCS or callable(func):
                output_key = f"agg_{key_counter[0]}"
                kwds = getattr(func, "kwds", {})
                desc = {
                    "raw_func_name": func_name,
                    "map_func_name": func_name,
                    "agg_func_name": func_name,
                    "cols": target_cols,
                    "output_key": output_key,
                    "kwds": dict(kwds),
                    "passthrough": True,
                }
                if func_name == "quantile" and kwds.get("approx"):
                    desc["map_func_name"] = _SKETCH_MAP_FUNC
                    desc["agg_func_name"] = _SKETCH_AGG_FUNC
                    desc["passthrough"] = False
                    desc["sketch"] = True
                agg_funcs.append(desc)
                key_counter[0] += 1

            else:
                raise ValueError(f"Unknown aggregation function: {func_name}")

    if any(count > 1 for count in direct_requests.values()):
        projection_needed = True

    if any(d.get("raw_func_name") == "quantile" for d in agg_funcs):
        projection_needed = True

    # When multiple directly-decomposable funcs are applied to multiple
    # columns (e.g. agg(["first", "last"])), the descriptor order
    # (func-first) differs from the output-column-name order (column-first).
    # The positional rename in _rename_columns would mismatch, so force a
    # projection (which renames by intermediate name, not position).
    if not projection_needed:
        for col, funcs in func_dict.items():
            target_cols = list(columns) if col is None else [col]
            if col is None and len(target_cols) > 1 and len(funcs) > 1:
                # Only needed if all funcs are directly decomposable (no
                # recipes, no passthrough, no quantile — those already
                # trigger projection above).
                if all(
                    _resolve_func_name(f) in _DIRECTLY_DECOMPOSABLE_FUNCS for f in funcs
                ):
                    projection_needed = True
                    break

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

# Functions that preserve the input column's dtype.
_DTYPE_PRESERVING_FUNCS = frozenset({"min", "max", "first", "last"})

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
    if func_name in _DTYPE_PRESERVING_FUNCS:
        return input_dtype

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

    func_name = func if isinstance(func, str) else _get_callable_name(func)

    # first/last select a value, no type conversion — output dtype = input dtype.
    # Short-circuit before _infer_func_dtype, which would call pd.Series.first()
    # (a deprecated time-based method, not an aggregation).
    if func_name in ("first", "last"):
        return input_dtypes.get(target_col, wrap_arrow_dtype(pa.float64()))

    if func_name == "quantile":
        kwds = getattr(func, "kwds", {})
        interp = kwds.get("interpolation", "linear")
        col_dtype = input_dtypes.get(target_col, wrap_arrow_dtype(pa.float64()))
        return _quantile_output_dtype(interp, col_dtype)

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


def _size_intermediate_name(output_key_prefix: str = "") -> str:
    """Executor alias of the frame-level size intermediate column.

    The size descriptor is frame-level (one ``pl.len()`` per group / whole
    frame, see ``_build_agg_funcs``); the DPE/SPE executors alias the single
    emitted column ``{prefix}__size__`` where ``prefix`` is ``{output_key}__``
    when a prefix is used (empty for the single-descriptor case).  Consumers
    of the agg output (post-agg projection references, intermediate dtype
    dicts, the DPE combine stage) must use the same name -- this helper is
    the single owner of the convention.
    """
    return f"{output_key_prefix}__size__"


# --- idxmin/idxmax reserved intermediate column names ---
# These are cross-engine wire constants (DPE executor, SPE adapter,
# custom_agg.py, tests). A typo in any one copy breaks another engine
# silently. Keep them centralized here as the single owner.
_ROW_POS_COLUMN = "__mf_agg_row_pos__"
_VALUE_SUFFIX = "__value"
_IDX_SUFFIX = "__idx"


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

    # Build lookup for passthrough (func_name, col, suffix, kwds_sig)
    # -> output_key.  The kwds signature distinguishes non-decomposable
    # calls with different arguments (e.g. two quantile aggregations on
    # the same column with different q values), so each maps to its own
    # intermediate column.
    def _kwds_sig(kwds):
        """Build a hashable signature from the kwargs that distinguish
        distinct non-decomposable aggregation calls."""
        if not kwds:
            return ()
        # Only include kwargs that affect the result (not output_suffix,
        # which is handled separately).  Stable sorted ordering.
        return tuple(sorted((k, v) for k, v in kwds.items() if k != "output_suffix"))

    passthrough_lookup = {}
    for desc in agg_funcs:
        if desc.get("passthrough") or desc.get("sketch"):
            for c in desc["cols"]:
                kwds = desc.get("kwds", {})
                suffix = kwds.get("output_suffix")
                sig = _kwds_sig(kwds)
                passthrough_lookup[(desc["raw_func_name"], c, suffix, sig)] = desc[
                    "output_key"
                ]

    def _inter_name(primitive, col, skipna=None):
        """Get the intermediate column name for a (primitive, col) pair."""
        if skipna is not None:
            output_key = registry[(primitive, col, skipna)]
        else:
            output_key = registry[(primitive, col)]
        return f"{output_key}__{col}_{primitive}"

    def _ref(primitive, col):
        """Create a LiteFrameColumn referencing an intermediate."""
        return LiteFrameColumn(name=_inter_name(primitive, col), dtype=None)

    def _recipe_expr(func_name, col, ddof=1):
        """Build arithmetic expression for a recipe-based function."""
        s = _ref("sum", col)
        c = _ref("count", col)
        one = LiteFrameLiteral(value=1, dtype=None)
        two = LiteFrameLiteral(value=2, dtype=None)
        three = LiteFrameLiteral(value=3, dtype=None)
        ddof_lit = LiteFrameLiteral(value=ddof, dtype=None)
        _float64 = wrap_arrow_dtype(pa.float64())
        _nan = LiteFrameLiteral(value=float("nan"), dtype=_float64)

        if func_name == "mean":
            return s / c
        elif func_name in ("var", "std", "sem"):
            sq = _ref("sum_sq", col)
            # variance = (sum_sq - sum^2/count) / (count - ddof)
            # Guard n <= ddof: pandas returns NaN, not a negative value.
            raw_var = (sq - s**two / c) / (c - ddof_lit)
            variance = LiteFrameWhere(
                condition=c > ddof_lit,
                true_value=raw_var,
                false_value=_nan,
                dtype=_float64,
            )
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
        elif func_name == "kurt":
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

        # size is frame-level: one "size" output name (see
        # _infer_output_column_names) even though the column-first iteration
        # yields one (tc, "size") item per target column -- emit the single
        # rename expression once and skip the rest.
        size_projection_emitted = False

        for tc, func in items:
            if isinstance(func, str):
                func_name = func
            elif callable(func):
                func_name = getattr(func, "__name__", repr(func))
            else:
                func_name = str(func)
            _reject_unsupported_min_count(func, func_name)

            if func_name == "size":
                if size_projection_emitted:
                    continue
                size_projection_emitted = True
                out_name = output_column_names[out_name_idx]
                out_name_idx += 1
                size_key = registry[("size", None)]
                projection_exprs.append(
                    LiteFrameColumn(
                        name=_size_intermediate_name(f"{size_key}__"), dtype=None
                    ).rename(out_name)
                )
                continue

            # Use the pre-computed output column name (handles named agg,
            # multi-func suffixes, etc.) instead of recomputing here.
            out_name = output_column_names[out_name_idx]
            out_name_idx += 1

            if func_name in _DECOMPOSITION_RECIPES:
                ddof = getattr(func, "kwds", {}).get("ddof", 1)
                expr = _recipe_expr(func_name, tc, ddof=ddof)
                projection_exprs.append(expr.rename(out_name))
            elif func_name in _DIRECTLY_DECOMPOSABLE_FUNCS:
                min_count = _effective_min_count(func, func_name)
                if min_count > 0:
                    # where(count >= min_count, func_result, None).
                    # dtype=None on all branches lets bind() infer the
                    # widest common type — fixes both decimal truncation
                    # and integer widening (int8 sum → int64) when
                    # _widen_decimal_agg_dtypes or _infer_func_dtype
                    # widens the true_value's intermediate column.
                    if func_name in ("first", "last"):
                        true_value = LiteFrameColumn(
                            name=_inter_name(func_name, tc, skipna=_skipna_of(func)),
                            dtype=None,
                        )
                    else:
                        true_value = _ref(func_name, tc)
                    expr = LiteFrameWhere(
                        condition=_ref("count", tc)
                        >= LiteFrameLiteral(value=min_count, dtype=None),
                        true_value=true_value,
                        false_value=LiteFrameLiteral(value=None, dtype=None),
                        dtype=None,
                    )
                    projection_exprs.append(expr.rename(out_name))
                elif func_name in ("first", "last"):
                    inter_name = _inter_name(func_name, tc, skipna=_skipna_of(func))
                    projection_exprs.append(
                        LiteFrameColumn(name=inter_name, dtype=None).rename(out_name)
                    )
                else:
                    inter_name = _inter_name(func_name, tc)
                    projection_exprs.append(
                        LiteFrameColumn(name=inter_name, dtype=None).rename(out_name)
                    )
            elif func_name in _NON_DECOMPOSABLE_FUNCS or callable(func):
                func_kwds = getattr(func, "kwds", {})
                sig = _kwds_sig(func_kwds)
                output_key = passthrough_lookup[
                    (func_name, tc, func_kwds.get("output_suffix"), sig)
                ]
                map_func = (
                    _SKETCH_MAP_FUNC
                    if (func_name == "quantile" and func_kwds.get("approx"))
                    else func_name
                )
                inter_name = f"{output_key}__{tc}_{map_func}"
                projection_exprs.append(
                    LiteFrameColumn(name=inter_name, dtype=None).rename(out_name)
                )

    return projection_exprs


def _build_pre_agg_projection_exprs(agg_funcs, input_dtypes):
    """Build LiteFrameProjection expressions for pre-aggregation transforms.

    For each power-sum primitive (sum_sq, sum_cube, sum_fourth), creates a
    NamedExpr that applies LiteFramePow(col, exponent) with float64 dtype
    to avoid integer overflow.

    For each cross-product primitive (sum_xy), creates a NamedExpr that
    applies LiteFrameMul(col_x, col_y) with float64 dtype.  The column
    pair is read from desc["pair"] (a (col_x, col_y) tuple) — never
    round-tripped through a string encoding.

    For pairwise mask primitives (sum_x_pair, sum_y_pair, count_pair),
    creates a NamedExpr that multiplies the column by a both-non-null
    mask (1.0 where both columns are non-null, 0.0 otherwise), so that
    the sum uses the same pairwise-complete observation set as sum_xy.

    Returns a list of NamedExpr objects, or empty list if no pre-agg
    primitives are present.
    """
    float64_dtype = wrap_arrow_dtype(pa.float64())
    one_lit = LiteFrameLiteral(value=1.0, dtype=float64_dtype)
    zero_lit = LiteFrameLiteral(value=0.0, dtype=float64_dtype)
    pre_agg_exprs = []

    for desc in agg_funcs:
        primitive = desc["raw_func_name"]
        if primitive in _POWER_SUM_EXPONENTS:
            col = desc["cols"][0]
            exponent = _POWER_SUM_EXPONENTS[primitive]
            pre_agg_col = desc["pre_agg_col"]

            col_expr = LiteFrameColumn(name=col, dtype=float64_dtype)
            lit_expr = LiteFrameLiteral(value=exponent, dtype=float64_dtype)
            pow_expr = LiteFramePow(left=col_expr, right=lit_expr, dtype=float64_dtype)
            named = NamedExpr(name=pre_agg_col, expr=pow_expr, dtype=float64_dtype)
            pre_agg_exprs.append(named)

        elif primitive == _CROSS_PRODUCT_PRIMITIVE:
            col_x, col_y = desc["pair"]
            pre_agg_col = desc["pre_agg_col"]

            left = LiteFrameColumn(name=col_x, dtype=float64_dtype)
            right = LiteFrameColumn(name=col_y, dtype=float64_dtype)
            mul_expr = LiteFrameMul(left=left, right=right, dtype=float64_dtype)
            named = NamedExpr(name=pre_agg_col, expr=mul_expr, dtype=float64_dtype)
            pre_agg_exprs.append(named)

        elif primitive in _PAIRWISE_MASK_PRIMITIVES:
            col_x, col_y = desc["pair"]
            pre_agg_col = desc["pre_agg_col"]

            # both-non-null mask: 1.0 where both ci and cj are non-null,
            # 0.0 otherwise.  col * mask null-propagates on col, so
            # sum(col * mask) = sum(col) over both-non-null rows — the
            # same observation set as sum_xy (which also null-propagates).
            col_x_not_null = LiteFrameInvert(
                operand=LiteFrameIsNA(operand=LiteFrameColumn(name=col_x, dtype=None)),
                dtype=wrap_arrow_dtype(pa.bool_()),
            )
            col_y_not_null = LiteFrameInvert(
                operand=LiteFrameIsNA(operand=LiteFrameColumn(name=col_y, dtype=None)),
                dtype=wrap_arrow_dtype(pa.bool_()),
            )
            mask = LiteFrameWhere(
                condition=LiteFrameAnd(
                    left=col_x_not_null,
                    right=col_y_not_null,
                    dtype=wrap_arrow_dtype(pa.bool_()),
                ),
                true_value=one_lit,
                false_value=zero_lit,
                dtype=float64_dtype,
            )
            if primitive == "count_pair":
                expr = mask
            elif primitive in ("sum_x_pair_sq", "sum_y_pair_sq"):
                col = col_y if primitive.endswith("y_pair_sq") else col_x
                col_expr = LiteFrameColumn(name=col, dtype=float64_dtype)
                sq_expr = LiteFramePow(
                    left=col_expr,
                    right=LiteFrameLiteral(value=2, dtype=float64_dtype),
                    dtype=float64_dtype,
                )
                expr = LiteFrameMul(left=sq_expr, right=mask, dtype=float64_dtype)
            else:
                col = col_y if primitive == "sum_y_pair" else col_x
                col_expr = LiteFrameColumn(name=col, dtype=float64_dtype)
                expr = LiteFrameMul(left=col_expr, right=mask, dtype=float64_dtype)
            named = NamedExpr(name=pre_agg_col, expr=expr, dtype=float64_dtype)
            pre_agg_exprs.append(named)

    return pre_agg_exprs


def _quantile_output_dtype(interp, input_dtype):
    """Pandas-rationale dtype: float64 for linear/midpoint, input dtype
    preserved for lower/higher/nearest. Decimal -> float64 always.
    Datetime/date -> input dtype preserved (the executor produces
    timestamps, not floats, for temporal quantile)."""
    if is_decimal_dtype(input_dtype):
        return wrap_arrow_dtype(pa.float64())
    if interp in _QUANTILE_PRESERVING_INTERP or is_datetime_dtype(input_dtype):
        return input_dtype
    return wrap_arrow_dtype(pa.float64())


def _build_pairwise_projection_exprs(
    func_name,
    pairs,
    pair_keys,
    registry,
    groupby_params,
    input_dtypes,
    ddof,
    min_periods,
):
    """Build LiteFrameExpr projection expressions for pairwise cov/corr.

    Each pair (ci, cj) produces one output column.  Diagonal pairs
    (ci == cj) yield var(ci) for cov, or literal 1.0 for corr.

    For off-diagonal pairs, sum_x/sum_y/n are drawn from pairwise mask
    primitives (sum_x_pair, sum_y_pair, count_pair) so that all four
    terms use the same both-non-null observation set as sum_xy —
    matching pandas' pairwise-complete covariance/correlation.
    """
    float64_dtype = wrap_arrow_dtype(pa.float64())
    two = LiteFrameLiteral(value=2, dtype=None)
    ddof_lit = LiteFrameLiteral(value=ddof, dtype=None)
    nan_lit = LiteFrameLiteral(value=float("nan"), dtype=float64_dtype)

    projection_exprs = []

    # Pass through group keys
    if groupby_params:
        by = groupby_params["by"]
        group_keys = [by] if isinstance(by, str) else list(by)
        for gk in group_keys:
            gk_dtype = input_dtypes.get(gk)
            projection_exprs.append(LiteFrameColumn(name=gk, dtype=gk_dtype).rename(gk))

    def _inter_name(primitive, col):
        """Get the intermediate column name for a (primitive, col) pair."""
        output_key = registry[primitive][col]
        return f"{output_key}__{col}_{primitive}"

    def _ref(primitive, col_key):
        """Create a LiteFrameColumn referencing an intermediate."""
        output_key = registry[primitive][col_key]
        return LiteFrameColumn(name=f"{output_key}__{col_key}_{primitive}", dtype=None)

    for ci, cj in pairs:
        out_name = (
            f"{ci}__{cj}__{func_name}"
            if ci == cj
            else f"{pair_keys[(ci, cj)]}__{func_name}"
        )

        if ci == cj:
            # Diagonal: per-column stats are pairwise-consistent
            # (sum_sq and count both skip nulls in the same column).
            s_i = _ref("sum", ci)
            n = _ref("count", ci)
            if func_name == "cov":
                sq = _ref("sum_sq", ci)
                raw_var = (sq - s_i**two / n) / (n - ddof_lit)
                # Guard n <= ddof: pandas returns NaN, not negative.
                expr = LiteFrameWhere(
                    condition=n > ddof_lit,
                    true_value=raw_var,
                    false_value=nan_lit,
                    dtype=float64_dtype,
                )
            else:
                # corr diagonal: var/var.  Yields 1.0 when variance
                # is positive, NaN when variance is zero (constant
                # column) or n <= ddof — matching pandas, which
                # returns NaN for undefined correlation.
                sq = _ref("sum_sq", ci)
                raw_var = (sq - s_i**two / n) / (n - ddof_lit)
                var_i = LiteFrameWhere(
                    condition=n > ddof_lit,
                    true_value=raw_var,
                    false_value=nan_lit,
                    dtype=float64_dtype,
                )
                expr = var_i / var_i
        else:
            # Off-diagonal: use pairwise mask primitives so sum_x,
            # sum_y, and n all use the same both-non-null observation
            # set as sum_xy (which null-propagates via LiteFrameMul).
            pair_key = pair_keys[(ci, cj)]
            s_xy = _ref("sum_xy", pair_key)
            s_i = _ref("sum_x_pair", pair_key)
            s_j = _ref("sum_y_pair", pair_key)
            n = _ref("count_pair", pair_key)
            raw_cov = (s_xy - s_i * s_j / n) / (n - ddof_lit)
            # Guard n <= ddof: pandas returns NaN, not negative.
            cov = LiteFrameWhere(
                condition=n > ddof_lit,
                true_value=raw_cov,
                false_value=nan_lit,
                dtype=float64_dtype,
            )

            if func_name == "cov":
                expr = cov
            else:
                # corr: cov / (std_i * std_j).  The std_i/std_j here
                # use the same pairwise mask, so they reflect the
                # spread of ci over both-non-null rows — matching
                # pandas' pairwise-complete correlation.
                sq_i = _ref("sum_x_pair_sq", pair_key)
                sq_j = _ref("sum_y_pair_sq", pair_key)
                raw_var_i = (sq_i - s_i**two / n) / (n - ddof_lit)
                raw_var_j = (sq_j - s_j**two / n) / (n - ddof_lit)
                var_i = LiteFrameWhere(
                    condition=n > ddof_lit,
                    true_value=raw_var_i,
                    false_value=nan_lit,
                    dtype=float64_dtype,
                )
                var_j = LiteFrameWhere(
                    condition=n > ddof_lit,
                    true_value=raw_var_j,
                    false_value=nan_lit,
                    dtype=float64_dtype,
                )
                expr = cov / (var_i.sqrt() * var_j.sqrt())

        if min_periods is not None:
            condition = n >= LiteFrameLiteral(value=min_periods, dtype=None)
            expr = LiteFrameWhere(
                condition=condition,
                true_value=expr,
                false_value=nan_lit,
                dtype=float64_dtype,
            )

        projection_exprs.append(expr.rename(out_name))

    return projection_exprs


def _compile_pairwise_agg(op, liteframe):
    """Compile a pairwise (cov/corr) aggregation.

    Builds primitives for all unique column pairs (including diagonal),
    plus a post-agg projection that reconstructs the final values.
    ``ddof`` and ``min_periods`` are carried via ``AggCall.kwds``.
    """
    func_name = op.raw_func
    kwds = getattr(op.raw_func, "kwds", {})
    ddof = kwds.get("ddof", 1)
    min_periods = kwds.get("min_periods", None)
    numeric_only = kwds.get("numeric_only", False)
    groupby_params = op.groupby_params
    input_dtypes = liteframe.dtypes
    all_input_dtypes = input_dtypes

    # Determine target columns: all columns excluding group keys and
    # respecting the groupby selection (if any).
    group_key_cols = []
    if groupby_params:
        by = groupby_params["by"]
        group_key_cols = [by] if isinstance(by, str) else list(by)
    selection = groupby_params.get("selection") if groupby_params else None
    all_target_cols = [c for c in input_dtypes.index if c not in group_key_cols]
    if selection is not None:
        sel_set = set(selection)
        all_target_cols = [c for c in all_target_cols if c in sel_set]

    # Enforce numeric_only invariant at the pairwise boundary:
    #   True  → silently exclude non-numeric columns.
    #   False → reject non-numeric target columns (pandas 2.0+ contract).
    non_numeric = [c for c in all_target_cols if not _is_numeric_dtype(input_dtypes[c])]
    if non_numeric and not numeric_only:
        raise TypeError(
            f"{func_name}() requires numeric columns, got non-numeric: {non_numeric}"
        )

    numeric_cols = [c for c in all_target_cols if _is_numeric_dtype(input_dtypes[c])]
    if not numeric_cols:
        raise ValueError(f"No numeric columns available for {func_name}()")

    # Sort so frame column order matches the lexicographic ordering used
    # by _pairwise_matrix_expand's lo/hi lookup — preventing KeyError on
    # frames whose numeric columns are not already sorted (e.g. ['b','a']).
    numeric_cols = sorted(numeric_cols)

    # Generate all unique pairs (i <= j), including diagonal (i == j)
    pairs = [
        (numeric_cols[i], numeric_cols[j])
        for i in range(len(numeric_cols))
        for j in range(i, len(numeric_cols))
    ]

    # Register primitives with deduplication
    agg_funcs = []
    registry = {}  # primitive -> {col_or_pair -> output_key}
    key_counter = [0]
    pair_counter = [0]
    pair_keys = {}  # (ci, cj) -> pair_key string

    def _register(primitive, col_key, pair=None):
        prim_registry = registry.setdefault(primitive, {})
        if col_key in prim_registry:
            return prim_registry[col_key]
        output_key = f"agg_{key_counter[0]}"
        agg_func = _COMBINE_FUNC_MAP.get(primitive, primitive)
        desc = {
            "raw_func_name": primitive,
            "map_func_name": primitive,
            "agg_func_name": agg_func,
            "cols": [col_key],
            "output_key": output_key,
            "kwds": {},
        }
        if primitive in _POWER_SUM_EXPONENTS:
            desc["pre_agg_col"] = f"{output_key}__{col_key}_{primitive}_input"
        elif primitive == _CROSS_PRODUCT_PRIMITIVE:
            desc["pre_agg_col"] = f"{output_key}__{col_key}_{primitive}_input"
            desc["pair"] = pair  # (col_x, col_y) tuple — no string round-trip
        elif primitive in _PAIRWISE_MASK_PRIMITIVES:
            desc["pre_agg_col"] = f"{output_key}__{col_key}_{primitive}_input"
            desc["pair"] = pair  # (col_x, col_y) tuple — no string round-trip
        agg_funcs.append(desc)
        prim_registry[col_key] = output_key
        key_counter[0] += 1
        return output_key

    for ci, cj in pairs:
        _register("sum", ci)
        _register("sum", cj)
        _register("count", ci)
        if func_name == "corr":
            _register("sum_sq", ci)
            _register("sum_sq", cj)
        if ci == cj:
            # Diagonal: sum_xy(A,A) == sum_sq(A) — reuse sum_sq
            if func_name == "cov":
                _register("sum_sq", ci)
        else:
            # Use a unique pair index as the routing key — never derive
            # identity from joined user column names, which collide when
            # names contain the join separator (e.g. pairs (a, b__c) and
            # (a__b, c) both flatten to "a__b__c").  The (ci, cj) tuple
            # is stored structurally in desc["pair"] for expansion.
            pair_key = f"pair_{pair_counter[0]}"
            pair_keys[(ci, cj)] = pair_key
            pair_counter[0] += 1
            _register("sum_xy", pair_key, pair=(ci, cj))
            # Pairwise mask primitives: all use the same both-non-null
            # observation set as sum_xy, matching pandas' pairwise-complete
            # covariance/correlation.
            _register("sum_x_pair", pair_key, pair=(ci, cj))
            _register("sum_y_pair", pair_key, pair=(ci, cj))
            _register("count_pair", pair_key, pair=(ci, cj))
            if func_name == "corr":
                _register("sum_x_pair_sq", pair_key, pair=(ci, cj))
                _register("sum_y_pair_sq", pair_key, pair=(ci, cj))

    # Build post-agg projection expressions
    projection_exprs = _build_pairwise_projection_exprs(
        func_name,
        pairs,
        pair_keys,
        registry,
        groupby_params,
        input_dtypes,
        ddof,
        min_periods,
    )

    # Build pre-agg projection exprs (for sum_sq, sum_xy, and mask primitives)
    pre_agg_projection_exprs = _build_pre_agg_projection_exprs(agg_funcs, input_dtypes)

    # Build intermediate output names and dtypes — use __ joiner to avoid
    # name collisions when column names contain underscores (C3 fix).

    agg_output_columns = [
        (
            f"{ci}__{cj}__{func_name}"
            if ci == cj
            else f"{pair_keys[(ci, cj)]}__{func_name}"
        )
        for ci, cj in pairs
    ]
    float64_dtype = wrap_arrow_dtype(pa.float64())
    agg_output_dtypes_map = {col: float64_dtype for col in agg_output_columns}
    agg_output_dtypes = pd.Series(agg_output_dtypes_map)

    # Build final output dtypes (group keys + projection output)
    out_dtypes_dict = OrderedDict()
    for gk in group_key_cols:
        out_dtypes_dict[gk] = all_input_dtypes.get(gk, wrap_arrow_dtype(pa.string()))
    for col_name in agg_output_columns:
        out_dtypes_dict[col_name] = float64_dtype
    out_dtypes = pd.Series(out_dtypes_dict)

    out_shape = (np.nan if groupby_params else 1, len(out_dtypes))

    return CompileAggResult(
        agg_funcs=agg_funcs,
        agg_output_columns=agg_output_columns,
        output_column_names=None,  # projection handles naming
        agg_output_dtypes=agg_output_dtypes,
        out_dtypes=out_dtypes,
        out_shape=out_shape,
        projection_exprs=projection_exprs,
        pre_agg_projection_exprs=pre_agg_projection_exprs,
    )


def compile_agg(op, liteframe):
    """Compile aggregation: normalize func, build agg_funcs, compute output info.

    Parameters
    ----------
    - agg_funcs: list of agg descriptors
    - agg_output_columns: column names from raw agg output
    - output_column_names: final output column names (with suffix rules)
    - agg_output_dtypes: dtypes for each agg output column
    - out_dtypes: pd.Series of final output dtypes
    - out_shape: tuple (n_rows, n_cols)
    - pre_agg_projection_exprs: list of NamedExpr for pre-agg power transforms (or empty)
    """
    raw_func = op.raw_func
    raw_func_kw = op.raw_func_kw
    groupby_params = op.groupby_params
    input_dtypes = liteframe.dtypes
    all_input_dtypes = input_dtypes  # preserve unfiltered dtypes for group key lookup
    # Detect custom agg (callable or CustomAgg) and delegate
    from maxframe.liteframe.reduction.custom_agg import CustomAgg, compile_custom_agg

    if raw_func is not None and (callable(raw_func) or isinstance(raw_func, CustomAgg)):
        return compile_custom_agg(op, liteframe)

    # Pairwise aggregation (cov, corr) — only via AggCall (from
    # lf.cov()/lf.corr() or groupby().cov()/groupby().corr()).  Bare
    # string agg("cov")/agg("corr") must not enter the pairwise
    # compiler, which would leak internal projection fields.
    # One API, one output layout: direct callers to the right method.
    if (
        isinstance(raw_func, str)
        and not isinstance(raw_func, AggCall)
        and raw_func in ("cov", "corr")
    ):
        if groupby_params:
            raise ValueError(
                f"agg({raw_func!r}) is a pairwise function that produces "
                f"an N×N matrix; call groupby().{raw_func}() directly for "
                f"a predictable output layout"
            )
        raise ValueError(
            f"agg({raw_func!r}) is a pairwise function that produces "
            f"an N×N matrix; call lf.{raw_func}() directly for a "
            f"predictable output layout"
        )
    if isinstance(raw_func, AggCall) and raw_func in ("cov", "corr"):
        return _compile_pairwise_agg(op, liteframe)

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
    # so that the aggregation is only applied to non-key columns.  When an
    # explicit selection is present (``groupby("key")[["a"]]``), further
    # restrict to only the selected columns — unselected columns are not
    # aggregated and do not appear in the output.
    if group_key_cols:
        agg_cols = [c for c in input_cols if c not in group_key_cols]
    else:
        agg_cols = input_cols
    if groupby_params:
        selection = groupby_params.get("selection")
        if selection is not None:
            sel_set = set(selection)
            agg_cols = [c for c in agg_cols if c in sel_set]

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
