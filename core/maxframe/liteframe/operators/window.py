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

import logging
import operator
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.types as patypes

from maxframe import opcodes
from maxframe.config import options
from maxframe.core.operator import Operator
from maxframe.lib.version import parse as parse_version
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.datatypes import (
    infer_decimal_agg_dtype,
    infer_decimal_arithmetic_dtype,
)
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.liteframe.utils import is_numeric_dtype, normalize_col_name
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FieldTypes,
    Int32Field,
    ListField,
    StringField,
)
from maxframe.utils import wrap_arrow_dtype

logger = logging.getLogger(__name__)

# Aggregation functions supported by the rolling_agg transform kind.
# Polars rolling_* primitives back all of these; "count" is synthesized as a
# non-null rolling sum and is the only one that does not require numeric input.
ROLLING_AGG_FUNCS = frozenset(
    {"sum", "mean", "min", "max", "std", "var", "median", "count"}
)

# Aggregation functions supported by the expanding_agg transform kind.
# pandas Expanding has no .prod() (verified on pandas 2.3.2), so "prod" is
# cum*-only.  ``std``/``var``/``mean`` use additive-moment decomposition
# (sum + count [+ sum2]) and are expanding-only.
EXPANDING_AGG_FUNCS = frozenset({"sum", "mean", "min", "max", "count", "std", "var"})
CUM_AGG_FUNCS = frozenset({"sum", "max", "min", "prod", "count"})

# Aggregation functions supported by the ewm_agg transform kind.
EWM_AGG_FUNCS = frozenset({"mean", "std", "var"})

# Int32 value range bounds, used to validate ``limit`` (an Int32Field) at the
# API boundary before field serialization — out-of-range integers otherwise
# slip past plan-time validation and only fail deep in the DPE executor.
INT32_MIN_VALUE = -2147483648
INT32_MAX_VALUE = 2147483647


def _check_numeric_columns(in_dtypes, columns, kind):
    """Validate that all target columns are numeric. Raise TypeError if not."""
    non_numeric = []
    for col in columns:
        if not is_numeric_dtype(in_dtypes[col]):
            non_numeric.append(f"'{col}' ({in_dtypes[col]})")
    if non_numeric:
        raise TypeError(
            f"{kind} requires numeric columns, but found non-numeric: "
            f"[{', '.join(non_numeric)}]"
        )


def _get_arrow_type(in_dtype):
    """Return the Arrow type for a pandas dtype, or None if not Arrow-backed."""
    return in_dtype.pyarrow_dtype if hasattr(in_dtype, "pyarrow_dtype") else None


def _infer_shift_dtype(in_dtype, fill_value):
    """Infer the output dtype for kind='shift', accounting for fill_value.

    When fill_value is None the input dtype is preserved (shift only
    introduces nulls). When fill_value is provided, Polars widens the
    output if the fill value's natural type is wider than the input column
    type — e.g. an integer column shifted with fill_value=0.5 produces
    Float64 because the float fill value cannot be represented as an
    integer. This function replicates that widening at plan time so the
    planned schema matches the executed frame.

    Verified type promotion rules (Polars 1.3x):

    +-------------------+-------------------+-------------------+
    | Input dtype       | fill_value type   | Output dtype      |
    +===================+===================+===================+
    | Integer (any)     | None              | preserved         |
    | Integer (any)     | int / bool        | preserved         |
    | Integer (any)     | float             | Float64           |
    | Integer (any)     | str               | String            |
    | Float (any)       | any numeric       | preserved         |
    | Float (any)       | str               | String            |
    | String            | anything          | String            |
    | Other (bool, ...) | ...               | preserved         |
    +-------------------+-------------------+-------------------+
    """
    if fill_value is None:
        return in_dtype

    in_arrow = _get_arrow_type(in_dtype)
    if in_arrow is None:
        return in_dtype

    # Resolve the fill_value's natural Arrow type
    try:
        fv_arrow = pa.array([fill_value]).type
    except Exception:
        return in_dtype

    # Null-typed fill_value does not widen the output
    if patypes.is_null(fv_arrow):
        return in_dtype

    # Integer input: float fill -> Float64, string fill -> String,
    # int/bool fill -> preserved.
    if patypes.is_integer(in_arrow):
        if patypes.is_floating(fv_arrow):
            return wrap_arrow_dtype(pa.float64())
        if patypes.is_string(fv_arrow):
            return wrap_arrow_dtype(pa.string())
        return in_dtype

    # Float input: preserved for any numeric fill; string fill -> String.
    if patypes.is_floating(in_arrow):
        if patypes.is_string(fv_arrow):
            return wrap_arrow_dtype(pa.string())
        return in_dtype

    # Boolean input: match pandas (Arrow-backed) semantics. Numeric and bool
    # fills are coerced to bool (0 -> False, non-zero -> True) and the column
    # stays Boolean (the executor casts the fill to bool); polars' native
    # ``shift(fill_value=...)`` instead widens bool+int to Int32 and raises on
    # a float fill. string fills that cannot be coerced are rejected at plan time
    # (pandas-Arrow raises ArrowInvalid for them at execution).
    if patypes.is_boolean(in_arrow):
        if patypes.is_string(fv_arrow):
            raise TypeError(
                f"shift fill_value {fill_value!r} cannot be coerced to "
                f"boolean; bool columns only accept bool or numeric fills"
            )
        return in_dtype

    # Decimal input: Polars widens decimal + scalar fill to
    # decimal128(38, scale). To match the project convention (Hive decimal
    # rules, used by maxframe.dataframe/liteframe arithmetic and
    # aggregation), infer the result via the Hive *add* rule for int/Decimal
    # fills and let the executor cast the result to this dtype so the planned
    # schema and the executed frame agree. A float fill promotes the column
    # to Float64 (verified on Polars 1.3x); other fills preserve the input.
    if patypes.is_decimal(in_arrow):
        if isinstance(fill_value, bool):
            return in_dtype
        if isinstance(fill_value, int):
            result = infer_decimal_arithmetic_dtype(
                in_dtype, np.dtype(np.int64), operator.add
            )
            return result if result is not None else in_dtype
        if isinstance(fill_value, Decimal):
            fill_dtype = _decimal_fill_dtype(fill_value)
            if fill_dtype is not None:
                result = infer_decimal_arithmetic_dtype(
                    in_dtype, fill_dtype, operator.add
                )
                if result is not None:
                    return result
            return in_dtype
        if patypes.is_floating(fv_arrow):
            return wrap_arrow_dtype(pa.float64())
        return in_dtype

    # String, temporal, etc.: preserve input type.
    return in_dtype


def _decimal_fill_dtype(fill_value):
    """Build a dtype for a ``Decimal`` fill_value usable by
    ``infer_decimal_arithmetic_dtype``.

    Returns an ``ArrowDtype(decimal128(p, s))`` whose precision/scale are
    derived from the literal's decimal digits, or ``None`` if the value
    cannot be represented as a decimal operand.
    """
    t = fill_value.as_tuple()
    scale = -t.exponent
    precision = max(len(t.digits), scale if scale > 0 else 1)
    try:
        return wrap_arrow_dtype(pa.decimal128(precision, scale))
    except Exception:
        return None


def _infer_diff_dtype(in_dtype):
    """Infer the output dtype for kind='diff', handling unsigned widening.

    For unsigned integer input, Polars' diff widens to the next larger
    signed integer type, because subtraction can produce negative values
    that cannot be represented in the unsigned input type. For signed
    integers and floats, the input dtype is preserved.

    Mapping (verified against Polars 1.3x):

    +----------+----------+
    | Input    | Output   |
    +==========+==========+
    | UInt8    | Int16    |
    | UInt16   | Int32    |
    | UInt32   | Int64    |
    | UInt64   | Int64    |
    | Int*     | preserved|
    | Float*   | preserved|
    +----------+----------+
    """
    in_arrow = _get_arrow_type(in_dtype)
    if in_arrow is None:
        return in_dtype

    if patypes.is_uint8(in_arrow):
        return wrap_arrow_dtype(pa.int16())
    if patypes.is_uint16(in_arrow):
        return wrap_arrow_dtype(pa.int32())
    if patypes.is_uint32(in_arrow) or patypes.is_uint64(in_arrow):
        return wrap_arrow_dtype(pa.int64())

    return in_dtype


def _infer_agg_dtype(in_dtype, func):
    """Infer the output dtype for a single aggregation column.

    Shared by rolling_agg and expanding_agg dtype inference.
    - sum: decimal -> Hive sum rule; small int -> Int64; else preserve
    - prod: decimal rejected; all signed ints + UInt8/16/32 -> Int64;
      UInt64 preserved (matches Polars cum_prod/product); else preserve
    - mean/std/var: float preserve; else Float64
    - mean/std/var: always Float64 (execution casts to Float64 before
        division/accumulation, even for Float32 input)
    - count: Int64
    - min/max: preserve
    """
    in_arrow = _get_arrow_type(in_dtype)

    if func in ("mean", "std", "var"):
        return wrap_arrow_dtype(pa.float64())
    if func == "count":
        return wrap_arrow_dtype(pa.int64())
    if func == "sum":
        if in_arrow is not None:
            if patypes.is_decimal(in_arrow):
                return infer_decimal_agg_dtype(in_dtype, "sum")
            if patypes.is_integer(in_arrow) and (
                patypes.is_int8(in_arrow)
                or patypes.is_int16(in_arrow)
                or patypes.is_uint8(in_arrow)
                or patypes.is_uint16(in_arrow)
            ):
                return wrap_arrow_dtype(pa.int64())
        return in_dtype
    if func == "prod":
        # Polars cum_prod()/product() widen every integer except UInt64 to
        # Int64 (verified on Polars 1.3x); UInt64 is preserved. Decimal is
        # rejected elsewhere. Without this the planned dtype (e.g. Int32)
        # disagrees with the fetched Int64 result.
        if in_arrow is not None:
            if patypes.is_decimal(in_arrow):
                return infer_decimal_agg_dtype(in_dtype, "prod")
            if patypes.is_integer(in_arrow) and not patypes.is_uint64(in_arrow):
                return wrap_arrow_dtype(pa.int64())
        return in_dtype
    # min / max preserve input dtype
    return in_dtype


def _normalize_order_by(order_by, ascending):
    """Normalize *order_by* / *order_by_ascending* to canonical list form.

    *order_by* accepts str | list[str] | None.  *ascending* accepts
    bool | list[bool].  When *ascending* is a single bool it is broadcast
    to the length of *order_by*.  Returns (order_by, ascending) as lists
    (or (None, None) when *order_by* is falsy).

    These two fields do **not** reorder output columns — they control the
    order in which the window *expands* (equivalent to the ORDER BY clause
    inside a MaxCompute SQL OVER()).  The name order_by (rather than
    order_cols) makes this expansion-order semantics explicit and avoids
    confusion with column reordering.
    """
    if not order_by:
        return None, None
    if isinstance(order_by, str):
        order_by = [order_by]
    # Apply case-insensitive column-name normalization so uppercase spellings
    # match the internally-lowercased frame schema.
    order_by = [normalize_col_name(c) for c in order_by]
    if isinstance(ascending, list):
        if len(ascending) != len(order_by):
            raise ValueError(
                f"order_by_ascending must have the same length as order_by "
                f"({len(order_by)}), got {len(ascending)}"
            )
    else:
        ascending = [ascending] * len(order_by)
    return list(order_by), list(ascending)


def _normalize_columns(columns):
    """Normalize the *columns* argument to a lowercased list (or None).

    Applies ``normalize_col_name`` so an uppercase spelling (e.g. ``'VALUE'``)
    matches the internally-lowercased frame schema, consistent with the rest
    of LiteFrame.  Accepts str | list[str] | None.
    """
    if columns is None:
        return None
    if isinstance(columns, str):
        columns = [columns]
    return [normalize_col_name(c) for c in columns]


def _normalize_by(by):
    """Normalize a groupby ``by`` spec to a list of column names."""
    return [by] if isinstance(by, str) else list(by)


def _is_aggregation_kind(kind, remask_input_nan):
    """Return True for window kinds that *aggregate* rather than *reposition*.

    Positional/cumulative transforms (shift, diff, pct_change, cum*) move
    values between rows but do not reduce them; pandas treats an explicitly
    selected group key as a valid target for these — the key is shifted or
    cumulated alongside the other columns.

    Rolling/expanding *aggregation* (rolling_agg, and expanding_agg without
    ``remask_input_nan``) reduces a window of values to one; pandas never
    aggregates a group key, even one explicitly in the selection — it is
    excluded from the targets and dropped from the output.  MaxFrame follows
    the target-exclusion but, per the "explicit selection survives" principle,
    keeps the selected key as an unchanged passthrough column instead of
    dropping it (a documented divergence from pandas; see window.md §1).

    ``cum*`` maps to ``expanding_agg`` with ``remask_input_nan=True`` and is
    therefore classified as positional/cumulative, not aggregation — matching
    pandas ``groupby.cumsum`` which *does* cumulate a numeric group key.
    """
    if kind in ("shift", "diff", "pct_change", "rank"):
        return False
    if kind == "expanding_agg":
        return not remask_input_nan
    # rolling_agg and any future reducing kind.
    return True


def _resolve_target_cols(columns, groupby_params, hidden_cols, all_cols, is_agg=False):
    """Determine which columns are transform targets.

    Shared by the plan layer (``__call__``) and the executor
    (``_get_target_cols``).  The rules are:

    - groupby with ``selection`` (including empty ``[]`` → zero targets): use
      the selection minus hidden.  ``[]`` is distinct from ``None`` (no
      selection → all columns).  Group keys are excluded from targets when
      ``is_agg`` is True (kept as passthrough) or when there is no selection
      (dropped).
    - otherwise: all columns minus hidden.

    Parameters
    ----------
    columns : list[str] or None
        The ``columns`` field from the window operator.
    groupby_params : dict or None
        The ``groupby_params`` dict (may contain ``by`` and ``selection``).
    hidden_cols : set[str]
        Hidden column names — never transformed.
    all_cols : list[str]
        All available column names (``combined_dtypes.index`` in the plan
        layer, ``data.columns`` in the executor).
    is_agg : bool, default False
        True for rolling/expanding *aggregation* kinds (see
        ``_is_aggregation_kind``).  When True, explicitly selected group keys
        are excluded from transform targets and kept as passthrough.
    """
    excluded = set(hidden_cols)
    if groupby_params is not None:
        by_cols = set(_normalize_by(groupby_params["by"]))
        # Group keys: excluded from targets when aggregating (kept as
        # passthrough) or when there is no selection (dropped by contract).
        selection = groupby_params.get("selection")
        if is_agg or selection is None:
            excluded |= by_cols
        source = (
            columns
            if columns is not None
            else (selection if selection is not None else all_cols)
        )
    else:
        source = columns if columns is not None else all_cols
    return [c for c in source if c not in excluded]


class LiteFrameWindowTransform(LiteFrameOperatorMixin, LiteFrameOperator):
    """Window transform operator for shift/diff/pct_change/rolling_agg.

    Supports standalone and grouped transforms. Uses the sliced-data
    (concatenate-compute-slice) cross-chunk strategy for distributed execution.

    Parameters
    ----------
    kind : str
        Transform kind: "shift", "diff", "pct_change", "rolling_agg",
        "expanding_agg", "ffill", "bfill", or "rank".
    groupby_params : dict or None
        Grouping parameters for grouped transforms. None for whole-frame.
    periods : int
        Position offset (shift/diff/pct_change). Positive = forward,
        negative = backward.
    columns : list of str or None
        Columns to transform. None = all columns. Non-selected columns
        are passed through unchanged.
    window_size : int or None
        Window size in rows (rolling_agg only).
    min_periods : int or None
        Minimum non-null observations in a window (rolling_agg /
        expanding_agg).  Defaults to window_size for rolling.
    center : bool
        Center the window on the current row (rolling_agg only).
    skipna : bool
        NaN handling for cumulative ops (cumsum/cummax/cummin/cumprod/
        cumcount).  Expanding always uses True.
    order_by / order_by_ascending : list or None
        Columns that define the order in which the window expands.
    window_shift : int
        Offset the window frame by N rows.
    reverse : bool
        Flip an expanding window to grow from the last row backward.
    placeback : bool
        If True (default), output is row-aligned with the input.
    transform_params : dict or None
        Calculation parameters packed into a single dict.  Keys depend
        on ``kind``: ``periods`` / ``fill_value`` (shift/diff/pct_change),
        ``agg_func`` / ``remask_input_nan`` (expanding_agg), ``agg_func`` (rolling_agg),
        ``method`` / ``ascending`` / ``na_option`` / ``pct`` (rank).
    """

    _op_type_ = opcodes.LITEFRAME_WINDOW_TRANSFORM

    kind = StringField("kind")
    groupby_params = DictField("groupby_params", default=None)

    # Position offset for shift/diff/pct_change — sizes cross-chunk boundary
    # rows (tiler's _neighbor_row_counts), so it is a window-shape field.
    # positive = forward (data moves down), negative = backward (data moves up)
    periods = Int32Field("periods", default=1)

    columns = ListField("columns", FieldTypes.string, default=None)

    window_size = Int32Field("window_size", default=None)
    min_periods = Int32Field("min_periods", default=None)
    center = BoolField("center", default=False)
    # ``skipna`` controls NaN handling for cumulative ops (cumsum/cummax/
    # cummin/cumprod/cumcount): True skips NaN in the accumulation, False
    # propagates NaN forward. Expanding always uses True. Shift/diff/
    # pct_change/rolling_agg do not consume this field (pandas has no
    # ``skipna`` on those).
    skipna = BoolField("skipna", default=True)

    # --- MF-specific window fields (MaxCompute SQL semantics) ---
    #
    # These four fields mirror the OVER() clause options offered by
    # MaxCompute SQL window functions.  They are orthogonal to the standard
    # pandas window parameters and do *not* change output dtypes — only the
    # window frame / expansion order.
    #
    # order_by / order_by_ascending
    #     Columns that define the order in which the window expands, i.e. the
    #     ORDER BY inside OVER().  The output row order is unchanged;
    #     each row receives the aggregation computed over rows that precede it
    #     in *order_by* order.  order_by_ascending is a per-column bool
    #     list (broadcast from a single bool).
    # window_shift
    #     Offsets the window frame by N rows (the shift MF-specific arg).
    #     For rolling this shifts the window forward (positive) / backward
    #     (negative).  For expanding it extends the leading edge of the
    #     growing window.
    # reverse
    #     Flips an expanding window so that it grows from the last row
    #     backward (a "narrowing" window), matching ROWS BETWEEN … AND
    #     UNBOUNDED FOLLOWING.
    order_by = ListField("order_by", FieldTypes.string, default=None)
    order_by_ascending = ListField("order_by_ascending", FieldTypes.bool, default=None)
    window_shift = Int32Field("window_shift", default=0)
    reverse = BoolField("reverse", default=False)

    # When True (default), the output is row-aligned with the input and
    # inherits its alignment_id.  For standalone transforms this is a
    # metadata-only change (row order is already preserved).  For grouped
    # transforms the tiler inserts a LiteFramePlaceBack step that restores
    # the source chunk layout.  When False the output keeps a fresh id
    # (grouped output stays in reducer order).
    placeback = BoolField("placeback", default=True)

    # Calculation parameters packed into a single dict.  Keys depend on kind:
    #   shift: "fill_value" (scalar or None)
    #   ffill/bfill: "limit" (int, default None)
    #   rolling_agg/expanding_agg: "agg_func" (str)
    #   rank: "method" (str, default "average"), "ascending" (default True),
    #         "na_option" (str, default "keep"), "pct" (bool, default False)
    transform_params = DictField("transform_params", default=None)

    # EWM-specific calculation parameters packed into a single dict.
    # Keys: "com" (float), "adjust" (bool), "ignore_na" (bool),
    # "bias" (bool).  Only set for kind="ewm_agg".
    ewm_params = DictField("ewm_params", default=None)

    @classmethod
    def __serializable__(cls):
        """Return the 2.9 legacy class when the serializer used 2.9.

        ``framedriver_version`` wins; ``client_version`` is the fallback.
        """
        fv = options.session.framedriver_version
        cv = options.session.client_version
        for source, version in (("framedriver", fv), ("client", cv)):
            if not version:
                continue
            try:
                ver = parse_version(version)
            except Exception:
                logger.warning(
                    "Failed to parse %s version %s", source, version, exc_info=True
                )
                continue
            release = getattr(ver, "release", None)
            if not release:
                logger.warning(
                    "Failed to parse %s version %s (legacy version)", source, version
                )
                continue
            if release[:2] == (2, 9):
                logger.info(
                    "Using legacy LiteFrameWindowTransform for 2.9.x %s", source
                )
                return _LegacyLiteFrameWindowTransform_2_9
            if source == "framedriver":
                # A valid FrameDriver version decides the layout; do not fall
                # back to the client version when it is not 2.9.
                return None
        logger.info(
            "Using LiteFrameWindowTransform for client %s (framedriver %s)", cv, fv
        )
        return None

    def _get_transform_param(self, name, default=None):
        """Read a calculation parameter from ``transform_params``."""
        params = self.transform_params or {}
        return params.get(name, default)

    def _inherit_alignment_id(self, inputs):
        if self.placeback:
            return self._inherit_single_input_alignment_id(inputs)
        return None

    @classmethod
    def for_positional(
        cls,
        kind,
        periods=1,
        fill_value=None,
        columns=None,
        groupby_params=None,
        placeback=True,
    ):
        """Build an operator for a positional transform (shift/diff/pct_change).

        Normalizes a single column name to a one-element list so callers can
        accept ``str | list[str] | None`` uniformly. This is shared by the
        standalone (``LiteFrame.shift``) and grouped (``groupby.shift``) entry
        points to avoid repeating the normalization + construction logic.
        """
        columns = _normalize_columns(columns)
        return cls(
            kind=kind,
            periods=periods,
            transform_params={"fill_value": fill_value},
            columns=columns,
            groupby_params=groupby_params,
            placeback=placeback,
        )

    @classmethod
    def for_rank(
        cls,
        method="average",
        ascending=True,
        na_option="keep",
        pct=False,
        columns=None,
        groupby_params=None,
        placeback=True,
    ):
        """Build an operator for a per-group rank transform.

        Rank is a row-count-preserving *transform* (not an aggregation): it
        replaces each value with its numerical rank (1..n). Each target
        column is ranked independently. Output dtype is Float64 for targets
        (matching pandas ``rank``). ``method="first"`` maps to Polars
        ``"ordinal"``. Only ``na_option="keep"`` is supported in v1 (nulls
        stay null).

        ``groupby_params`` must be provided: rank is a **grouped-only**
        window kind (whole-frame ranking is served by ``LiteFrame.rank``).
        """
        if method not in ("average", "min", "max", "first", "dense"):
            raise ValueError(
                f"Invalid method {method!r}, must be one of "
                f"['average', 'min', 'max', 'first', 'dense']"
            )
        if na_option not in ("keep", "top", "bottom"):
            raise ValueError(
                f"Invalid na_option {na_option!r}, must be one of "
                f"['keep', 'top', 'bottom']"
            )
        if na_option in ("top", "bottom"):
            raise NotImplementedError(
                f"na_option={na_option!r} is not supported in v1; "
                f"only 'keep' is supported."
            )
        if groupby_params is None:
            raise TypeError(
                "`rank` requires groupby_params (rank is a grouped-only "
                "window kind); use LiteFrame.rank for whole-frame ranking."
            )
        columns = _normalize_columns(columns)
        return cls(
            kind="rank",
            transform_params={
                "method": method,
                "ascending": ascending,
                "na_option": na_option,
                "pct": pct,
            },
            columns=columns,
            groupby_params=groupby_params,
            placeback=placeback,
        )

    @classmethod
    def for_fill(
        cls, kind, limit=None, columns=None, groupby_params=None, placeback=True
    ):
        """Build an operator for a forward/backward fill (ffill/bfill).

        ``kind`` is "ffill" or "bfill".  ffill propagates the last non-null
        value forward; bfill propagates the next non-null value backward.
        ``limit`` caps consecutive fills per run (None = unlimited);
        non-integral / non-positive / out-of-Int32-range values are rejected
        at the API boundary (matching pandas 3.x).  Output dtype equals the
        input dtype for every target.
        """
        if limit is not None:
            # Match pandas 3.x at the API boundary: reject bool/float/str
            # (ValueError) and normalize integer-like values (incl. numpy
            # integer scalars) via operator.index.  ``<= 0`` is checked
            # before the Int32 range so a negative overflow (-(2**31)-1)
            # raises the pandas-compat ValueError, not OverflowError.
            if isinstance(limit, bool):
                raise ValueError("Limit must be an integer, got " + repr(limit))
            try:
                limit = operator.index(limit)
            except TypeError:
                raise ValueError("Limit must be an integer, got " + repr(limit))
            if limit <= 0:
                raise ValueError("Limit must be greater than 0, got " + repr(limit))
            if limit < INT32_MIN_VALUE or limit > INT32_MAX_VALUE:
                raise OverflowError("Limit value out of Int32 range: " + repr(limit))
        columns = _normalize_columns(columns)
        return cls(
            kind=kind,
            transform_params={"limit": limit},
            columns=columns,
            groupby_params=groupby_params,
            placeback=placeback,
        )

    @classmethod
    def for_cumulative(
        cls, func_name, skipna=True, columns=None, groupby_params=None, placeback=True
    ):
        """Build an operator for a cumulative transform (cumsum/cummax/
        cummin/cumprod/cumcount).

        These are expanding with ``min_periods=1`` and remask semantics.
        ``cumprod`` uses ``agg_func="prod"`` (not exposed on the public
        expanding object — pandas ``Expanding`` has no ``.prod()``).
        ``cumcount`` sets ``row_count_mode`` in ``extra_params`` so the
        executor counts rows (0-indexed) rather than non-null values.
        """
        columns = _normalize_columns(columns)
        agg = "count" if func_name == "cumcount" else func_name[len("cum") :]
        op = cls(
            kind="expanding_agg",
            transform_params={"agg_func": agg, "remask_input_nan": True},
            min_periods=1,
            skipna=skipna,
            columns=columns,
            groupby_params=groupby_params,
            placeback=placeback,
        )
        if func_name == "cumcount":
            op.extra_params["row_count_mode"] = True
        return op

    @classmethod
    def for_expanding(
        cls,
        agg_func,
        min_periods=1,
        columns=None,
        groupby_params=None,
        order_by=None,
        order_by_ascending=True,
        window_shift=0,
        reverse=False,
        placeback=True,
    ):
        """Build an operator for ``expanding().<func>()``.

        Expanding uses fill semantics (no remask) and always skips NaN in
        accumulation (pandas expanding has no ``skipna`` parameter).
        """
        columns = _normalize_columns(columns)
        order_by, order_by_ascending = _normalize_order_by(order_by, order_by_ascending)
        return cls(
            kind="expanding_agg",
            transform_params={"agg_func": agg_func},
            min_periods=min_periods,
            skipna=True,
            columns=columns,
            groupby_params=groupby_params,
            order_by=order_by,
            order_by_ascending=order_by_ascending,
            window_shift=window_shift,
            reverse=reverse,
            placeback=placeback,
        )

    @classmethod
    def for_ewm(
        cls,
        func,
        com,
        min_periods=0,
        *,
        adjust=True,
        ignore_na=False,
        bias=False,
        columns=None,
        groupby_params=None,
        order_by=None,
        order_by_ascending=True,
        window_shift=0,
        reverse=False,
        placeback=True,
    ):
        """Build an operator for ``ewm_agg``.

        ``func`` ("mean"/"std"/"var") is stored in
        ``transform_params["agg_func"]`` — same convention as rolling_agg and
        expanding_agg.  EWM-specific parameters are packed into ``ewm_params``.
        ``com`` (center of mass) is stored as the canonical decay parameter.
        """
        if func not in EWM_AGG_FUNCS:
            raise ValueError(
                f"Unsupported EWM aggregation function: {func!r}. "
                f"Supported: {sorted(EWM_AGG_FUNCS)}"
            )
        columns = _normalize_columns(columns)
        order_by, order_by_ascending = _normalize_order_by(order_by, order_by_ascending)
        return cls(
            kind="ewm_agg",
            transform_params={"agg_func": func},
            ewm_params={
                "com": com,
                "adjust": adjust,
                "ignore_na": ignore_na,
                "bias": bias,
            },
            min_periods=min_periods,
            columns=columns,
            groupby_params=groupby_params,
            order_by=order_by,
            order_by_ascending=order_by_ascending,
            window_shift=window_shift,
            reverse=reverse,
            placeback=placeback,
        )

    def _infer_rolling_agg_dtypes(self, in_dtypes, target_cols):
        """Validate parameters and infer output dtypes for ``kind='rolling_agg'``.

        Returns a copy of ``in_dtypes`` with target columns updated for the
        chosen aggregation. Raises ``ValueError``/``TypeError`` at plan time
        for unsupported configurations or non-numeric targets.
        """
        if self.window_size is None or self.window_size < 1:
            raise ValueError(
                f"rolling requires a positive integer window_size, "
                f"got {self.window_size!r}"
            )
        func = self._get_transform_param("agg_func")
        if func not in ROLLING_AGG_FUNCS:
            raise ValueError(
                f"Unsupported rolling aggregation function: {func!r}. "
                f"Supported: {sorted(ROLLING_AGG_FUNCS)}"
            )
        # count is a non-null tally and works on any dtype; all other
        # aggregations require numeric input (plan-time fail-fast).
        if func != "count":
            _check_numeric_columns(in_dtypes, target_cols, f"rolling.{func}")

        new_dtypes = in_dtypes.copy()
        for col in target_cols:
            # median maps to mean/std/var dtype logic (Float64 for int, preserve float)
            dt = "mean" if func == "median" else func
            new_dtypes[col] = _infer_agg_dtype(in_dtypes[col], dt)
        return new_dtypes

    def _infer_expanding_agg_dtypes(self, in_dtypes, target_cols):
        """Validate parameters and infer output dtypes for
        ``kind='expanding_agg'``.

        Returns a copy of ``in_dtypes`` with target columns updated for the
        chosen aggregation. Raises ``ValueError``/``TypeError`` at plan time
        for unsupported configurations, non-numeric targets, or decimal
        cumprod (polars ``cum_prod`` does not support decimal — verified on
        Polars 1.36).
        """
        func = self._get_transform_param("agg_func")
        if self._get_transform_param("remask_input_nan", False):
            # cum*: validate against CUM_AGG_FUNCS
            if func not in CUM_AGG_FUNCS:
                raise ValueError(
                    f"Unsupported cumulative aggregation function: {func!r}. "
                    f"Supported: {sorted(CUM_AGG_FUNCS)}"
                )
        else:
            # expanding: validate against EXPANDING_AGG_FUNCS
            if func not in EXPANDING_AGG_FUNCS:
                raise ValueError(
                    f"Unsupported expanding aggregation function: {func!r}. "
                    f"Supported: {sorted(EXPANDING_AGG_FUNCS)}"
                )

        if self.min_periods is not None and self.min_periods < 1:
            raise ValueError(f"min_periods must be >= 1, got {self.min_periods!r}")

        # count works on any dtype; all other aggregations require numeric.
        if func != "count":
            _check_numeric_columns(in_dtypes, target_cols, f"expanding_agg.{func}")

        # cumprod on decimal is rejected — polars cum_prod doesn't support it.
        if func == "prod":
            for col in target_cols:
                in_arrow = _get_arrow_type(in_dtypes[col])
                if in_arrow is not None and patypes.is_decimal(in_arrow):
                    raise TypeError(
                        f"cumprod does not support decimal columns, but found "
                        f"'{col}' ({in_dtypes[col]}). Polars cum_prod does not "
                        f"support decimal types."
                    )

        new_dtypes = in_dtypes.copy()
        for col in target_cols:
            new_dtypes[col] = _infer_agg_dtype(in_dtypes[col], func)
        return new_dtypes

    def _infer_ewm_agg_dtypes(self, in_dtypes, target_cols):
        """Validate parameters and infer output dtypes for ``kind='ewm_agg'``.

        All EWM functions (mean, std, var) produce Float64 for every target
        column, matching pandas EWM semantics.  All require numeric input.
        """
        func = self._get_transform_param("agg_func")
        if func not in EWM_AGG_FUNCS:
            raise ValueError(
                f"Unsupported EWM aggregation function: {func!r}. "
                f"Supported: {sorted(EWM_AGG_FUNCS)}"
            )
        _check_numeric_columns(in_dtypes, target_cols, f"ewm.{func}")

        new_dtypes = in_dtypes.copy()
        for col in target_cols:
            new_dtypes[col] = wrap_arrow_dtype(pa.float64())
        return new_dtypes

    def __call__(self, liteframe):
        # ``reverse`` is an expanding/EWM-only feature — fail fast before
        # touching any data (it does not depend on the input schema).
        if self.reverse and self.kind not in ("expanding_agg", "ewm_agg"):
            raise ValueError(
                "`reverse` is only supported for expanding and ewm windows, "
                f"not kind={self.kind!r}"
            )
        # ``rank`` is grouped-only: whole-frame ranking is served by
        # ``LiteFrame.rank`` (PSRS), so a standalone rank window operator is
        # rejected rather than silently ranked per chunk.
        if self.kind == "rank" and self.groupby_params is None:
            raise TypeError(
                "`rank` is only supported for grouped transforms; "
                "use LiteFrame.rank for whole-frame ranking."
            )

        in_dtypes = liteframe._data._physical_dtypes
        in_metadata = liteframe.frame_metadata

        # Materialize virtual range columns into dtypes for target selection,
        # matching sort.py's approach.  Range columns that are transform
        # targets are removed from the output range_columns and become
        # physical columns with their inferred dtype.
        input_range_columns = (
            in_metadata.range_columns
            if in_metadata and in_metadata.range_columns
            else {}
        )
        if input_range_columns:
            rc_dtypes = {
                rc_name: wrap_arrow_dtype(pa.int64()) for rc_name in input_range_columns
            }
            combined_dtypes = pd.concat([pd.Series(rc_dtypes), in_dtypes])
        else:
            combined_dtypes = in_dtypes

        # Collect hidden columns — they are never transformed
        hidden_cols = set()
        if in_metadata is not None and in_metadata.hidden_columns:
            hidden_cols = set(in_metadata.hidden_columns)

        # --- MF-specific window field validation ---
        # ``order_by`` columns must exist in the input frame.
        if self.order_by:
            missing = [c for c in self.order_by if c not in combined_dtypes.index]
            if missing:
                raise KeyError(f"order_by column(s) not found in frame: {missing}")

        # columns= / order_by= must reference cols in the groupby selection
        # (the selection narrows the frame before the shuffle).
        if self.groupby_params and self.groupby_params.get("selection") is not None:
            sel_set = set(self.groupby_params["selection"])
            for attr_name, label in [("columns", "columns"), ("order_by", "order_by")]:
                attr_val = getattr(self, attr_name, None)
                if not attr_val:
                    continue
                outside = [c for c in attr_val if c not in sel_set]
                if outside:
                    raise KeyError(
                        f"{label} {outside} not in groupby selection "
                        f"{list(sel_set)}"
                    )

        # Group-key-in-columns validation (operation- and selection-aware).
        #
        # Without a selection, group keys are always dropped (pandas
        # groupby.shift/rolling never include the key), so naming one in
        # ``columns`` is contradictory and rejected.  With a selection, a
        # group key is a valid target for positional/cumulative transforms
        # and excluded-from-target passthrough for aggregations — see
        # ``_is_aggregation_kind`` and window.md §1 for the full contract.
        is_agg = _is_aggregation_kind(
            self.kind, self._get_transform_param("remask_input_nan", False)
        )
        if (
            self.columns is not None
            and self.groupby_params is not None
            and self.groupby_params.get("selection") is None
        ):
            by_cols = set(_normalize_by(self.groupby_params["by"]))
            key_in_cols = by_cols & set(self.columns)
            if key_in_cols:
                raise ValueError(
                    f"Group key column(s) {sorted(key_in_cols)} cannot be "
                    f"transform targets; without a groupby selection, group "
                    f"keys are always dropped. Remove them from `columns` "
                    f"or add them to the groupby selection."
                )

        # Determine target columns (exclude hidden columns from transforms).
        # For aggregation kinds, selected group keys are excluded from targets
        # and kept as passthrough (see ``_is_aggregation_kind``).
        target_cols = _resolve_target_cols(
            self.columns,
            self.groupby_params,
            hidden_cols,
            combined_dtypes.index,
            is_agg=is_agg,
        )

        # Compute output dtypes
        if self.kind == "shift":
            new_dtypes = combined_dtypes.copy()
            fill_value = self._get_transform_param("fill_value")
            if fill_value is not None:
                for col in target_cols:
                    new_dtypes[col] = _infer_shift_dtype(
                        combined_dtypes[col], fill_value
                    )
        elif self.kind == "diff":
            _check_numeric_columns(combined_dtypes, target_cols, self.kind)
            new_dtypes = combined_dtypes.copy()
            for col in target_cols:
                new_dtypes[col] = _infer_diff_dtype(combined_dtypes[col])
        elif self.kind == "pct_change":
            _check_numeric_columns(combined_dtypes, target_cols, self.kind)
            new_dtypes = combined_dtypes.copy()
            for col in target_cols:
                new_dtypes[col] = wrap_arrow_dtype(pa.float64())
        elif self.kind == "rolling_agg":
            new_dtypes = self._infer_rolling_agg_dtypes(combined_dtypes, target_cols)
        elif self.kind == "expanding_agg":
            new_dtypes = self._infer_expanding_agg_dtypes(combined_dtypes, target_cols)
        elif self.kind == "ewm_agg":
            new_dtypes = self._infer_ewm_agg_dtypes(combined_dtypes, target_cols)
        elif self.kind in ("ffill", "bfill"):
            # ffill/bfill copy existing values; forward_fill/backward_fill
            # are non-arithmetic, so the output dtype equals the input
            # dtype for every target (no widening, even for decimal).
            new_dtypes = combined_dtypes.copy()
        elif self.kind == "rank":
            # Rank returns float64 for every target column (pandas rank()
            # always returns float64). Non-target columns keep their dtype.
            new_dtypes = combined_dtypes.copy()
            for col in target_cols:
                new_dtypes[col] = wrap_arrow_dtype(pa.float64())
        else:
            raise ValueError(f"Unknown window transform kind: {self.kind}")

        # Groupby: narrow visible output, matching pandas (see window.md §1).
        if self.groupby_params is not None:
            by_cols = _normalize_by(self.groupby_params["by"])
            by_set = set(by_cols)
            sel = self.groupby_params.get("selection")
            if sel is not None:
                keep = [c for c in sel if c in new_dtypes.index]
            else:
                keep = [c for c in new_dtypes.index if c not in by_set]
            keep += [c for c in new_dtypes.index if c in hidden_cols and c not in keep]
            # Polars cannot represent (N, 0); add a structural hidden group key.
            if not keep:
                structural = next((c for c in by_cols if c in new_dtypes.index), None)
                if structural:
                    keep = [structural]
                    hidden_cols = hidden_cols | {structural}
            new_dtypes = new_dtypes[keep]

        # Preserve metadata.
        # range_columns are preserved for standalone transforms (row order
        # unchanged). For grouped transforms the hash shuffle reorders rows
        # into reducer/partition order, so range_columns are no longer
        # meaningful and are dropped (LiteFrameGroupByOp already strips
        # range_columns from its output metadata) — see docs "Grouped
        # transform row-order limitation".
        # Range columns that were transformed are no longer range columns.
        transformed_rcs = set(target_cols) & set(input_range_columns)
        if self.groupby_params is not None:
            new_range_columns = {}
        else:
            new_range_columns = {
                k: v for k, v in input_range_columns.items() if k not in transformed_rcs
            }
        # Non-transformed range columns remain virtual — remove them from
        # physical_dtypes so they are not double-counted.
        for rc in input_range_columns:
            if rc not in transformed_rcs and rc in new_dtypes.index:
                new_dtypes = new_dtypes.drop(rc)
        new_metadata = FrameMetadata(
            range_columns=new_range_columns,
            hidden_columns=list(hidden_cols) if hidden_cols else None,
        )
        n_rows = liteframe.shape[0]
        shape = (n_rows, len(new_dtypes) + len(new_metadata.range_columns))

        return self.new_liteframe(
            [liteframe],
            shape=shape,
            physical_dtypes=new_dtypes,
            frame_metadata=new_metadata,
        )


class _LegacyLiteFrameWindowTransform_2_9(Operator):
    """Legacy 2.9 layout of ``LiteFrameWindowTransform`` for deserializing
    requests from 2.9.x clients.

    In 2.9, ``fill_value``, ``limit``, ``agg_func``, and
    ``remask_input_nan`` were declared top-level fields.  They have since
    been consolidated into ``transform_params``.  This class mirrors the
    2.9 field order so the serialization framework can deserialize 2.9
    bytes, then ``__on_deserialize__`` converts to the current operator.
    """

    _legacy_name = "LiteFrameWindowTransform"
    _ignore_non_existing_keys = True

    kind = StringField("kind")
    groupby_params = DictField("groupby_params", default=None)
    periods = Int32Field("periods", default=1)
    fill_value = AnyField("fill_value", default=None)
    limit = Int32Field("limit", default=None)
    columns = ListField("columns", FieldTypes.string, default=None)
    window_size = Int32Field("window_size", default=None)
    min_periods = Int32Field("min_periods", default=None)
    agg_func = StringField("agg_func", default=None)
    center = BoolField("center", default=False)
    skipna = BoolField("skipna", default=True)
    remask_input_nan = BoolField("remask_input_nan", default=False)
    order_by = ListField("order_by", FieldTypes.string, default=None)
    order_by_ascending = ListField("order_by_ascending", FieldTypes.bool, default=None)
    window_shift = Int32Field("window_shift", default=0)
    reverse = BoolField("reverse", default=False)

    def __on_deserialize__(self):
        tp = {
            k: getattr(self, k)
            for k in ("fill_value", "limit", "agg_func")
            if getattr(self, k) is not None
        }
        tp["remask_input_nan"] = self.remask_input_nan

        target = LiteFrameWindowTransform.__new__(LiteFrameWindowTransform)
        target_fields = type(target)._PROPS._FIELDS
        for k, field in type(self)._PROPS._FIELDS.items():
            if k in target_fields:
                try:
                    target_fields[k].set(target, field.__get__(self, k))
                except AttributeError:
                    continue
        if target._outputs is None:
            target._outputs = []
        target.transform_params = tp
        return target
