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

import re

import numpy as np
import pandas as pd
import pyarrow as pa

from maxframe.liteframe.accessors._docstring import (
    _adapt_liteframe_docstring,
    _copy_signature,
)
from maxframe.liteframe.accessors.string_.core import (
    LiteFrameStrExpr,
    LiteFrameStrMultiExpr,
    StrMethodBaseHandler,
    str_method_to_handlers,
    str_multi_method_to_handlers,
)
from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _resolve_column_expr,
)
from maxframe.liteframe.core import LiteFrameResult
from maxframe.liteframe.expressions import LiteFrameMultiColumnSubExpr
from maxframe.liteframe.functional.map import map_ as lf_map
from maxframe.remote import spawn
from maxframe.utils import wrap_arrow_dtype

# Default dtype for the expanded split columns.
_STR_DTYPE = wrap_arrow_dtype(pa.string())


def _scalar(val):
    """Extract a scalar from a reduction result.

    ``resolve_tileable_input=True`` passes ``LiteFrameResult`` wrappers
    (merge_chunks preserves them for single-chunk LiteFrame reductions);
    materialize each, then extract the scalar from a DataFrame/Series.
    """
    if isinstance(val, LiteFrameResult):
        val = val.materialize().to_pandas()
    if isinstance(val, pd.DataFrame):
        return val.iloc[0, 0]
    if isinstance(val, pd.Series):
        return val.iloc[0]
    return val


def _build_split_dtypes(max_df, n_rows):
    """Build a dtypes Series from the resolved max field count.

    Called inside ``mr.spawn`` with the 1-row result of ``count_lf.max()``
    and the row count of the source (``n_rows``).  Returns a ``pd.Series``
    mapping ``"0".."N-1"`` to the string ArrowDtype.

    Edge cases mirror pandas ``str.split(expand=True)`` semantics:
    - Empty source (``n_rows == 0``): zero expanded columns.
    - Non-empty all-null source (``max`` is ``NaN``): one null column.
    - Non-null rows with zero fields (``max == 0``): zero columns.
    """
    n_rows = _scalar(n_rows)
    max_val = _scalar(max_df)
    if n_rows == 0:
        n_cols = 0
    elif pd.isna(max_val):
        # Non-empty all-null input: pandas produces one null column.
        n_cols = 1
    else:
        n_cols = int(max_val)
    return pd.Series(
        {str(i): _STR_DTYPE for i in range(n_cols)},
    )


def _can_use_native_probe(method, sep, kwargs):
    """Whether polars' native str.split produces field counts identical to
    pandas for this call.

    Polars ``str.split(sep)`` splits literally and ignores ``regex``.  It
    matches pandas only when pandas also splits literally: a single-char
    separator with ``regex`` not True (i.e. ``regex=False`` or the default
    ``regex=None`` with ``len(pat) == 1``).  ``rsplit`` has no native polars
    path, and multi-char/regex/whitespace separators diverge, so those
    route to the Python UDF.
    """
    if method != "split" or not isinstance(sep, str) or len(sep) != 1:
        return False
    return kwargs.get("regex", None) is not True


def _build_dynamic_split_dtypes(source, sep, method="split", args=(), kwargs=None):
    """Build a runtime-resolved dtypes object for ``split``/``rsplit`` expand.

    Pipeline: native ``str.split(sep).list.len().max()`` when polars' literal
    split matches pandas (single-char, non-regex); otherwise a per-row Python
    UDF that replicates pandas semantics (regex/whitespace/rsplit).  Both
    paths feed ``mr.spawn(_build_split_dtypes)`` with (max_field_count,
    row_count).

    The probe and the executor (``eval_dynamic_split``) both ultimately
    resolve the split with pandas semantics — the earlier polars-codegen
    probe ignored ``regex`` and silently truncated columns — but the native
    path is ~300x faster for the common literal single-char case.
    """
    kwargs = kwargs or {}

    # Short-circuit: when the source row count is known to be zero at build
    # time, skip the probe (split -> map -> max fails on empty data — polars
    # represents the empty column as String, so .max() raises).  Returns
    # zero columns, matching pandas str.split(expand=True) on an empty Series.
    if not np.isnan(source.data.shape[0]) and source.data.shape[0] == 0:
        return spawn(
            _build_split_dtypes,
            args=(np.nan, 0),
            resolve_tileable_input=True,
        )

    n = kwargs.get("n", -1)
    if n is None:
        n = -1

    if _can_use_native_probe(method, sep, kwargs):
        # Native polars path: str.split(sep, n) -> list.len() -> max().
        # The codegen's bounded form (_str_split) already produces a
        # pandas-equivalent list (head + conditional tail), so list.len()
        # is the correct clamped field count — no manual min(count, n+1).
        list_lf = source.str.split(sep, expand=False, n=n)
        count_lf = list_lf.list.len()
    else:
        # Slow path: per-row Python UDF replicating pandas split semantics
        # (regex=True, regex=None+len(pat)>1, pat=None, or rsplit).  Uses
        # val.split/re.split directly — far cheaper than pd.Series([val])
        # construction + accessor dispatch (~58us/row → ~5us/row).
        regex = kwargs.get("regex", None)
        use_regex = bool(regex) or (
            regex is None and isinstance(sep, str) and len(sep) > 1
        )
        if sep is None:
            split_fn = lambda v: v.split(None, n)  # noqa: E731
        elif use_regex:
            _rx = re.compile(sep)
            # re.split treats maxsplit <= 0 as "no splits" (unlike str.split
            # which treats n <= 0 as unlimited); pass 0 for unlimited.
            _maxsplit = 0 if n <= 0 else n
            split_fn = lambda v: _rx.split(v, maxsplit=_maxsplit)  # noqa: E731
        elif method == "rsplit":
            split_fn = lambda v: v.rsplit(sep, n)  # noqa: E731
        else:
            split_fn = lambda v: v.split(sep, n)  # noqa: E731

        def _split_field_count(val):
            if not isinstance(val, str):
                return float("nan")
            fields = split_fn(val)
            if fields is None:
                return float("nan")
            return float(len(fields))

        count_lf = lf_map(
            source,
            _split_field_count,
            dtype=wrap_arrow_dtype(pa.float64()),
            _check_output_dtype="warns",
        )

    return spawn(
        _build_split_dtypes,
        args=(count_lf.max(), source.size().max()),
        resolve_tileable_input=True,
    )


class StringAccessor:
    """
    Vectorized string functions for LiteFrame.

    Provides pandas-style .str accessor for single-column LiteFrame.
    Multi-column LiteFrames raise ValueError on accessor access.

    Examples
    --------
    >>> df = LiteFrame({"name": ["alice", "bob"]})
    >>> df.str.upper().execute()
       name
    0 ALICE
    1   BOB
    """

    def __init__(self, liteframe):
        if len(liteframe.columns) != 1:
            raise ValueError("Cannot apply .str accessor to multi-column LiteFrame")
        self._liteframe = liteframe

    def _build_single_str_projection(self, method, args, kwargs, dtype):
        """Build a fused Projection wrapping a single-column LiteFrameStrExpr."""
        col_name = self._liteframe.columns[0]
        col_dtype = self._liteframe.dtypes.iloc[0]
        col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)
        expr = LiteFrameStrExpr(
            operand=col_expr,
            method=method,
            args=args,
            kwargs=kwargs,
            dtype=dtype,
        )
        named_expr = expr.rename(col_name)
        return _build_fused_projection(self._liteframe, [named_expr])

    @classmethod
    def _gen_func(cls, method):
        """Generate method wrapper with dtype inference."""

        def _inner(self, *args, **kwargs):
            handler = str_method_to_handlers.get(method, StrMethodBaseHandler)
            dtype = handler.infer_dtype(method, self._liteframe.dtypes.iloc[0])
            return self._build_single_str_projection(method, args, kwargs, dtype)

        pd_method = getattr(pd.Series.str, method, None)
        if pd_method is not None:
            _inner.__doc__ = _adapt_liteframe_docstring(pd_method.__doc__)
        return _copy_signature(_inner, pd.Series.str, method)

    @classmethod
    def _gen_multi_func(cls, method):
        """Generate method wrapper for multi-column string operations.

        Inspects the call arguments to determine whether the result is
        single-column (falls back to the base path) or multi-column (uses
        the LiteFrameMultiColumnExpr projection path).
        """

        def _inner(self, *args, **kwargs):
            handler = str_multi_method_to_handlers[method]
            input_dtype = self._liteframe.dtypes.iloc[0]
            dtypes_map = handler.infer_dtypes(method, input_dtype, *args, **kwargs)

            # dtypes_map is None → dynamic path (data-dependent column count).
            # Build a count->max->spawn pipeline to discover the schema at
            # execution time, then construct a LiteFrameStrMultiExpr whose
            # dtypes field holds the object tileable.
            if dtypes_map is None:
                return self._build_dynamic_str_projection(method, args, kwargs)

            # If the handler returns an empty mapping, the call produces a
            # single-column result.  split/rsplit with expand=False returns a
            # list-of-string column (pandas semantics).
            # extract with expand=False is a single string column.
            if not dtypes_map:
                if method in ("split", "rsplit") and not kwargs.get("expand", False):
                    # List-of-string output; the codegen emits str.split/rsplit
                    # and the dtype is a list ArrowDtype.
                    dtype = wrap_arrow_dtype(pa.list_(pa.string()))
                else:
                    base_handler = str_method_to_handlers.get(
                        method, StrMethodBaseHandler
                    )
                    dtype = base_handler.infer_dtype(method, input_dtype)
                return self._build_single_str_projection(method, args, kwargs, dtype)

            output_names = [str(k) for k in dtypes_map.keys()]
            col_name = self._liteframe.columns[0]
            col_dtype = self._liteframe.dtypes.iloc[0]
            col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)

            multi_expr = LiteFrameStrMultiExpr(
                operands=[col_expr],
                method=method,
                args=args,
                kwargs=kwargs,
                output_names=output_names,
                dtypes={str(k): str(v) for k, v in dtypes_map.items()},
            )

            projections = []
            for key, name in zip(dtypes_map.keys(), output_names):
                sub_expr = LiteFrameMultiColumnSubExpr(
                    multi_expr=multi_expr,
                    output_name=name,
                    dtype=dtypes_map[key],
                )
                named = sub_expr.rename(name)
                projections.append(named)

            return _build_fused_projection(self._liteframe, projections)

        pd_method = getattr(pd.Series.str, method, None)
        if pd_method is not None:
            _inner.__doc__ = _adapt_liteframe_docstring(pd_method.__doc__)
        return _copy_signature(_inner, pd.Series.str, method)

    def _build_dynamic_str_projection(self, method, args, kwargs):
        """Build a schema-dynamic projection for split/rsplit(expand=True, n<=0).

        Runs the count->max->spawn pipeline to discover the column count at
        execution time, then constructs a LiteFrameStrMultiExpr whose
        ``dtypes`` field holds the object tileable (marking it dynamic).
        The projection carries the dtypes tileable as an extra input.
        """
        # Normalize n=None to n=-1 (pandas treats both as unlimited).  The
        # normalization in SplitHandler.infer_dtypes only affects the
        # dynamic-path decision; the forwarded kwargs must also be normalized
        # so polars str.split doesn't receive n=None and raise TypeError.
        if kwargs.get("n") is None:
            kwargs = {**kwargs, "n": -1}
        sep = args[0] if args else kwargs.get("pat", None)
        dtypes_obj = _build_dynamic_split_dtypes(
            self._liteframe, sep, method, args=args, kwargs=kwargs
        )

        col_name = self._liteframe.columns[0]
        col_dtype = self._liteframe.dtypes.iloc[0]
        col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)

        multi_expr = LiteFrameStrMultiExpr(
            operands=[col_expr],
            method=method,
            args=args,
            kwargs=kwargs,
            output_names=None,
            # Use True as a dynamic marker — the actual dtypes tileable lives
            # in op.inputs[1:] (passed via extra_inputs).  Storing the raw
            # tileable here would break tokenization (Serializable objects are
            # tokenized via cloudpickle, which cannot handle tileable weakrefs).
            dtypes=True,
        )

        return _build_fused_projection(
            self._liteframe, [multi_expr], extra_inputs=(dtypes_obj,)
        )

    @classmethod
    def _register_multi(cls, method):
        """Register a multi-column method to accessor class."""
        func = cls._gen_multi_func(method)
        setattr(cls, method, func)

    @classmethod
    def _register(cls, method):
        """Register method to accessor class."""
        setattr(cls, method, cls._gen_func(method))
