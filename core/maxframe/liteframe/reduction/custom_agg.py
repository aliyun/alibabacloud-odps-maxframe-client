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

"""Custom aggregation support for LiteFrame.agg().

Provides the ``CustomAgg`` base class (pre / agg / post three-step pattern)
and ``compile_custom_agg`` which lowers a custom agg into pre/post
``transform_chunk`` projections and a tree-reducible agg descriptor.
"""

import copy
from typing import Optional

import cloudpickle
import numpy as np
import pandas as pd
import pyarrow as pa

from maxframe.liteframe.expressions import (
    LiteFrameColumn,
    LiteFrameMultiColumnSubExpr,
    LiteFrameUDFExpr,
    NamedExpr,
)
from maxframe.liteframe.functional.type_infer import infer_output_dtypes
from maxframe.liteframe.functional.utils import _build_scheduling_hint_from_func
from maxframe.liteframe.reduction.compile import CompileAggResult
from maxframe.udf import BuiltinFunction
from maxframe.utils import wrap_arrow_dtype


class CustomAgg:
    """Base class for liteframe custom aggregations.

    Subclasses override ``pre``/``agg``/``post``.  A single function passed
    to ``agg()`` is wrapped as agg-only (``pre``/``post`` are identity).

    ``agg`` MUST be associative: it is called at map, combine, and agg stages
    of tree reduction — the same function re-aggregates its own output.
    """

    name: Optional[str] = None

    def __init__(self, name=None):
        self.name = name or "<custom>"

    @property
    def __name__(self):
        return self.name

    def pre(self, df):
        return df

    def agg(self, df):
        raise NotImplementedError

    def post(self, df):
        return df

    def __maxframe_tokenize__(self):
        return cloudpickle.dumps(self)


class _SingleFuncAgg(CustomAgg):
    """Wrap a single callable as an agg-only CustomAgg (identity pre/post)."""

    def __init__(self, func):
        super().__init__(name=getattr(func, "__name__", "<custom>"))
        self._func = func

    def agg(self, df):
        return self._func(df)

    def __maxframe_tokenize__(self):
        return cloudpickle.dumps(self._func)


def _resolve_custom_agg(raw_func):
    """Normalize a raw_func into a CustomAgg instance or BuiltinFunction.

    BuiltinFunction wrappers are kept as-is so ``has_custom_code`` returns
    False.  Plain callables are wrapped in ``_SingleFuncAgg``.
    """
    if isinstance(raw_func, (CustomAgg, BuiltinFunction)):
        return raw_func
    if isinstance(raw_func, type) and issubclass(raw_func, CustomAgg):
        return raw_func()
    if callable(raw_func):
        return _SingleFuncAgg(raw_func)
    raise TypeError(f"Cannot use {type(raw_func)} as a custom aggregation")


def _is_identity_method(method):
    """Check if a method is the default identity implementation."""
    return method is CustomAgg.pre or method is CustomAgg.post


def _class_overrides(cls, method_name):
    """True if *cls* overrides *method_name* before reaching ``CustomAgg``.

    Avoids the ``BuiltinMethod.__func__`` re-wrap that breaks the identity
    check for inherited hooks.
    """
    for klass in cls.__mro__:
        if klass is CustomAgg:
            break
        if method_name in klass.__dict__:
            return True
    return False


def _resolve_scheduling_source(custom_agg, hook_name):
    """Resolve the object whose running options should reach a hook.

    ``copy_func_scheduling_hints`` / ``_build_scheduling_hint_from_func`` /
    ``discover_marked_functions`` must find the ``@with_running_options``
    ``MarkedFunction`` that decorated the hook, but the naive
    ``custom_func.<hook>`` reads through the wrong layer for two
    ``@builtin_function`` shapes:

    - Standalone builtin callable (a plain function wrapped in
      ``BuiltinFunction``): ``custom_func.<hook>`` routes through
      ``BuiltinFunction.__getattr__`` which, because ``self.func`` is a
      function (not a type), returns ``getattr(self.func, hook)`` and raises
      ``AttributeError`` for any hook besides the callable itself.  The
      underlying ``custom_agg.func`` *is* the decorated ``MarkedFunction`` (or
      the plain callable); reading it directly lets
      ``discover_marked_functions`` find the options.  Returning the outer
      ``BuiltinFunction`` instead loses the hint: ``discover_marked_functions``
      only traverses plain functions (``__code__``/``__closure__``/``__globals__``)
      and never descends into a ``BuiltinFunction``'s ``_func``.
    - Builtin ``CustomAgg`` class: ``custom_func.<hook>`` returns a
      ``BuiltinMethod`` whose ``func`` reloads the class, instantiates it, and
      returns ``getattr(instance, hook)`` — a ``functools.partial`` around the
      ``MarkedFunction`` that ``discover_marked_functions`` never reaches
      through ``BuiltinMethod``, so the hint is lost.  The raw class attribute
      ``underlying.__dict__[hook]`` *is* the decorated ``MarkedFunction``
      before any instance/builtin re-wrap; reading it directly lets
      ``discover_marked_functions`` find the options.

    For a plain ``CustomAgg`` instance / ``_SingleFuncAgg`` wrapper,
    ``custom_agg.<hook>`` is already the bound, partial-wrapped
    ``MarkedFunction`` (or the plain callable) — the existing working path
    (returned as ``None`` here only for the agg shape, which the caller
    resolves via ``custom_agg.agg`` directly).

    ``hook_name`` is ``"pre"``/``"agg"``/``"post"``.  The agg shape needs the
    standalone-callable resolution (``custom_agg.func``) and a fallback to the
    inherited ``underlying.agg`` when the class doesn't override ``agg``; the
    pre/post shapes only need the raw class attribute (``None`` when absent,
    meaning no decorated hook).
    """
    if isinstance(custom_agg, BuiltinFunction):
        underlying = custom_agg.func
        if isinstance(underlying, type) and issubclass(underlying, CustomAgg):
            # Builtin CustomAgg class: read the raw decorated class attr,
            # falling back to the inherited method for agg (identity, no
            # options) when the class doesn't override it.
            if hook_name == "agg":
                return underlying.__dict__.get("agg", underlying.agg)
            return underlying.__dict__.get(hook_name)
        # Standalone builtin callable: the underlying decorated function (a
        # MarkedFunction when @with_running_options is stacked below
        # @builtin_function, else the plain callable).  discover_marked_functions
        # finds the hint on the MarkedFunction directly; the outer
        # BuiltinFunction is opaque to it.
        return underlying
    # Plain CustomAgg instance / _SingleFuncAgg: only the agg shape reaches
    # here (pre/post are resolved as BuiltinMethod at the call site for the
    # builtin shape, or as bound methods for the plain shape).
    if hook_name == "agg":
        return custom_agg.agg
    return None


def _build_udf_expr(
    func, input_dtypes, output_dtypes, input_frame_type, scheduling_source=None
):
    """Build a LiteFrameUDFExpr for a pre or post step.

    Propagates the UDF's running options (``@with_running_options`` /
    ``@with_resources`` / ``@with_fs_mount`` / ...) as a
    ``SchedulingHint`` on the expression, mirroring
    ``functional.utils.build_udf_projections``: the projection tiler reads
    ``LiteFrameUDFExpr.scheduling_hint`` for color-grouping and split-op
    resource allocation, so dropping it (the prior regression) lets a
    decorated ``pre``/``post`` run without its requested CPU/memory.
    ``_build_scheduling_hint_from_func`` returns ``None`` when the function
    carries no running options.

    ``func`` is stored on the expression for *execution* (it must be the
    callable the executor invokes — a ``BuiltinMethod`` for the builtin-class
    shape, so ``self`` binds correctly).  ``scheduling_source`` overrides the
    object used for *hint extraction* when ``func`` is opaque to
    ``discover_marked_functions``: a ``BuiltinMethod`` reloads+rebinds the
    class, so ``discover_marked_functions`` cannot traverse it to the
    underlying decorated ``MarkedFunction``.  For the builtin-class shape the
    raw class attribute (``underlying.__dict__["pre"]``) is the
    ``MarkedFunction`` ``discover_marked_functions`` can read; pass it here
    so a decorated builtin-class ``pre``/``post`` does not silently drop its
    running options.
    """
    output_names = list(output_dtypes.index)
    dtypes = dict(output_dtypes)
    input_operands = [
        LiteFrameColumn(name=c, dtype=input_dtypes[c]) for c in input_dtypes.index
    ]
    hint_source = scheduling_source if scheduling_source is not None else func
    scheduling_hint = _build_scheduling_hint_from_func(hint_source)
    udf_expr = LiteFrameUDFExpr(
        func=func,
        mode="transform_chunk",
        operands=input_operands,
        output_names=output_names,
        dtypes=dtypes,
        input_frame_type=input_frame_type,
        check_output_dtypes="raises",
        args=(),
        kwargs={},
        scheduling_hint=scheduling_hint,
    )
    # Compile-time assertion that the hint actually landed on the expression:
    # a future refactor that decouples ``LiteFrameUDFExpr`` from
    # ``scheduling_hint`` would silently reintroduce the regression, so fail
    # fast at compile.  Identity is the right check — the field stores the
    # reference verbatim at construction.
    assert udf_expr.scheduling_hint is scheduling_hint, (
        "scheduling_hint not propagated: "
        f"func={getattr(func, '__name__', func)!r} "
        f"computed={scheduling_hint!r} expr={udf_expr.scheduling_hint!r}"
    )
    return udf_expr


def _wrap_projections(udf_expr):
    """Wrap a LiteFrameUDFExpr into NamedExpr projections."""
    projections = []
    for col_name in udf_expr.output_names:
        sub_expr = LiteFrameMultiColumnSubExpr(
            multi_expr=udf_expr,
            output_name=col_name,
            dtype=udf_expr.dtypes[col_name],
        )
        projections.append(
            NamedExpr(name=col_name, expr=sub_expr, dtype=udf_expr.dtypes[col_name])
        )
    return projections


def _build_hook_projection(
    hook_func,
    input_dtypes,
    output_dtypes,
    input_frame_type,
    group_key_cols,
    full_dtypes,
    custom_agg,
    hook_name,
):
    """Build a pre/post hook projection: a UDF expr bound to the *original*
    hook (not the inference clone) plus framework-owned group-key
    pass-through.  Scheduling hints are resolved from the raw class
    attribute for the builtin-class shape (see ``_resolve_scheduling_source``).
    """
    udf_expr = _build_udf_expr(
        hook_func,
        input_dtypes,
        output_dtypes,
        input_frame_type,
        scheduling_source=_resolve_scheduling_source(custom_agg, hook_name),
    )
    user_projections = _wrap_projections(udf_expr)
    return _key_pass_through(group_key_cols, full_dtypes) + [
        p for p in user_projections if p.name not in group_key_cols
    ]


def _key_pass_through(group_key_cols, full_dtypes):
    """Build framework-owned group-key pass-through NamedExprs.

    Group keys are always passed through as the *original* input column,
    overriding any same-named column a user ``pre``/``post`` returns (e.g.
    ``return df.assign(key=0)`` must not collapse distinct groups).
    """
    projections = []
    for gk in group_key_cols:
        gk_dtype = full_dtypes.get(gk, wrap_arrow_dtype(pa.string()))
        projections.append(
            NamedExpr(
                name=gk, expr=LiteFrameColumn(name=gk, dtype=gk_dtype), dtype=gk_dtype
            )
        )
    return projections


def _reject_group_key_collisions(group_key_cols, hook_dtypes):
    """Raise if any hook output column name collides with a group key.

    Group keys are framework-owned under the ``as_index=False`` column-based
    groupby contract: the executor re-materializes them from the group tuple
    (``_custom_groupby_agg`` does ``res.with_columns(key_cols)``),
    unconditionally overwriting any same-named user column, so a
    user-produced group-key-named output is silently discarded.  Group keys
    are also excluded from the UDF input, so a same-named user output can
    never be a pass-through of the original key; rejecting it is safe.  Every
    hook boundary (pre/agg/post) is checked unconditionally — the earlier
    lineage exemption (rejecting only newly-introduced key-named columns) left
    a chained ``pre -> agg`` re-declaration hole.
    """
    for hook_name, dtypes in hook_dtypes:
        collisions = [c for c in dtypes.index if c in group_key_cols]
        if collisions:
            raise ValueError(
                f"{hook_name} output column(s) collide with group-key "
                f"name(s) {collisions}: rename the {hook_name} output or "
                "the group key. Group keys are framework-owned; a "
                "same-named user output is silently discarded."
            )


def compile_custom_agg(op, liteframe):
    """Compile a custom agg into pre-projection, agg descriptor, and post-projection.

    Returns a :class:`CompileAggResult` with ``custom_pre_projection`` and
    ``custom_post_projection`` set in addition to the standard fields.
    """
    raw_func = op.raw_func
    full_dtypes = liteframe.dtypes
    input_frame_type = getattr(op, "input_frame_type", "pandas") or "pandas"
    groupby_params = op.groupby_params
    # Group key columns (computed once; reused for input narrowing, pre
    # pass-through preservation, and output metadata).  Per the groupby
    # selection contract (pandas 3.0): keys are excluded from the UDF input
    # by default, and ``selection`` further restricts non-key columns.
    # Key dtypes come from the full frame, not the narrowed schema.
    input_dtypes = full_dtypes
    group_key_cols = []
    if groupby_params:
        by = groupby_params["by"]
        group_key_cols = [by] if isinstance(by, str) else list(by)
        selection = groupby_params.get("selection")
        if selection is not None:
            input_cols = [c for c in selection if c not in group_key_cols]
        else:
            input_cols = [c for c in full_dtypes.index if c not in group_key_cols]
        # Polars can't represent a zero-column frame with nonzero height, so
        # explicit-empty selection + polars is unsatisfiable and would silently
        # leak every column into the UDF.  Reject here.
        if (
            selection is not None
            and len(input_cols) == 0
            and input_frame_type == "polars"
        ):
            raise ValueError(
                "explicit-empty selection (groupby(...)[[]]) is "
                "unsatisfiable with input_frame_type='polars': polars "
                "cannot represent a zero-column frame with nonzero "
                "height. Use input_frame_type='pandas' or select at "
                "least one column."
            )
        input_dtypes = full_dtypes.reindex(input_cols)

    custom_agg = _resolve_custom_agg(raw_func)

    # Extract the three step functions.
    if isinstance(custom_agg, BuiltinFunction):
        underlying = custom_agg.func
        if isinstance(underlying, type) and issubclass(underlying, CustomAgg):
            pre_func = custom_agg.pre
            post_func = custom_agg.post
            # BuiltinMethod.__func__ re-wraps as a BuiltinMethod, so the
            # identity check misclassifies inherited hooks as overrides;
            # walk the class MRO instead.
            has_pre = _class_overrides(underlying, "pre")
            has_post = _class_overrides(underlying, "post")
        else:
            # Plain function wrapped in BuiltinFunction: agg-only
            pre_func = CustomAgg.pre
            post_func = CustomAgg.post
            has_pre = False
            has_post = False
        # Resolve bound instance methods for inference so return-type
        # annotations are reachable AND mock inference can call the body.
        # ``pre_func``/``post_func`` are ``BuiltinMethod``
        # objects (or the ``BuiltinFunction`` for the standalone-callable
        # shape); a ``BuiltinMethod`` has no ``__wrapped__`` and
        # ``unwrap_function`` returns it unchanged, so
        # ``_get_return_annotation`` cannot descend to the user's annotated
        # method and mock inference is the only fallback -- which executes
        # the body via ``BuiltinMethod``'s reload+instantiate path.  Bind
        # the hooks on a fresh instance of the underlying class instead:
        # ``getattr(instance, "agg")`` yields ``functools.partial(
        # MarkedFunction, instance)``, which ``unwrap_function`` crosses
        # (partial -> ``MarkedFunction.__wrapped__`` -> raw method) so the
        # annotation is read without executing the body, and mock inference
        # still calls the bound method when there is no annotation.
        # Builtins reload+instantiate fresh per call,
        # so inference state never reaches execution.
        if isinstance(underlying, type) and issubclass(underlying, CustomAgg):
            infer_instance = underlying()
            infer_pre = getattr(infer_instance, "pre") if has_pre else pre_func
            infer_agg = getattr(infer_instance, "agg")
            infer_post = getattr(infer_instance, "post") if has_post else post_func
        else:
            # Standalone builtin callable: inference on the raw function.
            infer_pre = pre_func
            infer_agg = underlying
            infer_post = post_func
    else:
        pre_func = custom_agg.pre
        post_func = custom_agg.post
        has_pre = not _is_identity_method(
            pre_func.__func__ if hasattr(pre_func, "__func__") else pre_func
        )
        has_post = not _is_identity_method(
            post_func.__func__ if hasattr(post_func, "__func__") else post_func
        )
        # Inference runs on a deep copy so a stateful agg does not carry
        # inference state into execution; builtins reload fresh per call.
        # The clone is local to inference only — the runtime projections
        # below bind to the *original* ``custom_agg.pre``/``.post`` so the
        # serialized execution instance is never touched by inference.
        infer_instance = copy.deepcopy(custom_agg)
        infer_pre = infer_instance.pre if has_pre else pre_func
        infer_agg = infer_instance.agg
        infer_post = infer_instance.post if has_post else post_func

    # --- Chained type inference: pre -> agg -> post ---
    # Uses the clone-bound hooks; runtime projections use the original hooks.
    if has_pre:
        pre_output = infer_output_dtypes(
            infer_pre, input_dtypes, input_frame_type=input_frame_type
        )
        pre_output_dtypes = pre_output[0] if pre_output else input_dtypes
    else:
        pre_output_dtypes = input_dtypes

    agg_output = infer_output_dtypes(
        infer_agg, pre_output_dtypes, input_frame_type=input_frame_type
    )
    agg_output_dtypes = agg_output[0] if agg_output else pre_output_dtypes

    if has_post:
        post_output = infer_output_dtypes(
            infer_post, agg_output_dtypes, input_frame_type=input_frame_type
        )
        post_output_dtypes = post_output[0] if post_output else agg_output_dtypes
    else:
        post_output_dtypes = agg_output_dtypes
    # Reject user-produced pre/agg/post output columns whose names collide
    # with a group key (framework-owned; silently discarded otherwise).  See
    # ``_reject_group_key_collisions`` for the contract and the dropped
    # lineage exemption.
    if group_key_cols:
        _reject_group_key_collisions(
            group_key_cols,
            [
                ("pre", pre_output_dtypes),
                ("agg", agg_output_dtypes),
                ("post", post_output_dtypes),
            ],
        )

    # --- Build pre-agg projection ---
    # Binds to the *original* hooks (pre_func), not infer_pre, so a
    # stateful pre/post has zero inference calls recorded on the instance
    # that actually executes.  The pre UDF binds to the *narrowed*
    # ``input_dtypes`` (selection-narrowed, group-key-excluded) so the
    # framework key and unselected columns don't leak into ``pre``.
    custom_pre_projection = []
    if has_pre:
        custom_pre_projection = _build_hook_projection(
            pre_func,
            input_dtypes,
            pre_output_dtypes,
            input_frame_type,
            group_key_cols,
            full_dtypes,
            custom_agg,
            "pre",
        )
    # --- Build agg descriptor (tree-reducible, NOT passthrough) ---
    intermediate_cols = list(pre_output_dtypes.index)
    agg_output_cols = list(agg_output_dtypes.index)

    agg_funcs = [
        {
            "raw_func_name": getattr(custom_agg, "__name__", None)
            or getattr(custom_agg, "func_name", "<custom>"),
            "map_func_name": "custom_agg",
            "agg_func_name": "custom_agg",
            "cols": intermediate_cols,
            "agg_output_cols": agg_output_cols,
            "output_key": "agg_0",
            "kwds": {},
            "custom_func": custom_agg,
            "agg_output_dtypes": agg_output_dtypes,
            "input_frame_type": input_frame_type,
        }
    ]

    # --- Build post-agg projection ---
    # Group keys must survive the post projection: pandas as_index=False
    # keeps group keys as regular output columns even when post renames
    # the agg output (e.g. agg=[key,total], post -> [final] yields
    # [key,final]).  The framework-owned key is passed through
    # unconditionally, overriding any same-named post output.
    custom_post_projection = []
    if has_post:
        custom_post_projection = _build_hook_projection(
            post_func,
            agg_output_dtypes,
            post_output_dtypes,
            input_frame_type,
            group_key_cols,
            full_dtypes,
            custom_agg,
            "post",
        )

    # --- Compute output metadata ---
    final_dtypes = post_output_dtypes

    out_dtypes_dict = {
        p.name: p.dtype for p in _key_pass_through(group_key_cols, full_dtypes)
    }
    for col in final_dtypes.index:
        out_dtypes_dict[col] = final_dtypes[col]
    out_dtypes = pd.Series(out_dtypes_dict)

    # When post is present, the agg op keeps agg-level names (post handles
    # final naming).  Otherwise use the final names directly.
    if has_post:
        output_column_names = group_key_cols + list(agg_output_dtypes.index)
    else:
        output_column_names = list(out_dtypes_dict.keys())

    out_shape = (np.nan if groupby_params else 1, len(out_dtypes))

    return CompileAggResult(
        agg_funcs=agg_funcs,
        agg_output_columns=agg_output_cols,
        output_column_names=output_column_names,
        agg_output_dtypes=agg_output_dtypes,
        out_dtypes=out_dtypes,
        out_shape=out_shape,
        projection_exprs=[],
        pre_agg_projection_exprs=[],
        custom_pre_projection=custom_pre_projection,
        custom_post_projection=custom_post_projection,
    )
