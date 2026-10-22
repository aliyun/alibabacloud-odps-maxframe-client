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

"""Unit tests for custom aggregation (CustomAgg)."""

import pandas as pd
import pytest

from maxframe.liteframe.core import LiteFrame
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.liteframe.operators.reduction import LiteFrameAgg
from maxframe.liteframe.reduction.compile import compile_agg
from maxframe.liteframe.reduction.custom_agg import CustomAgg
from maxframe.udf import (
    BuiltinFunction,
    BuiltinMethod,
    MarkedFunction,
    builtin_function,
    with_running_options,
)
from maxframe.utils import deserialize_serializable, serialize_serializable

# Module-level @builtin_function targets (need importable qualified names).


@builtin_function
def _builtin_single_agg(df):
    return df


@builtin_function
@with_running_options(cpu=7)
def _builtin_single_agg_decorated(df):
    # Standalone builtin callable with @with_running_options stacked below
    # @builtin_function: the outer BuiltinFunction is
    # opaque to discover_marked_functions, so the scheduling source must
    # resolve to the underlying MarkedFunction or cpu=7 is lost.
    return df


@builtin_function
class _BuiltinCustomAgg(CustomAgg):
    def pre(self, df):
        return df

    def agg(self, df):
        return df

    def post(self, df):
        return df


@builtin_function
class _BuiltinCustomAggWithPre(CustomAgg):
    def pre(self, df):
        return df * 2

    def agg(self, df):
        return df


@builtin_function
class _BuiltinAggOnly(CustomAgg):
    def agg(self, df):
        return df


@builtin_function
class _BuiltinAggDecorated(CustomAgg):
    @with_running_options(cpu=7)
    def agg(self, df) -> LiteFrame["total":int]:  # noqa: F821
        return df


@builtin_function
class _BuiltinAggDecoratedPre(CustomAgg):
    @with_running_options(cpu=3)
    def pre(self, df) -> LiteFrame["value":int]:  # noqa: F821
        return df[["value"]]

    def agg(self, df):
        return df


@builtin_function
class _BuiltinAggDecoratedPost(CustomAgg):
    def agg(self, df) -> LiteFrame["total":int]:  # noqa: F821
        return df

    @with_running_options(memory="2GiB")
    def post(self, df) -> LiteFrame["final":int]:  # noqa: F821
        return df


@builtin_function
class _BuiltinAggAnnotatedOnly(CustomAgg):
    @with_running_options(cpu=1)
    def agg(self, df) -> LiteFrame["total":int]:  # noqa: F821
        # Body rejects the mock input, so the annotation is the only
        # valid source for the output schema.
        raise RuntimeError("mock input rejected")


def _get_agg_op(result):
    op = result.data.op
    if isinstance(op, LiteFrameProjection):
        return result.data.inputs[0].op
    return op


def _lf(dtypes_dict):
    return LF({col: [1, 2, 3] for col in dtypes_dict}, dtypes_dict)


def test_custom_agg_class_basic():
    agg = CustomAgg()
    assert agg.name == "<custom>"
    assert agg.__name__ == "<custom>"
    assert agg.pre(42) == 42
    assert agg.post(42) == 42
    with pytest.raises(NotImplementedError):
        agg.agg(None)


def test_compile_single_function_agg():
    """Single callable -> agg-only, no pre/post projections."""
    lf = _lf({"a": int, "b": int})

    def my_agg(df):
        return df

    op = LiteFrameAgg(raw_func=my_agg, input_frame_type="pandas")
    compiled = compile_agg(op, lf)

    assert len(compiled.agg_funcs) == 1
    desc = compiled.agg_funcs[0]
    assert desc["map_func_name"] == "custom_agg"
    assert not desc.get("passthrough")
    assert desc["input_frame_type"] == "pandas"
    assert compiled.custom_pre_projection == []
    assert compiled.custom_post_projection == []


def test_compile_class_with_pre_agg_post():
    class MyAgg(CustomAgg):
        def pre(self, df):
            return df

        def agg(self, df):
            return df

        def post(self, df):
            return df

    lf = _lf({"a": int, "b": int})
    op = LiteFrameAgg(raw_func=MyAgg, input_frame_type="pandas")
    compiled = compile_agg(op, lf)

    desc = compiled.agg_funcs[0]
    assert not desc.get("passthrough")
    assert len(compiled.custom_pre_projection) > 0
    assert len(compiled.custom_post_projection) > 0


def test_compile_class_instance():
    class MyAgg(CustomAgg):
        def agg(self, df):
            return df

    lf = _lf({"a": int})
    op = LiteFrameAgg(raw_func=MyAgg(), input_frame_type="pandas")
    compiled = compile_agg(op, lf)
    assert isinstance(compiled.agg_funcs[0]["custom_func"], MyAgg)


def test_has_custom_code_plain_vs_builtin():
    """Plain callable -> True; @builtin_function -> False."""

    def my_agg(df):
        return df

    lf = _lf({"a": int})
    op = LiteFrameAgg(raw_func=my_agg, input_frame_type="pandas")
    op.agg_funcs = compile_agg(op, lf).agg_funcs
    assert op.has_custom_code() is True

    op2 = LiteFrameAgg(raw_func=_builtin_single_agg, input_frame_type="pandas")
    op2.agg_funcs = compile_agg(op2, lf).agg_funcs
    assert op2.has_custom_code() is False

    op3 = LiteFrameAgg(raw_func=_BuiltinCustomAgg, input_frame_type="pandas")
    op3.agg_funcs = compile_agg(op3, lf).agg_funcs
    assert op3.has_custom_code() is False


def test_builtin_method_creation_and_call():
    """Accessing methods on a @builtin_function class returns BuiltinMethod."""
    assert isinstance(_BuiltinCustomAgg, BuiltinFunction)
    for method_name in ("pre", "agg", "post"):
        m = getattr(_BuiltinCustomAgg, method_name)
        assert isinstance(m, BuiltinMethod)
        assert isinstance(m, BuiltinFunction)
        assert m.method_name == method_name

    # __call__ instantiates the class and invokes the bound method
    assert _BuiltinCustomAggWithPre.pre(42) == 84


def test_builtin_method_in_compile():
    """@builtin_function CustomAgg class compiles with BuiltinMethod projections."""
    lf = _lf({"a": int})
    op = LiteFrameAgg(raw_func=_BuiltinCustomAgg, input_frame_type="pandas")
    compiled = compile_agg(op, lf)

    assert len(compiled.custom_pre_projection) > 0
    assert len(compiled.custom_post_projection) > 0
    assert isinstance(compiled.agg_funcs[0]["custom_func"], BuiltinFunction)


def test_builtin_agg_only_no_projections():
    """A @builtin_function CustomAgg that does NOT override pre/post must not
    produce identity pre/post projections (the identity check should detect
    the inherited default method)."""
    lf = _lf({"a": int})
    op = LiteFrameAgg(raw_func=_BuiltinAggOnly, input_frame_type="pandas")
    compiled = compile_agg(op, lf)
    assert compiled.custom_pre_projection == []
    assert compiled.custom_post_projection == []
    assert isinstance(compiled.agg_funcs[0]["custom_func"], BuiltinFunction)


@pytest.mark.parametrize("frame_type", ["pandas", "polars"])
def test_input_frame_type_threading(frame_type):
    lf = _lf({"a": int})

    def my_agg(df):
        return df

    op = LiteFrameAgg(raw_func=my_agg, input_frame_type=frame_type)
    compiled = compile_agg(op, lf)
    assert compiled.agg_funcs[0]["input_frame_type"] == frame_type


def test_call_single_function_lowers_to_agg():
    lf = _lf({"a": int})

    def my_agg(df):
        return df

    result = lf.agg(my_agg)
    assert isinstance(result, LiteFrame)
    assert isinstance(_get_agg_op(result), LiteFrameAgg)


def test_call_class_with_pre_post_lowers_projections():
    class MyAgg(CustomAgg):
        def pre(self, df):
            return df

        def agg(self, df):
            return df

        def post(self, df):
            return df

    lf = _lf({"a": int})
    result = lf.agg(MyAgg)
    assert isinstance(result.data.op, LiteFrameProjection)
    assert isinstance(result.data.inputs[0].op, LiteFrameAgg)


def test_grouped_pre_projection_preserves_group_key():
    """A grouped pre that drops the group key must still preserve it as a
    pass-through projection so the executor's drop_nulls/group_by succeed."""

    class DropKeyPre(CustomAgg):
        def pre(self, df):
            return df[["value"]]

        def agg(self, df):
            return df

    lf = _lf({"key": int, "value": int})
    op = LiteFrameAgg(
        raw_func=DropKeyPre,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    compiled = compile_agg(op, lf)

    proj_names = [p.name for p in compiled.custom_pre_projection]
    # Group key "key" is prepended as a pass-through even though pre drops it.
    assert "key" in proj_names
    # The user-visible pre output ("value") is still present.
    assert "value" in proj_names


def test_custom_agg_with_post_uses_intermediate_node_schema():
    """With a custom post, the agg node carries the intermediate schema
    (group keys + agg output), not the final post output."""

    class AggWithPost(CustomAgg):
        def agg(self, df):
            return df

        def post(self, df):
            return df

    lf = _lf({"key": int, "value": int})
    result = lf.agg(AggWithPost)
    assert isinstance(result.data.op, LiteFrameProjection)
    agg_node = result.data.inputs[0]
    node_cols = list(agg_node._physical_dtypes.index)
    assert "key" in node_cols
    assert "value" in node_cols


def test_grouped_post_projection_preserves_group_key():
    """A grouped post that renames the agg output must still preserve the
    group key as a pass-through projection, matching pandas as_index=False
    semantics where the group key remains a regular output column."""

    class RenamePost(CustomAgg):
        def agg(self, df):
            return df

        def post(self, df):
            return df[["value"]].rename(columns={"value": "final"})

    lf = _lf({"key": int, "value": int})
    op = LiteFrameAgg(
        raw_func=RenamePost,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    compiled = compile_agg(op, lf)

    proj_names = [p.name for p in compiled.custom_post_projection]
    # Group key "key" is prepended as a pass-through even though post drops it.
    assert "key" in proj_names
    # The user-visible post output ("final") is still present.
    assert "final" in proj_names
    # Final output metadata keeps the group key alongside the post output.
    assert "key" in compiled.out_dtypes.index
    assert "final" in compiled.out_dtypes.index


def test_inference_does_not_bind_projection_to_clone():
    """Regression: runtime pre/post projections must bind to the original
    CustomAgg hooks, not the inference clone.  A stateful pre/post that
    increments a counter during inference must not record those calls on
    the instance that actually executes.

    Prior fix only protected ``agg_funcs[0]['custom_func']``; the
    projection-bound hooks were still the clone, so a stateful pre/post
    carried inference state into execution.
    """

    class StatefulPrePost(CustomAgg):
        def __init__(self):
            self.pre_calls = 0
            self.post_calls = 0

        def pre(self, df):
            self.pre_calls += 1
            return df

        def agg(self, df):
            return df

        def post(self, df):
            self.post_calls += 1
            return df

    lf = _lf({"key": int, "value": int})
    agg = StatefulPrePost()
    op = LiteFrameAgg(
        raw_func=agg,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    compiled = compile_agg(op, lf)

    # The original instance (stored for execution AND bound to the runtime
    # projections) was never invoked during inference.
    assert agg.pre_calls == 0
    assert agg.post_calls == 0
    assert compiled.agg_funcs[0]["custom_func"] is agg


def test_grouped_pre_returning_same_named_key_rejected_at_compile():
    """Regression: a grouped ``pre`` that returns a
    same-named key column (e.g. ``return df.assign(key=0)[["key", "value"]]``)
    must be rejected at compile time, not silently overwritten.  Group keys
    are framework-owned and excluded from the UDF input, so a same-named user
    output can never be a pass-through of the original key — it is a
    newly-created column that the executor would silently discard when it
    re-materializes the framework key (``res.with_columns(key_cols)``).
    Rejecting at compile surfaces the data loss instead of hiding it.
    """

    class SameNameKeyPre(CustomAgg):
        def pre(self, df):
            # Mutate the key to a constant; under the old silent-overwrite
            # contract this collapsed every group to key=0.  Under the new
            # contract it must fail at compile instead.
            return df.assign(key=0)[["key", "value"]]

        def agg(self, df):
            return df

    lf = _lf({"key": int, "value": int})
    op = LiteFrameAgg(
        raw_func=SameNameKeyPre,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    with pytest.raises(ValueError, match=r"pre output column\(s\) collide"):
        compile_agg(op, lf)


def test_custom_agg_descriptor_carries_agg_output_cols():
    """Regression: the agg descriptor must carry ``agg_output_cols`` (the
    agg output column names) distinct from ``cols`` (the pre output / map
    input), so the executor can select the correct columns at the combine
    and agg (final) stages where the input is the renamed agg output.
    """

    class RenameAgg(CustomAgg):
        def agg(self, df):
            return df[["value"]].rename(columns={"value": "total"})

    lf = _lf({"key": int, "value": int})
    op = LiteFrameAgg(
        raw_func=RenameAgg,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    compiled = compile_agg(op, lf)
    desc = compiled.agg_funcs[0]

    # ``cols`` = pre output (map-stage input); ``agg_output_cols`` = agg
    # output (combine/agg-stage input).  They differ for a renaming agg.
    assert desc["cols"] == ["value"]
    assert desc["agg_output_cols"] == ["total"]


def test_grouped_pre_udf_binds_to_selection_not_full_frame():
    """Regression: ``groupby("key")[["value"]].agg(CapturePre())`` must
    build the ``pre`` UDF from the *selection-narrowed* schema, not the
    full input frame.  The UDF operands double as the column set the
    executor selects before calling ``pre`` (see
    ``build_udf_projections``); binding to the full frame leaked the
    framework key and explicitly-excluded columns into ``pre`` before
    the map-stage executor narrowed the agg input.

    Minimal repro: ``CapturePre.pre`` records
    ``list(df.columns)`` and returns ``df[["value"]]``.  Compilation
    inferred ``["value"]``, but the runtime ``transform_chunk`` pre
    projection recorded ``["key", "value", "extra"]``.
    """

    class CapturePre(CustomAgg):
        def pre(self, df):
            return df[["value"]]

        def agg(self, df):
            return df

    lf = _lf({"key": int, "value": int, "extra": int})
    op = LiteFrameAgg(
        raw_func=CapturePre,
        groupby_params={
            "by": "key",
            "sort": False,
            "dropna": True,
            "selection": ["value"],
        },
        input_frame_type="pandas",
    )
    compiled = compile_agg(op, lf)

    # The pre UDF operands must reference only the selected column
    # ("value"), not the framework key or the excluded "extra".
    user_pre = [p for p in compiled.custom_pre_projection if p.name not in ("key",)]
    assert user_pre, "expected a non-key pre projection"
    udf_expr = user_pre[0].expr.multi_expr
    operand_names = [op.name for op in udf_expr.operands]
    assert operand_names == [
        "value"
    ], f"pre UDF should bind to selection ['value'], got {operand_names}"


def test_decorated_pre_post_propagate_scheduling_hint():
    """Regression: a ``CustomAgg`` whose ``pre``/
    ``post`` are decorated with ``@with_running_options`` must propagate
    the running options as a ``SchedulingHint`` onto the compiled
    ``LiteFrameUDFExpr``.  The projection tiler reads the expression hint
    for color-grouping and split-op resource allocation; dropping it lets
    the decorated ``pre``/``post`` run without the requested CPU/memory.

    Root cause was two-fold:
    (1) ``_build_udf_expr`` (the custom-agg-specific replacement for
        ``build_udf_projections``) omitted the ``scheduling_hint=`` arg.
    (2) ``@with_running_options`` on an instance method stored an
        *unbound* ``MarkedFunction`` on the class (``MarkedFunction`` had
        no ``__get__``), so ``custom_agg.pre`` dropped ``self`` and the
        method could not run at all.  The fix gives ``MarkedFunction`` a
        ``__get__`` that binds the instance via ``functools.partial``.
    """

    class DecoPrePost(CustomAgg):
        @with_running_options(cpu=2, memory="4GiB")
        def pre(self, df):
            return df[["value"]]

        def agg(self, df):
            return df

        @with_running_options(cpu=3)
        def post(self, df):
            return df

    lf = _lf({"key": int, "value": int, "extra": int})
    op = LiteFrameAgg(
        raw_func=DecoPrePost,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    compiled = compile_agg(op, lf)

    # Pre: first non-key projection carries the decorated hint.
    pre_user = [p for p in compiled.custom_pre_projection if p.name != "key"]
    assert pre_user, "expected a non-key pre projection"
    pre_hint = pre_user[0].expr.multi_expr.scheduling_hint
    assert pre_hint is not None, "pre scheduling_hint dropped"
    assert pre_hint.expect_resources == {"cpu": 2, "memory": "4GiB"}

    # Post: first non-key projection carries the decorated hint.
    post_user = [p for p in compiled.custom_post_projection if p.name != "key"]
    assert post_user, "expected a non-key post projection"
    post_hint = post_user[0].expr.multi_expr.scheduling_hint
    assert post_hint is not None, "post scheduling_hint dropped"
    assert post_hint.expect_resources == {"cpu": 3}


def test_undecorated_pre_post_have_none_scheduling_hint():
    """Complement to the decorated test: a plain (undecorated) ``pre``/
    ``post`` compiles to ``scheduling_hint is None`` — the helper returns
    ``None`` when no running options are present, and the compile-time
    identity assert must accept that without raising.
    """

    class PlainAgg(CustomAgg):
        def pre(self, df):
            return df[["value"]]

        def agg(self, df):
            return df

    lf = _lf({"key": int, "value": int})
    op = LiteFrameAgg(
        raw_func=PlainAgg,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    compiled = compile_agg(op, lf)
    pre_user = [p for p in compiled.custom_pre_projection if p.name != "key"]
    assert pre_user, "expected a non-key pre projection"
    assert pre_user[0].expr.multi_expr.scheduling_hint is None


def test_builtin_class_decorated_pre_post_propagate_scheduling_hint():
    """Regression: a ``@builtin_function`` ``CustomAgg``
    class whose ``pre``/``post`` are decorated with ``@with_running_options``
    must propagate the running options onto the pre/post ``LiteFrameUDFExpr``.
    The builtin-class ``pre_func``/``post_func`` are ``BuiltinMethod`` objects,
    which ``discover_marked_functions`` cannot traverse (they reload+rebind
    the class on every access, hiding the underlying decorated
    ``MarkedFunction``), so the hint was silently dropped — the same
    regression class as the standalone-builtin-agg shape.  The fix resolves
    the hint from the raw class attribute (``underlying.__dict__["pre"]``).
    """
    lf = _lf({"key": int, "value": int})
    op = LiteFrameAgg(
        raw_func=_BuiltinAggDecoratedPre,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    compiled = compile_agg(op, lf)
    pre_user = [p for p in compiled.custom_pre_projection if p.name != "key"]
    assert pre_user, "expected a non-key pre projection"
    pre_hint = pre_user[0].expr.multi_expr.scheduling_hint
    assert pre_hint is not None, "builtin-class pre scheduling_hint dropped"
    assert pre_hint.expect_resources.get("cpu") == 3, pre_hint.expect_resources

    op2 = LiteFrameAgg(
        raw_func=_BuiltinAggDecoratedPost,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    compiled2 = compile_agg(op2, lf)
    post_user = [p for p in compiled2.custom_post_projection if p.name != "key"]
    assert post_user, "expected a non-key post projection"
    post_hint = post_user[0].expr.multi_expr.scheduling_hint
    assert post_hint is not None, "builtin-class post scheduling_hint dropped"
    assert (
        post_hint.expect_resources.get("memory") == "2GiB"
    ), post_hint.expect_resources


def test_polars_explicit_empty_selection_rejected_at_compile():
    """Regression: polars + explicit-empty selection
    is rejected at compile time (polars can't represent a zero-column frame
    with nonzero height)."""

    class _Agg(CustomAgg):
        def agg(self, df):
            return df

    lf = _lf({"key": int, "value": int, "extra": int})
    op = LiteFrameAgg(
        raw_func=_Agg,
        groupby_params={
            "by": "key",
            "sort": False,
            "dropna": True,
            "selection": [],
        },
        input_frame_type="polars",
    )
    with pytest.raises(ValueError, match="explicit-empty selection"):
        compile_agg(op, lf)


def test_decorated_hook_annotation_survives_partial_binding():
    """Regression: a decorated ``CustomAgg`` hook whose body cannot be
    mock-inferred must still resolve its return annotation.  A bound
    method arrives as ``functools.partial(MarkedFunction, instance)``;
    annotation lookup must descend through it to the raw method.
    """

    class DecoPreAnn(CustomAgg):
        @with_running_options(cpu=1)
        def pre(self, df) -> LiteFrame["x":int]:  # noqa: F821
            # Body rejects the mock input, so the annotation is the only
            # valid source for the output schema.
            raise RuntimeError("mock input rejected")

        @with_running_options(cpu=2)
        def agg(self, df) -> LiteFrame["total":int]:  # noqa: F821
            # Body rejects the mock input too; annotation must drive inference.
            raise RuntimeError("mock input rejected")

    lf = _lf({"key": int, "value": int})
    op = LiteFrameAgg(
        raw_func=DecoPreAnn,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    # Must not raise: annotation drives inference despite the partial binding
    # and the un-inferable body.
    compiled = compile_agg(op, lf)
    pre_user = [p for p in compiled.custom_pre_projection if p.name != "key"]
    assert pre_user, "expected a non-key pre projection"
    assert pre_user[0].name == "x"


def test_agg_output_colliding_with_group_key_rejected():
    """Regression: an ``agg`` output colliding with a group-key name is
    silently discarded; reject it with a ``ValueError`` naming the column.
    """

    class CollideAgg(CustomAgg):
        def agg(self, df) -> LiteFrame["key":int]:  # noqa: F821
            return df

    lf = _lf({"key": int, "value": int})
    op = LiteFrameAgg(
        raw_func=CollideAgg,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    with pytest.raises(ValueError, match=r"\['key'\]"):
        compile_agg(op, lf)


def test_post_output_colliding_with_group_key_rejected():
    """Regression: a ``post`` output colliding with a group-key name is
    silently discarded; reject it at compile.
    """

    class CollidePost(CustomAgg):
        def agg(self, df) -> LiteFrame["total":int]:  # noqa: F821
            return df

        def post(self, df) -> LiteFrame["key":int]:  # noqa: F821
            return df

    lf = _lf({"key": int, "value": int})
    op = LiteFrameAgg(
        raw_func=CollidePost,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    with pytest.raises(ValueError, match=r"\['key'\]"):
        compile_agg(op, lf)


def test_chained_pre_agg_group_key_collision_rejected_at_compile():
    """Regression: the lineage exemption used to let a
    chained ``pre -> agg`` re-declaring a group-key name through silently.
    Repro: ``pre`` declares ``LiteFrame["key": int, "value": int]`` and
    returns it; ``agg`` declares ``LiteFrame["key": int]`` and returns
    ``key=999``.  Under the old exemption, ``key`` was already in
    ``pre_output_dtypes`` so the agg collision was exempted, compilation
    succeeded, and execution returned only the framework keys ``[1, 2]`` —
    the user's ``999`` was discarded.  Now every boundary is checked
    unconditionally, so the ``pre`` output (which contains ``key``) is
    rejected first.
    """

    class PreDeclaresKey(CustomAgg):
        def pre(self, df) -> LiteFrame["key":int, "value":int]:  # noqa: F821
            return df.assign(key=0)[["key", "value"]]

        def agg(self, df) -> LiteFrame["key":int]:  # noqa: F821
            return pd.DataFrame({"key": [999]})

    lf = _lf({"key": int, "value": int})
    op = LiteFrameAgg(
        raw_func=PreDeclaresKey,
        groupby_params={"by": "key", "sort": False, "dropna": True},
        input_frame_type="pandas",
    )
    with pytest.raises(ValueError, match=r"pre output column\(s\) collide"):
        compile_agg(op, lf)


def test_decorated_agg_propagates_resources_to_liteframe_agg():
    """Regression: ``@with_running_options`` on ``CustomAgg.agg`` must
    propagate onto the ``LiteFrameAgg`` operator node (mirrors
    ``LiteFrameFunctional``); the tiler colors/splits on that node for the
    map/combine/final stages.
    """

    class DecoAgg(CustomAgg):
        @with_running_options(cpu=7)
        def agg(self, df) -> LiteFrame["total":int]:  # noqa: F821
            return df

    lf = _lf({"a": int})
    result = lf.agg(DecoAgg(), input_frame_type="pandas")
    agg_op = result.data.op
    assert isinstance(agg_op, LiteFrameAgg)
    assert (
        agg_op.expect_resources.get("cpu") == 7
    ), f"agg cpu not propagated: {agg_op.expect_resources}"


def test_standalone_builtin_callable_compiles():
    """Regression: a standalone
    ``@builtin_function`` callable is a valid custom-agg input.  Reading
    ``custom_func.agg`` raised ``AttributeError`` because a plain function
    has no ``.agg``; the scheduling source must resolve to the callable
    itself.
    """
    lf = _lf({"a": int})
    result = lf.agg(_builtin_single_agg, input_frame_type="pandas")
    agg_op = _get_agg_op(result)
    assert isinstance(agg_op, LiteFrameAgg)


def test_builtin_class_decorated_agg_preserves_resources():
    """Regression: a ``@builtin_function``
    ``CustomAgg`` class whose ``agg`` is decorated with
    ``@with_running_options(cpu=7)`` must propagate ``cpu=7`` onto
    ``LiteFrameAgg``.  The naive ``custom_func.agg`` resolves to a
    ``BuiltinMethod`` whose ``func`` instantiates a fresh instance and
    returns a ``partial`` around the ``MarkedFunction`` that
    ``discover_marked_functions`` cannot reach through
    ``BuiltinMethod``, so the operator fell back to default ``cpu=1``.
    """

    lf = _lf({"a": int})
    result = lf.agg(_BuiltinAggDecorated, input_frame_type="pandas")

    agg_op = _get_agg_op(result)
    assert isinstance(agg_op, LiteFrameAgg)
    assert (
        agg_op.expect_resources.get("cpu") == 7
    ), f"builtin-class agg cpu not propagated: {agg_op.expect_resources}"


def test_builtin_class_annotated_agg_infers_without_executing_body():
    """Regression: a ``@builtin_function`` ``CustomAgg``
    class whose ``agg`` is decorated with ``@with_running_options`` and
    annotated ``-> LiteFrame["total": int]`` must infer the output dtype from
    the annotation *without* executing the body.  ``custom_func.agg`` returns
    a ``BuiltinMethod`` that ``_get_return_annotation`` /
    ``unwrap_function`` cannot descend into, so inference fell back to mock
    execution (which the body rejects) and raised ``TypeError: Cannot infer
    output schema``.  Resolving the bound instance method at the
    callable-normalization boundary lets ``unwrap_function`` cross
    ``partial`` -> ``MarkedFunction.__wrapped__`` -> the annotated method.
    """
    lf = _lf({"a": int})
    op = LiteFrameAgg(raw_func=_BuiltinAggAnnotatedOnly, input_frame_type="pandas")
    compiled = compile_agg(op, lf)
    agg_dt = compiled.agg_output_dtypes
    assert "total" in list(
        agg_dt.index
    ), f"annotation not used for inference: {list(agg_dt.index)}"
    assert "int64" in str(
        agg_dt["total"]
    ), f"total dtype not int64 from annotation: {agg_dt['total']}"


def test_standalone_builtin_callable_decorated_preserves_resources():
    """Regression: a standalone ``@builtin_function``
    callable with ``@with_running_options(cpu=7)`` stacked *below*
    ``@builtin_function`` must propagate ``cpu=7`` onto ``LiteFrameAgg``.
    The outer ``BuiltinFunction`` is opaque to
    ``discover_marked_functions`` (it only traverses plain functions via
    ``__code__``/``__closure__``/``__globals__``, never descending into a
    ``BuiltinFunction``'s ``_func``), so resolving the source to the
    ``BuiltinFunction`` itself loses the hint and the operator falls back to
    default ``cpu=1``.  The resolver must return the underlying
    ``MarkedFunction`` (``custom_agg.func``).
    """
    lf = _lf({"a": int})
    result = lf.agg(_builtin_single_agg_decorated, input_frame_type="pandas")
    agg_op = _get_agg_op(result)
    assert isinstance(agg_op, LiteFrameAgg)
    assert (
        agg_op.expect_resources.get("cpu") == 7
    ), f"standalone decorated agg cpu not propagated: {agg_op.expect_resources}"


def test_custom_agg_descriptor_has_no_persisted_scheduling_source():
    """Regression: the agg descriptor must not persist
    the scheduling source callable (``agg_scheduling_func``).  Persisting a
    raw ``MarkedFunction`` would leak a Python function body into serialized
    operator state — ``BuiltinFunction`` promises *not* to serialize the
    body, and ``has_custom_code()`` must stay ``False`` for builtin aggs.
    Scheduling extraction is client-local: hints are copied onto the
    operator at call time and the source is discarded.
    """
    lf = _lf({"a": int})
    for raw in (
        _builtin_single_agg,
        _builtin_single_agg_decorated,
        _BuiltinAggDecorated,
    ):
        agg_op = _get_agg_op(lf.agg(raw, input_frame_type="pandas"))
        desc = agg_op.agg_funcs[0]
        assert (
            "agg_scheduling_func" not in desc
        ), f"scheduling source persisted in descriptor for {raw!r}: {desc.keys()}"
        assert isinstance(desc["custom_func"], BuiltinFunction), (
            f"custom_func is not a BuiltinFunction for {raw!r}: "
            f"{type(desc['custom_func'])}"
        )
        leaked = [k for k, v in desc.items() if isinstance(v, MarkedFunction)]
        assert (
            not leaked
        ), f"MarkedFunction leaked into descriptor for {raw!r}: {leaked}"
        assert (
            agg_op.has_custom_code() is False
        ), f"has_custom_code should be False for builtin {raw!r}"


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(_builtin_single_agg, id="plain"),
        pytest.param(_builtin_single_agg_decorated, id="decorated"),
        pytest.param(_BuiltinCustomAgg, id="builtin-class"),
    ],
)
def test_builtin_agg_survives_serialization_roundtrip(raw):
    """Regression: every ``@builtin_function`` agg shape must reload
    ``.func`` after operator serialization without recursion.

    ``MarkedFunction.__module__`` resolved to the class's module
    (``maxframe.udf``), so ``BuiltinFunction.__init__`` registered the
    builtin under the wrong module name; worker reload then recursed through
    ``load_member`` -> ``.func`` -> ``load_member`` infinitely.
    """
    lf = _lf({"a": int})
    result = lf.agg(raw, input_frame_type="pandas")
    agg_op = _get_agg_op(result)

    restored = deserialize_serializable(serialize_serializable(agg_op))
    custom_func = restored.agg_funcs[0]["custom_func"]
    assert isinstance(custom_func, BuiltinFunction)
    # func_name must point at the test module, not maxframe.udf.
    assert custom_func.func_name.startswith(
        "maxframe.liteframe.reduction.tests.test_custom_agg#"
    )

    # Worker-side reload: must not raise RecursionError.
    func = custom_func.func
    assert func is not None
