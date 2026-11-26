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

import pickle

import pandas as pd
import pytest

from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.functional.apply import apply
from maxframe.liteframe.groupby.groupby import groupby
from maxframe.liteframe.merge.concat import concat
from maxframe.liteframe.misc.rechunk import rechunk
from maxframe.utils import tokenize

# --- FrameMetadata default / key behavior -------------------------------


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"alignment_id": 12345}, 12345),
        ({"alignment_id": None}, None),
    ],
    ids=["explicit-id", "explicit-none"],
)
def test_frame_metadata_explicit_alignment_id(kwargs, expected):
    assert FrameMetadata(**kwargs).alignment_id == expected


def test_frame_metadata_default_random_id():
    """Two fresh FrameMetadata objects get distinct non-None random ids,
    and so do two independent reads of the same data."""
    a, b = FrameMetadata(), FrameMetadata()
    assert a.alignment_id is not None
    assert a.alignment_id != b.alignment_id

    lf_a = from_local_df(pd.DataFrame({"a": [1, 2]}))
    lf_b = from_local_df(pd.DataFrame({"a": [1, 2]}))
    assert lf_a.frame_metadata.alignment_id != lf_b.frame_metadata.alignment_id


def test_frame_metadata_key_excludes_alignment_id():
    a, b = FrameMetadata(), FrameMetadata()
    assert tokenize(a) == tokenize(b)


# --- Operator propagation ---------------------------------------------


@pytest.fixture
def src():
    return from_local_df(pd.DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]}))


def test_datasource_has_alignment_id(src):
    assert src.frame_metadata.alignment_id is not None


@pytest.mark.parametrize(
    "op_fn, expect_inherit",
    [
        (lambda s: s[["a"]], True),
        (lambda s: s["a"] + 1, True),
        (lambda s: s[["a"]][["a"]], True),
        (lambda s: s.rank(method="min"), True),
        (lambda s: s.sort(by="a"), False),
        (lambda s: s[s["a"] > 1], False),
        # Window transforms: placeback=True (default) inherits,
        # placeback=False gets a fresh id.  One representative per kind
        # is sufficient -- all use the same _inherit_alignment_id.
        (lambda s: s.shift(1), True),
        (lambda s: s.shift(1, placeback=False), False),
        (lambda s: s.rolling(2).sum(), True),
        (lambda s: s.ffill(), True),
    ],
    ids=[
        "projection",
        "arithmetic",
        "chained-projection",
        "rank",
        "sort",
        "filter",
        "shift",
        "shift-no-pb",
        "rolling",
        "ffill",
    ],
)
def test_operator_alignment_id_inheritance(src, op_fn, expect_inherit):
    result_id = op_fn(src).frame_metadata.alignment_id
    src_id = src.frame_metadata.alignment_id
    assert (result_id == src_id) if expect_inherit else (result_id != src_id)


@pytest.fixture
def gb_src():
    return from_local_df(
        pd.DataFrame({"k": ["a", "a", "b", "b"], "v": [1.0, 2.0, 3.0, 4.0]})
    )


@pytest.mark.parametrize(
    "method, args, kwargs, expect_inherit",
    [
        ("shift", (1,), {}, True),
        ("diff", (1,), {}, True),
        ("ffill", (), {}, True),
        ("cumsum", (), {}, True),
        ("cumcount", (), {}, True),
        ("rank", (), {}, True),
        ("shift", (1,), {"placeback": False}, False),
    ],
    ids=["shift", "diff", "ffill", "cumsum", "cumcount", "rank", "shift-no-pb"],
)
def test_grouped_window_alignment_id(gb_src, method, args, kwargs, expect_inherit):
    result = getattr(gb_src.groupby("k"), method)(*args, **kwargs)
    src_id = gb_src.frame_metadata.alignment_id
    result_id = result.frame_metadata.alignment_id
    assert (result_id == src_id) if expect_inherit else (result_id != src_id)


# --- Functional transform / apply / duplicated -------------------------
#
# NOTE: This file is the plan-time metadata/plumbing oracle — all cases run
# on the single-chunk ``src`` fixture and only assert the
# ``_inherit_alignment_id`` plan-time decision.  The layout claims
# (multi-chunk nan-row-splits, multi-chunk skip_infer=True) are asserted in
# the DPE execution tests in
# ``dpe/.../tilers/liteframe/functional/tests/test_functional_execution.py``.
#
# transform(axis=0) inherits by default; placeback=False drops the id
# (axis=0 re-split is skipped -> single chunk, fresh id).  transform(axis=1)
# / transform_chunk are 1:1 projections; placeback=False is metadata-only.
# apply(axis=1) is always 1:1 (no placeback flag) -> always inherits.
# apply(axis=0) may change the row count -> never inherits.  duplicated()
# mask is 1:1 row-preserving -> inherits; drop_duplicates removes rows ->
# fresh id.
@pytest.mark.parametrize(
    "op_fn, expect_inherit",
    [
        (lambda s: s.transform(lambda c: c * 2), True),
        (lambda s: s.transform(lambda c: c * 2, placeback=False), False),
        (
            lambda s: s.transform(
                lambda r: {"ab": r["a"] + r["b"]}, axis=1, dtypes={"ab": int}
            ),
            True,
        ),
        (
            lambda s: s.transform(
                lambda r: {"ab": r["a"] + r["b"]},
                axis=1,
                dtypes={"ab": int},
                placeback=False,
            ),
            False,
        ),
        (lambda s: s.transform_chunk(lambda df: df * 2), True),
        (lambda s: s.transform_chunk(lambda df: df * 2, placeback=False), False),
        (
            lambda s: s.apply(
                lambda r: {"ab": r["a"] + r["b"]}, axis=1, dtypes={"ab": int}
            ),
            True,
        ),
        (lambda s: s.apply(lambda c: c.sum()), False),
        (lambda s: s.duplicated(), True),
        (lambda s: s.drop_duplicates(), False),
        (lambda s: groupby(s, by="a").transform(lambda g: g["b"] + 1), True),
        (
            lambda s: groupby(s, by="a").transform(
                lambda g: g["b"] + 1, placeback=False
            ),
            False,
        ),
        (lambda s: groupby(s, by="a").transform_chunk(lambda df: df * 2), True),
        (
            lambda s: groupby(s, by="a").transform_chunk(
                lambda df: df * 2, placeback=False
            ),
            False,
        ),
    ],
    ids=[
        "transform-axis0",
        "transform-axis0-no-pb",
        "transform-axis1",
        "transform-axis1-no-pb",
        "transform-chunk",
        "transform-chunk-no-pb",
        "apply-axis1",
        "apply-axis0",
        "duplicated",
        "drop-duplicates",
        "groupby-transform",
        "groupby-transform-no-pb",
        "groupby-transform-chunk",
        "groupby-transform-chunk-no-pb",
    ],
)
def test_functional_alignment_id_inheritance(src, op_fn, expect_inherit):
    result_id = op_fn(src).frame_metadata.alignment_id
    src_id = src.frame_metadata.alignment_id
    assert (result_id == src_id) if expect_inherit else (result_id != src_id)


def test_two_projection_branches_share_id(src):
    a, b = src[["a"]], src[["b"]]
    assert a.frame_metadata.alignment_id == src.frame_metadata.alignment_id
    assert b.frame_metadata.alignment_id == src.frame_metadata.alignment_id


# --- Concat alignment -------------------------------------------------


@pytest.mark.parametrize(
    "axis, aligned, expect_equal_src",
    [
        (1, True, True),
        (1, False, False),
        (0, True, False),
    ],
    ids=["axis1-aligned", "axis1-unaligned", "axis0-fresh"],
)
def test_concat_alignment_id(src, axis, aligned, expect_equal_src):
    if aligned:
        frames = [src[["a"]], src[["b"]]]
    else:
        a = from_local_df(pd.DataFrame({"a": [1, 2]}))
        b = from_local_df(pd.DataFrame({"x": [3, 4]}))
        frames = [a[["a"]], b[["x"]]]
    c = concat(frames, axis=axis)
    if expect_equal_src:
        assert c.frame_metadata.alignment_id == src.frame_metadata.alignment_id
    else:
        assert c.frame_metadata.alignment_id != src.frame_metadata.alignment_id


# --- Serialization -----------------------------------------------------


def test_pickle_roundtrip_preserves_alignment_id(src):
    md = src.frame_metadata
    md2 = pickle.loads(pickle.dumps(md))
    assert md2.alignment_id == md.alignment_id


# --- Rechunk & multi-output functional ---------------------------------


def test_rechunk_gets_fresh_alignment_id(src):
    """Rechunk repartitions rows, so the output must not inherit the
    source's alignment_id."""
    rechunked = rechunk(src, 2)
    assert rechunked.frame_metadata.alignment_id != src.frame_metadata.alignment_id


def test_multi_output_functional_has_distinct_metadata(src):
    """Multi-output build_functional_output must give each output its own
    FrameMetadata instance (and therefore a distinct alignment_id)."""
    results = apply(
        src, lambda row: (row["a"], row["b"]), axis=1, dtypes=[{"x": int}, {"y": int}]
    )
    assert len(results) == 2
    assert results[0].frame_metadata is not results[1].frame_metadata
    assert (
        results[0].frame_metadata.alignment_id != results[1].frame_metadata.alignment_id
    )


def test_setitem_guard_raises_on_unaligned_same_name(src):
    """Assigning an unaligned LiteFrame column to the same name must raise
    rather than silently bind key to itself (discarding the UDF result).
    The setitem guard (setitem.py:194-204) catches this case."""
    import pytest

    other = from_local_df(pd.DataFrame({"a": [10, 20]}))
    with pytest.raises(ValueError, match="different alignment_id"):
        src["a"] = other
