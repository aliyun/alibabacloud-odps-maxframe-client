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

import pandas as pd
import pyarrow as pa
import pytest

from maxframe.liteframe.core import LiteFrame
from maxframe.liteframe.expressions import (
    LiteFrameAnd,
    LiteFrameGe,
    LiteFrameGt,
    LiteFrameLe,
    LiteFrameLt,
)
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.utils import wrap_arrow_dtype

from .conftest import _eval_projection, _inner_expr

_BOOL = wrap_arrow_dtype(pa.bool_())


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inclusive,left_cls,right_cls",
    [
        ("both", LiteFrameGe, LiteFrameLe),
        ("left", LiteFrameGe, LiteFrameLt),
        ("right", LiteFrameGt, LiteFrameLe),
        ("neither", LiteFrameGt, LiteFrameLt),
    ],
    ids=["both", "left", "right", "neither"],
)
def test_between_structure(inclusive, left_cls, right_cls):
    lf = LF({"a": [1, 2, 3, 4]})
    result = lf.between(2, 3, inclusive=inclusive)
    assert isinstance(result, LiteFrame)
    expr = _inner_expr(result)
    assert isinstance(expr, LiteFrameAnd)
    assert isinstance(expr.left, left_cls)
    assert result.dtypes["a"] == _BOOL


def test_between_invalid_inclusive_raises():
    lf = LF({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        lf.between(1, 3, inclusive="bad")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "df_data,left,right,inclusive",
    [
        ({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}, 2, 3, "both"),
        ({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}, 2, 3, "left"),
        ({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}, 2, 3, "right"),
        ({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}, 2, 3, "neither"),
        ({"a": [0.5, 1.5, 2.5]}, 1.0, 2.0, "both"),
    ],
    ids=["both", "left", "right", "neither", "floats"],
)
def test_between_evaluation(df_data, left, right, inclusive):
    df = pd.DataFrame(df_data)
    lf = LF(df)
    result = lf.between(left, right, inclusive=inclusive)
    evaluated = _eval_projection(result, df)
    expected = pd.DataFrame(
        {c: df[c].between(left, right, inclusive=inclusive) for c in df.columns}
    )
    pd.testing.assert_frame_equal(evaluated, expected)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def test_between_after_arithmetic_fuses():
    """between on an arithmetic result should fuse into a single projection."""
    from maxframe.liteframe.operators.project import LiteFrameProjection

    lf = LF({"a": [1, 2, 3]})
    result = (lf + 10).between(12, 13)
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    expr = _inner_expr(result)
    assert isinstance(expr, LiteFrameAnd)
    # The fused arithmetic operand should be referenced directly, not via a
    # bare LiteFrameColumn.
    assert expr.left is not None
    assert expr.right is not None
