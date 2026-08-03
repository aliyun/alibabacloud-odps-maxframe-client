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

import pyarrow as pa
import pytest

from maxframe.liteframe import LiteFrame
from maxframe.liteframe.accessors.list_.core import LiteFrameListExpr
from maxframe.liteframe.expressions import NamedExpr
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import wrap_arrow_dtype

_LIST_DTYPE = wrap_arrow_dtype(pa.list_(pa.int64()))


def test_list_accessor_single_column_validation():
    """Test accessor only works on single-column LiteFrame."""
    df = LiteFrame(
        {"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]]},
        dtype=_LIST_DTYPE,
    )
    with pytest.raises(ValueError, match="multi-column"):
        df.list.len()


def test_list_accessor_dtype_validation():
    """Test accessor raises on non-list dtype."""
    df = LiteFrame({"col": [1, 2, 3]})
    with pytest.raises(AttributeError, match="list values"):
        df.list.len()


@pytest.mark.parametrize(
    "method,args,expected_dtype",
    [
        ("__getitem__", (0,), wrap_arrow_dtype(pa.int64())),
        ("get", (0,), wrap_arrow_dtype(pa.int64())),
        ("len", (), wrap_arrow_dtype(pa.int64())),
    ],
)
def test_list_methods_dtype_inference(method, args, expected_dtype):
    """Test list methods: dtype inference without execution."""
    df = LiteFrame({"col": [[1, 2], [3, 4]]}, dtype=_LIST_DTYPE)
    result = getattr(df.list, method)(*args)

    # Verify result is LiteFrame
    assert isinstance(result, LiteFrame)

    # Verify operator structure
    op = result.data.op
    assert isinstance(op, LiteFrameProjection)
    assert len(op.projections) == 1

    # Verify projection structure
    projection = op.projections[0]
    assert isinstance(projection, NamedExpr)
    assert projection.name == "col"
    assert isinstance(projection.expr, LiteFrameListExpr)
    assert projection.expr.method == method
    assert projection.expr.args == tuple(args)

    # Verify dtype inference
    assert result.dtypes.iloc[0] == expected_dtype
    assert projection.dtype == expected_dtype
