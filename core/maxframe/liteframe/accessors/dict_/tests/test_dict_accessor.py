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
from maxframe.liteframe.accessors.dict_.core import LiteFrameDictExpr
from maxframe.liteframe.expressions import NamedExpr
from maxframe.liteframe.operators.project import LiteFrameProjection
from maxframe.utils import wrap_arrow_dtype

_DICT_DTYPE = wrap_arrow_dtype(pa.map_(pa.string(), pa.int64()))
_RAW = [[("k1", 1), ("k2", 2)], [("k1", 3)]]


def test_dict_accessor_single_column_validation():
    """Test accessor only works on single-column LiteFrame."""
    df = LiteFrame({"a": _RAW, "b": _RAW}, dtype=_DICT_DTYPE)
    with pytest.raises(ValueError, match="multi-column"):
        df.dict.len()


def test_dict_accessor_dtype_validation():
    """Test accessor raises on non-map dtype."""
    df = LiteFrame({"col": [1, 2, 3]})
    with pytest.raises(AttributeError, match="dict values"):
        df.dict.len()


@pytest.mark.parametrize(
    "method,args,kwargs,expected_dtype",
    [
        ("__getitem__", ("k1",), {}, wrap_arrow_dtype(pa.int64())),
        ("get", ("k1",), {}, wrap_arrow_dtype(pa.int64())),
        ("get", ("k3",), {"default_value": 99}, wrap_arrow_dtype(pa.int64())),
        ("len", (), {}, wrap_arrow_dtype(pa.int64())),
        ("contains", ("k1",), {}, wrap_arrow_dtype(pa.bool_())),
        ("remove", ("k1",), {}, _DICT_DTYPE),
    ],
)
def test_dict_methods_dtype_inference(method, args, kwargs, expected_dtype):
    """Test dict methods: dtype inference without execution."""
    df = LiteFrame({"col": _RAW}, dtype=_DICT_DTYPE)
    result = getattr(df.dict, method)(*args, **kwargs)

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
    assert isinstance(projection.expr, LiteFrameDictExpr)
    assert projection.expr.method == method
    assert projection.expr.args == tuple(args)
    assert projection.expr.kwargs == kwargs

    # Verify dtype inference
    assert result.dtypes.iloc[0] == expected_dtype
    assert projection.dtype == expected_dtype
