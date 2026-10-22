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

import numpy as np
import pandas as pd
import pytest

import maxframe.tensor as mt
from maxframe import opcodes
from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.datasource.from_tensor import liteframe_from_tensor
from maxframe.liteframe.initializer import LiteFrame
from maxframe.liteframe.operators.datasource import LiteFrameFromTensor
from maxframe.utils import is_arrow_dtype_supported


def _tensor(shape, chunk_size=2):
    return mt.tensor(np.arange(np.prod(shape)).reshape(shape), chunk_size=chunk_size)


@pytest.mark.parametrize(
    "shape,columns,expected_columns,expected_shape",
    [
        ((4, 3), None, ["0", "1", "2"], (4, 3)),
        ((4, 3), ["a", "b", "c"], ["a", "b", "c"], (4, 3)),
        ((5,), None, ["0"], (5, 1)),
        ((5,), "val", ["val"], (5, 1)),
        ((5,), ["val"], ["val"], (5, 1)),
        # column names are normalized to lowercase
        ((2, 3), ["A", "B", "C"], ["a", "b", "c"], (2, 3)),
    ],
    ids=[
        "2d-default",
        "2d-named",
        "1d-default",
        "1d-str",
        "1d-list",
        "2d-lowercase",
    ],
)
def test_from_tensor_columns(shape, columns, expected_columns, expected_shape):
    lf = liteframe_from_tensor(_tensor(shape), columns=columns)
    assert isinstance(lf, LITEFRAME_TYPE)
    assert lf.shape == expected_shape
    assert list(lf.columns) == expected_columns


def test_from_tensor_operator_uses_shared_opcode():
    tensor = _tensor((4, 3))
    lf = liteframe_from_tensor(tensor)
    op = lf.data.op
    assert isinstance(op, LiteFrameFromTensor)
    assert op.input is tensor.data
    # LITEFRAME_FROM_TENSOR is an alias sharing DATAFRAME_FROM_TENSOR.
    assert op._op_type_ == opcodes.LITEFRAME_FROM_TENSOR
    assert opcodes.LITEFRAME_FROM_TENSOR == opcodes.DATAFRAME_FROM_TENSOR


@pytest.mark.skipif(
    not is_arrow_dtype_supported(), reason="ArrowDtype is not supported"
)
def test_from_tensor_dtype_is_arrow():
    tensor = mt.tensor(np.arange(5, dtype=np.int64))
    lf = liteframe_from_tensor(tensor)
    # LiteFrame stores Arrow dtypes, matching from_local_df convention.
    assert isinstance(lf.dtypes.iloc[0], pd.ArrowDtype)


def test_init_with_dtype():
    tensor = mt.tensor(np.arange(5, dtype=np.int64))
    lf = LiteFrame(tensor, columns="val", dtype="float64")
    assert lf.shape == (5, 1)
    assert list(lf.columns) == ["val"]


def test_init_with_chunk_size():
    lf = LiteFrame(_tensor((4, 3), chunk_size=None), chunk_size=2)
    assert lf.shape == (4, 3)


@pytest.mark.parametrize(
    "shape,columns,exc",
    [
        ((2, 2, 2), None, TypeError),
        ((2, 3), "x", ValueError),
        ((2, 3), ["a", "b"], ValueError),
        ((2, 3), ["A", "a", "b"], ValueError),
    ],
    ids=["ndim", "str-cols-2d", "len-mismatch", "dup-after-normalize"],
)
def test_from_tensor_errors(shape, columns, exc):
    with pytest.raises(exc):
        liteframe_from_tensor(_tensor(shape), columns=columns)
