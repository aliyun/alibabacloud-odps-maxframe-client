# Copyright 1999-2025 Alibaba Group Holding Ltd.
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

from copy import copy

import numpy as np
import pytest
import scipy.sparse as sps

from .... import dataframe as md
from ....core import enter_mode
from ... import arange, diag, full, ones, ones_like, tensor, zeros
from ...core import SparseTensor, Tensor
from .. import array, asarray, ascontiguousarray, asfortranarray, fromdense
from ..array import CSRMatrixDataSource
from ..from_dataframe import from_dataframe
from ..from_dense import DenseToSparse
from ..ones import TensorOnesLike


def test_array():
    a = tensor([0, 1, 2], chunk_size=2)

    b = array(a)
    assert a is not b

    c = asarray(a)
    assert a is c


def test_ascontiguousarray():
    # dtype different
    raw_a = np.asfortranarray(np.random.rand(2, 4))
    raw_b = np.ascontiguousarray(raw_a, dtype="f4")

    a = tensor(raw_a, chunk_size=2)
    b = ascontiguousarray(a, dtype="f4")

    assert a.dtype == raw_a.dtype
    assert a.flags["C_CONTIGUOUS"] == raw_a.flags["C_CONTIGUOUS"]
    assert a.flags["F_CONTIGUOUS"] == raw_a.flags["F_CONTIGUOUS"]

    assert b.dtype == raw_b.dtype
    assert b.flags["C_CONTIGUOUS"] == raw_b.flags["C_CONTIGUOUS"]
    assert b.flags["F_CONTIGUOUS"] == raw_b.flags["F_CONTIGUOUS"]

    # no copy
    raw_a = np.random.rand(2, 4)
    raw_b = np.ascontiguousarray(raw_a)

    a = tensor(raw_a, chunk_size=2)
    b = ascontiguousarray(a)

    assert a.dtype == raw_a.dtype
    assert a.flags["C_CONTIGUOUS"] == raw_a.flags["C_CONTIGUOUS"]
    assert a.flags["F_CONTIGUOUS"] == raw_a.flags["F_CONTIGUOUS"]

    assert b.dtype == raw_b.dtype
    assert b.flags["C_CONTIGUOUS"] == raw_b.flags["C_CONTIGUOUS"]
    assert b.flags["F_CONTIGUOUS"] == raw_b.flags["F_CONTIGUOUS"]


def test_asfortranarray():
    # dtype different
    raw_a = np.random.rand(2, 4)
    raw_b = np.asfortranarray(raw_a, dtype="f4")

    a = tensor(raw_a, chunk_size=2)
    b = asfortranarray(a, dtype="f4")

    assert a.dtype == raw_a.dtype
    assert a.flags["C_CONTIGUOUS"] == raw_a.flags["C_CONTIGUOUS"]
    assert a.flags["F_CONTIGUOUS"] == raw_a.flags["F_CONTIGUOUS"]

    assert b.dtype == raw_b.dtype
    assert b.flags["C_CONTIGUOUS"] == raw_b.flags["C_CONTIGUOUS"]
    assert b.flags["F_CONTIGUOUS"] == raw_b.flags["F_CONTIGUOUS"]

    # no copy
    raw_a = np.asfortranarray(np.random.rand(2, 4))
    raw_b = np.asfortranarray(raw_a)

    a = tensor(raw_a, chunk_size=2)
    b = asfortranarray(a)

    assert a.dtype == raw_a.dtype
    assert a.flags["C_CONTIGUOUS"] == raw_a.flags["C_CONTIGUOUS"]
    assert a.flags["F_CONTIGUOUS"] == raw_a.flags["F_CONTIGUOUS"]

    assert b.dtype == raw_b.dtype
    assert b.flags["C_CONTIGUOUS"] == raw_b.flags["C_CONTIGUOUS"]
    assert b.flags["F_CONTIGUOUS"] == raw_b.flags["F_CONTIGUOUS"]


def test_ones():
    tensor = ones((10, 10, 8), chunk_size=(3, 3, 5))
    assert tensor.shape == (10, 10, 8)

    tensor = ones((10, 3), chunk_size=(4, 2))
    assert tensor.shape == (10, 3)

    tensor = ones((10, 5), chunk_size=(2, 3), gpu=True)
    assert tensor.op.gpu is True

    tensor = ones((2, 3, 4))
    assert len(list(tensor)) == 2

    tensor2 = ones((2, 3, 4), chunk_size=1)
    assert tensor.op.key != tensor2.op.key
    assert tensor.key != tensor2.key

    tensor3 = ones((2, 3, 3))
    assert tensor.op.key != tensor3.op.key
    assert tensor.key != tensor3.key


def test_zeros():
    tensor = zeros((2, 3, 4))
    assert len(list(tensor)) == 2
    assert tensor.op.gpu is None

    tensor2 = zeros((2, 3, 4), chunk_size=1)
    # tensor's op key must be equal to tensor2
    assert tensor.op.key != tensor2.op.key
    assert tensor.key != tensor2.key

    tensor3 = zeros((2, 3, 3))
    assert tensor.op.key != tensor3.op.key
    assert tensor.key != tensor3.key


def test_data_source():
    from ...misc.broadcast_to import TensorBroadcastTo

    data = np.random.random((10, 3))
    t = tensor(data, chunk_size=2)
    assert t.op.gpu is None

    t = tensor(data, chunk_size=2, gpu=True)
    assert t.op.gpu is True

    t = full((2, 2), 2, dtype="f4")
    assert t.op.gpu is None
    assert t.shape == (2, 2)
    assert t.dtype == np.float32

    t = full((2, 2), [1.0, 2.0], dtype="f4")
    assert t.shape == (2, 2)
    assert t.dtype == np.float32
    assert isinstance(t.op, TensorBroadcastTo)

    with pytest.raises(ValueError):
        full((2, 2), [1.0, 2.0, 3.0], dtype="f4")


def test_ufunc():
    t = ones((3, 10), chunk_size=2)

    x = np.add(t, [[1], [2], [3]])
    assert isinstance(x, Tensor)

    y = np.sum(t, axis=1)
    assert isinstance(y, Tensor)


def test_arange():
    t = arange(10, chunk_size=3)

    assert t.op.gpu is False
    assert t.shape == (10,)

    t = arange(0, 10, 3, chunk_size=2)
    assert t.shape == (4,)

    pytest.raises(TypeError, lambda: arange(10, start=0))
    pytest.raises(TypeError, lambda: arange(0, 10, stop=0))
    pytest.raises(TypeError, lambda: arange())
    pytest.raises(
        ValueError, lambda: arange("1066-10-13", dtype=np.datetime64, chunks=3)
    )


def test_set_tensor_inputs():
    t1 = tensor([1, 2], chunk_size=2)
    t2 = tensor([2, 3], chunk_size=2)
    t3 = t1 + t2

    t1c = copy(t1)
    t2c = copy(t2)

    assert t1c is not t1
    assert t2c is not t2

    assert t3.op.lhs is t1.data
    assert t3.op.rhs is t2.data
    assert t3.op.inputs == [t1.data, t2.data]
    assert t3.inputs == [t1.data, t2.data]

    msg = "Unexpected StopIteration happened"
    with pytest.raises(RuntimeError, match=msg):
        t3.inputs = []

    t1 = tensor([1, 2], chunk_size=2)
    t2 = tensor([True, False], chunk_size=2)
    t3 = t1[t2]

    t1c = copy(t1)
    t2c = copy(t2)
    t3c = copy(t3)
    t3c.inputs = [t1c, t2c]

    with enter_mode(build=True):
        assert t3c.op.input is t1c.data
        assert t3c.op.indexes[0] is t2c.data


def test_from_spmatrix():
    t = tensor(sps.csr_matrix([[0, 0, 1], [1, 0, 0]], dtype="f8"), chunk_size=2)

    assert isinstance(t, SparseTensor)
    assert isinstance(t.op, CSRMatrixDataSource)
    assert t.issparse() is True
    assert not t.op.gpu
    assert isinstance(t.op, CSRMatrixDataSource)
    assert not t.op.gpu


def test_from_dense():
    t = fromdense(tensor([[0, 0, 1], [1, 0, 0]], chunk_size=2))

    assert isinstance(t, SparseTensor)
    assert isinstance(t.op, DenseToSparse)
    assert t.issparse() is True
    assert isinstance(t.op, DenseToSparse)


def test_ones_like():
    t1 = tensor([[0, 0, 1], [1, 0, 0]], chunk_size=2).tosparse()
    t = ones_like(t1, dtype="f8")

    assert isinstance(t, SparseTensor)
    assert isinstance(t.op, TensorOnesLike)
    assert t.issparse() is True
    assert t.op.gpu is None

    assert isinstance(t.op, TensorOnesLike)


def test_from_array():
    x = array([1, 2, 3])
    assert x.shape == (3,)

    y = array([x, x])
    assert y.shape == (2, 3)

    z = array((x, x, x))
    assert z.shape == (3, 3)


def test_from_dataframe():
    mdf = md.DataFrame(
        {"a": [0, 1, 2], "b": [3, 4, 5], "c": [0.1, 0.2, 0.3]},
        index=["c", "d", "e"],
        chunk_size=2,
    )
    tensor = from_dataframe(mdf)
    assert tensor.shape == (3, 3)
    assert np.float64 == tensor.dtype


def test_diag():
    # test 2-d, shape[0] == shape[1], k == 0
    v = tensor(np.arange(16).reshape(4, 4), chunk_size=2)
    t = diag(v)
    assert t.shape == (4,)

    v = tensor(np.arange(16).reshape(4, 4), chunk_size=(2, 3))
    t = diag(v)
    assert t.shape == (4,)

    # test 1-d, k == 0
    v = tensor(np.arange(3), chunk_size=2)
    t = diag(v, sparse=True)
    assert t.shape == (3, 3)

    # test 2-d, shape[0] != shape[1]
    v = tensor(np.arange(24).reshape(4, 6), chunk_size=2)
    t = diag(v)
    assert t.shape == np.diag(np.arange(24).reshape(4, 6)).shape

    v = tensor(np.arange(24).reshape(4, 6), chunk_size=2)
    t = diag(v, k=1)
    assert t.shape == np.diag(np.arange(24).reshape(4, 6), k=1).shape
    t = diag(v, k=2)
    assert t.shape == np.diag(np.arange(24).reshape(4, 6), k=2).shape
    t = diag(v, k=-1)
    assert t.shape == np.diag(np.arange(24).reshape(4, 6), k=-1).shape
    t = diag(v, k=-2)
    assert t.shape == np.diag(np.arange(24).reshape(4, 6), k=-2).shape
