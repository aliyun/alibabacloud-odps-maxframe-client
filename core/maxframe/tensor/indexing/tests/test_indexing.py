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

import numpy as np
import pytest

from ...datasource import array, empty, ones, tensor
from ...datasource.ones import TensorOnes
from .. import choose, compress, fill_diagonal, unravel_index
from ..setitem import TensorIndexSetValue


def test_bool_indexing():
    t = ones((100, 200, 300))
    indexed = t[t < 2]
    assert len(indexed.shape) == 1
    assert np.isnan(indexed.shape[0])

    t2 = ones((100, 200))
    indexed = t[t2 < 2]
    assert len(indexed.shape) == 2
    assert np.isnan(indexed.shape[0])
    assert indexed.shape[1] == 300

    t2 = ones((100, 200))
    indexed = t[t2 < 2] + 1
    assert len(indexed.shape) == 2
    assert np.isnan(indexed.shape[0])
    assert indexed.shape[1] == 300

    t2 = ones((10, 20))
    rs = np.random.RandomState(0)
    i1 = np.zeros(10, dtype=bool)
    i1[rs.permutation(np.arange(10))[:5]] = True
    i2 = np.zeros(20, dtype=bool)
    i2[rs.permutation(np.arange(20))[:5]] = True
    indexed = t2[i1, i2]
    assert len(indexed.shape) == 1
    assert indexed.shape[0] == 5

    t3 = ones((101, 200))
    with pytest.raises(IndexError) as cm:
        _ = t[t3 < 2]  # noqa: F841
    e = cm.value.args[0]
    assert "along dimension 0" in e
    assert "dimension is 100 but corresponding boolean dimension is 101" in e

    t4 = ones((100, 201))
    with pytest.raises(IndexError) as cm:
        _ = t[t4 < 2]  # noqa: F841
    e = cm.value.args[0]
    assert "along dimension 1" in e
    assert "dimension is 200 but corresponding boolean dimension is 201" in e


def test_slice():
    t = ones((100, 200, 300))
    t2 = t[10:30, 199:, -30:303]
    assert t2.shape == (20, 1, 30)

    t3 = t[10:90:4, 20:80:5]
    s1 = len(list(range(100))[10:90:4])
    s2 = len(list(range(200))[20:80:5])
    assert t3.shape == (s1, s2, 300)


def test_fancy_indexing():
    t = ones((100, 200, 300))
    t2 = t[[0, 1], [2, 3]]
    assert t2.shape == (2, 300)

    t3 = t[[[0, 1], [2, 3]], [4, 5]]
    assert t3.shape == (2, 2, 300)

    with pytest.raises(IndexError) as cm:
        _ = t[[1, 2], [3, 4, 5]]  # noqa: F841
    e = cm.value.args[0]
    assert (
        e == "shape mismatch: indexing arrays could not be broadcast "
        "together with shapes (2,) (3,)"
    )

    with pytest.raises(IndexError):
        t[[100]]

    t = ones((100, 200, 300), chunk_size=10)

    # fancy index on numpy ndarrays

    t4 = t[:10, -10:, [13, 244, 151, 242, 34]]
    assert t4.shape == (10, 10, 5)

    t5 = t[:10, -10:, [1, 10, 20, 33, 34, 200]]
    assert t5.shape == (10, 10, 6)

    t6 = t[[20, 1, 33, 22, 11], :15, [255, 211, 2, 11, 121]]
    assert t6.shape == (5, 15)

    t7 = t[[5, 6, 33, 66], :15, [0, 9, 2, 11]]
    assert t7.shape == (4, 15)

    t8 = t[[[5, 33], [66, 6]], :15, [255, 11]]
    assert t8.shape == (2, 2, 15)

    # fancy index on tensors

    t9 = t[:10, -10:, tensor([13, 244, 151, 242, 34], chunk_size=2)]
    assert t9.shape == (10, 10, 5)

    t10 = t[:10, -10:, tensor([1, 10, 20, 33, 34, 200], chunk_size=4)]
    assert t10.shape == (10, 10, 6)

    t11 = t[
        tensor([20, 1, 33, 22, 11], chunk_size=2),
        :15,
        tensor([255, 211, 2, 11, 121], chunk_size=3),
    ]
    assert t11.shape == (5, 15)
    # need a concat, because the fancy indexes are not ascending according to chunk index

    t12 = t[tensor([5, 6, 33, 66], chunk_size=2), :15, [0, 9, 2, 11]]
    assert t12.shape == (4, 15)

    t13 = t[tensor([[5, 33], [66, 6]]), :15, tensor([255, 11])]
    assert t13.shape == (2, 2, 15)


def test_mixed_indexing():
    t = ones((100, 200, 300, 400))

    with pytest.raises(IndexError):
        _ = t[ones((100, 200), dtype=float)]  # noqa: F841

    t2 = t[ones(100) < 2, ..., 20::101, 2]
    assert len(t2.shape) == 3
    assert np.isnan(t2.shape[0])

    t3 = ones((2, 3, 4, 5))
    t4 = t3[1]
    assert t4.flags["C_CONTIGUOUS"] == np.ones((2, 3, 4, 5))[1].flags["C_CONTIGUOUS"]
    assert t4.flags["F_CONTIGUOUS"] == np.ones((2, 3, 4, 5))[1].flags["F_CONTIGUOUS"]


def test_setitem():
    shape = (10, 20, 30, 40)
    t = ones(shape, chunk_size=5, dtype="i4")
    t[5:20:3, 5, ..., :-5] = 2.2

    assert isinstance(t.op, TensorIndexSetValue)
    assert t.shape == shape
    assert isinstance(t.inputs[0].op.outputs[0].op, TensorOnes)

    t2 = ones(shape, chunk_size=5, dtype="i4")
    shape = t2[5:20:3, 5, ..., :-5].shape
    t2[5:20:3, 5, ..., :-5] = ones(shape, chunk_size=4, dtype="i4") * 2

    assert isinstance(t2.op, TensorIndexSetValue)


def test_choose():
    choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]

    with pytest.raises(TypeError):
        choose([2, 3, 1, 0], choices, out=1)

    with pytest.raises(ValueError):
        choose([2, 3, 1, 0], choices, out=tensor(np.empty((1, 4))))


def test_unravel_index():
    indices = tensor([22, 41, 37], chunk_size=1)
    t = unravel_index(indices, (7, 6))

    assert len(t) == 2

    with pytest.raises(TypeError):
        unravel_index([22, 41, 37], (7, 6), order="B")


def test_compress():
    a = np.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(TypeError):
        compress([0, 1], a, axis=0, out=1)

    with pytest.raises(TypeError):
        compress(
            [0, 1],
            array([[1, 2], [3, 4], [5, 6]], dtype="i8"),
            axis=0,
            out=empty((1, 2), dtype="f8"),
        )


def test_operator_key():
    t = ones((10, 2), chunk_size=5)
    t_slice1 = t[:5]
    t_slice2 = t[5:]

    assert t_slice1.op.key != t_slice2.op.key


def test_fill_diagonal():
    a = tensor(np.random.rand(10, 13))
    fill_diagonal(a, 10)

    assert a.shape == (10, 13)

    # must be Tensor
    with pytest.raises(TypeError):
        fill_diagonal(np.random.rand(11, 10), 1)

    # at least 2-d required
    with pytest.raises(ValueError):
        a = tensor(np.random.rand(4))
        fill_diagonal(a, 1)

    # for more than 2-d, shape on each dimension should be equal
    with pytest.raises(ValueError):
        a = tensor(np.random.rand(11, 10, 11))
        fill_diagonal(a, 1)
