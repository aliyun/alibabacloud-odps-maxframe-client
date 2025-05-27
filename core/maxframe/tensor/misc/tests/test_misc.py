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

from ...datasource import arange, ones, tensor
from .. import broadcast_to, ravel, transpose, where


def test_dir():
    a = tensor([0, 1, 2], chunk_size=2)
    tensor_dir = dir(a)
    for attr in dir(a.data):
        assert attr in tensor_dir


def test_astype():
    arr = ones((10, 20, 30), chunk_size=3)

    arr2 = arr.astype(np.int32)

    assert arr2.shape == (10, 20, 30)
    assert np.issubdtype(arr2.dtype, np.int32) is True
    assert arr2.op.casting == "unsafe"

    with pytest.raises(TypeError):
        arr.astype(np.int32, casting="safe")

    arr3 = arr.astype(arr.dtype, order="F")
    assert arr3.flags["F_CONTIGUOUS"] is True
    assert arr3.flags["C_CONTIGUOUS"] is False


def test_transpose():
    arr = ones((10, 20, 30), chunk_size=[4, 3, 5])

    arr2 = transpose(arr)

    assert arr2.shape == (30, 20, 10)

    with pytest.raises(ValueError):
        transpose(arr, axes=(1, 0))

    arr3 = transpose(arr, (-2, 2, 0))

    assert arr3.shape == (20, 30, 10)

    arr4 = arr.transpose(-2, 2, 0)

    assert arr4.shape == (20, 30, 10)

    arr5 = arr.T

    assert arr5.shape == (30, 20, 10)


def test_broadcast_to():
    arr = ones((10, 5), chunk_size=2)
    arr2 = broadcast_to(arr, (20, 10, 5))

    assert arr2.shape == (20, 10, 5)

    arr = ones((10, 5, 1), chunk_size=2)
    arr3 = broadcast_to(arr, (5, 10, 5, 6))

    assert arr3.shape == (5, 10, 5, 6)

    arr = ones((10, 1), chunk_size=2)
    arr4 = broadcast_to(arr, (20, 10, 5))

    assert arr4.shape == (20, 10, 5)

    with pytest.raises(ValueError):
        broadcast_to(arr, (10,))

    with pytest.raises(ValueError):
        broadcast_to(arr, (5, 1))

    arr = ones((4, 5), chunk_size=2)
    with pytest.raises((ValueError)):
        broadcast_to(arr[arr < 2], (3, 20))


def test_where():
    cond = tensor([[True, False], [False, True]], chunk_size=1)
    x = tensor([1, 2], chunk_size=1)

    with pytest.raises(ValueError):
        where(cond, x)

    x = arange(9.0).reshape(3, 3)
    y = where(x < 5, x, -1)

    assert y.dtype == np.float64


def test_ravel():
    arr = ones((10, 5), chunk_size=2)
    flat_arr = ravel(arr)
    assert flat_arr.shape == (50,)
