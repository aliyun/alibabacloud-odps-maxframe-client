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

from ....utils import collect_leaf_operators
from ...datasource import ones, tensor
from .. import *  # noqa: F401
from ..core import TensorReduction


def test_base_reduction():
    sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
    prod = lambda x, *args, **kwargs: x.prod(*args, **kwargs)
    max = lambda x, *args, **kwargs: x.max(*args, **kwargs)
    min = lambda x, *args, **kwargs: x.min(*args, **kwargs)
    all = lambda x, *args, **kwargs: x.all(*args, **kwargs)
    any = lambda x, *args, **kwargs: x.any(*args, **kwargs)

    for f in [sum, prod, max, min, all, any]:
        res = f(ones((8, 8), chunk_size=8))
        assert res.shape == ()

        res = f(ones((10, 8), chunk_size=3))
        assert res.dtype is not None
        assert res.shape == ()

        res = f(ones((10, 8), chunk_size=3), axis=0)
        assert res.shape == (8,)

        res = f(ones((10, 8), chunk_size=3), axis=1)
        assert res.shape == (10,)

        with pytest.raises(np.AxisError):
            f(ones((10, 8), chunk_size=3), axis=2)

        res = f(ones((10, 8), chunk_size=3), axis=-1)
        assert res.shape == (10,)

        with pytest.raises(np.AxisError):
            f(ones((10, 8), chunk_size=3), axis=-3)

        res = f(ones((10, 8), chunk_size=3), keepdims=True)
        assert res.shape == (1, 1)

        res = f(ones((10, 8), chunk_size=3), axis=0, keepdims=True)
        assert res.shape == (1, 8)

        res = f(ones((10, 8), chunk_size=3), axis=1, keepdims=True)
        assert res.shape == (10, 1)

        res = f(ones((10, 8, 10), chunk_size=3), axis=1)
        assert res.shape == (10, 10)

        res = f(ones((10, 8, 10), chunk_size=3), axis=1, keepdims=True)
        assert res.shape == (10, 1, 10)

        res = f(ones((10, 8, 10), chunk_size=3), axis=(0, 2))
        assert res.shape == (8,)

        res = f(ones((10, 8, 10), chunk_size=3), axis=(0, 2), keepdims=True)
        assert res.shape == (1, 8, 1)


def test_mean_reduction():
    mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)

    res = mean(ones((10, 8), chunk_size=3))
    assert res.shape == ()
    assert res.dtype is not None

    res = mean(ones((8, 8), chunk_size=8))
    assert res.shape == ()

    res = mean(ones((10, 8), chunk_size=3), axis=0)
    assert res.shape == (8,)

    res = mean(ones((10, 8), chunk_size=3), axis=1)
    assert res.shape == (10,)

    with pytest.raises(np.AxisError):
        mean(ones((10, 8), chunk_size=3), axis=2)

    res = mean(ones((10, 8), chunk_size=3), axis=-1)
    assert res.shape == (10,)

    with pytest.raises(np.AxisError):
        mean(ones((10, 8), chunk_size=3), axis=-3)

    res = mean(ones((10, 8), chunk_size=3), keepdims=True)
    assert res.shape == (1, 1)

    res = mean(ones((10, 8), chunk_size=3), axis=0, keepdims=True)
    assert res.shape == (1, 8)

    res = mean(ones((10, 8), chunk_size=3), axis=1, keepdims=True)
    assert res.shape == (10, 1)


def test_arg_reduction():
    argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
    argmin = lambda x, *args, **kwargs: x.argmin(*args, **kwargs)

    res1 = argmax(ones((10, 8, 10), chunk_size=3))
    res2 = argmin(ones((10, 8, 10), chunk_size=3))
    assert res1.shape == ()
    assert res1.dtype is not None
    assert res2.shape == ()

    res1 = argmax(ones((10, 8), chunk_size=3), axis=1)
    res2 = argmin(ones((10, 8), chunk_size=3), axis=1)
    assert res1.shape == (10,)
    assert res2.shape == (10,)

    pytest.raises(
        TypeError, lambda: argmax(ones((10, 8, 10), chunk_size=3), axis=(0, 1))
    )
    pytest.raises(
        TypeError, lambda: argmin(ones((10, 8, 10), chunk_size=3), axis=(0, 1))
    )
    pytest.raises(np.AxisError, lambda: argmin(ones((10, 8, 10), chunk_size=3), axis=3))
    pytest.raises(
        np.AxisError, lambda: argmin(ones((10, 8, 10), chunk_size=3), axis=-4)
    )


def test_cum_reduction():
    cumsum = lambda x, *args, **kwargs: x.cumsum(*args, **kwargs)
    cumprod = lambda x, *args, **kwargs: x.cumprod(*args, **kwargs)

    res1 = cumsum(ones((10, 8), chunk_size=3), axis=0)
    res2 = cumprod(ones((10, 8), chunk_size=3), axis=0)
    assert res1.shape == (10, 8)
    assert res1.dtype is not None
    assert res2.shape == (10, 8)
    assert res2.dtype is not None

    res1 = cumsum(ones((10, 8, 8), chunk_size=3), axis=1)
    res2 = cumprod(ones((10, 8, 8), chunk_size=3), axis=1)
    assert res1.shape == (10, 8, 8)
    assert res2.shape == (10, 8, 8)

    res1 = cumsum(ones((10, 8, 8), chunk_size=3), axis=-2)
    res2 = cumprod(ones((10, 8, 8), chunk_size=3), axis=-2)
    assert res1.shape == (10, 8, 8)
    assert res2.shape == (10, 8, 8)

    with pytest.raises(np.AxisError):
        cumsum(ones((10, 8), chunk_size=3), axis=2)
    with pytest.raises(np.AxisError):
        cumsum(ones((10, 8), chunk_size=3), axis=-3)


def test_all_reduction():
    o = tensor([False])

    with pytest.raises(ValueError):
        all([-1, 4, 5], out=o)


def test_var_reduction():
    var = lambda x, *args, **kwargs: x.var(*args, **kwargs)

    res1 = var(ones((10, 8), chunk_size=3), ddof=2)
    assert res1.shape == ()
    assert res1.op.ddof == 2

    res1 = var(ones((10, 8, 8), chunk_size=3), axis=1)
    assert res1.shape == (10, 8)


def test_reduction_op_func_name():
    # make sure all the binary op has defined the func name.

    results = collect_leaf_operators(TensorReduction)
    for op_type in results:
        assert hasattr(op_type, "_func_name")
