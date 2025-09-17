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

from ..arithmetic.utils import arithmetic_operator
from ..utils import implement_scipy, infer_scipy_dtype
from .core import TensorSpecialBinOp, TensorSpecialMultiOp, _register_special_op


@_register_special_op
class TensorHYP2F1(TensorSpecialMultiOp):
    _ARG_COUNT = 4
    _func_name = "hyp2f1"


@implement_scipy("scipy.special.hyp2f1")
@infer_scipy_dtype("scipy.special.hyp2f1")
def hyp2f1(a, b, c, z, **kwargs):
    op = TensorHYP2F1(**kwargs)
    return op(a, b, c, z)


@_register_special_op
class TensorHYP1F1(TensorSpecialMultiOp):
    _ARG_COUNT = 3
    _func_name = "hyp1f1"


@implement_scipy("scipy.special.hyp1f1")
@infer_scipy_dtype("scipy.special.hyp1f1")
def hyp1f1(a, b, x, out=None, **kwargs):
    op = TensorHYP1F1(**kwargs)
    return op(a, b, x, out=out)


@_register_special_op
class TensorHYPERU(TensorSpecialMultiOp):
    _ARG_COUNT = 3
    _func_name = "hyperu"


@implement_scipy("scipy.special.hyperu")
@infer_scipy_dtype("scipy.special.hyperu")
def hyperu(a, b, x, out=None, **kwargs):
    op = TensorHYPERU(**kwargs)
    return op(a, b, x, out=out)


@_register_special_op
@arithmetic_operator(sparse_mode="binary_and")
class TensorHYP0F1(TensorSpecialBinOp):
    _func_name = "hyp0f1"


@implement_scipy("scipy.special.hyp0f1")
@infer_scipy_dtype("scipy.special.hyp0f1")
def hyp0f1(v, z, out=None, **kwargs):
    op = TensorHYP0F1(**kwargs)
    return op(v, z, out=out)
