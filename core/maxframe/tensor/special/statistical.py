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
from .core import TensorSpecialUnaryOp, _register_special_op


@_register_special_op
@arithmetic_operator(sparse_mode="always_false")
class TensorExpit(TensorSpecialUnaryOp):
    _func_name = "expit"


@_register_special_op
@arithmetic_operator(sparse_mode="always_false")
class TensorLogit(TensorSpecialUnaryOp):
    _func_name = "logit"


@_register_special_op
@arithmetic_operator(sparse_mode="always_false")
class TensorLogExpit(TensorSpecialUnaryOp):
    _func_name = "log_expit"


@implement_scipy("scipy.special.expit")
@infer_scipy_dtype("scipy.special.expit")
def expit(x, **kwargs):
    op = TensorExpit(**kwargs)
    return op(x)


@implement_scipy("scipy.special.logit")
@infer_scipy_dtype("scipy.special.logit")
def logit(x, **kwargs):
    op = TensorLogit(**kwargs)
    return op(x)


@implement_scipy("scipy.special.log_expit")
@infer_scipy_dtype("scipy.special.log_expit")
def log_expit(x, **kwargs):
    op = TensorLogExpit(**kwargs)
    return op(x)
