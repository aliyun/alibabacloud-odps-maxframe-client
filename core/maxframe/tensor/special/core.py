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

from ... import opcodes
from ..arithmetic.core import TensorUnaryOp

_func_name_to_special_cls = {}


def _register_special_op(cls):
    if cls._func_name is not None:
        _func_name_to_special_cls[cls._func_name] = cls
    return cls


class TensorSpecialOperatorMixin:
    _op_type_ = opcodes.SPECIAL
    _func_name = None

    def __new__(cls, *args, **kwargs):
        if cls._func_name is not None:
            return object.__new__(_func_name_to_special_cls[cls._func_name])
        return super().__new__(cls, *args, **kwargs)


class TensorSpecialUnaryOp(TensorSpecialOperatorMixin, TensorUnaryOp):
    pass
