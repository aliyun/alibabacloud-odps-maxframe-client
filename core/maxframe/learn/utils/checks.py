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

from typing import List

import numpy as np

from ... import opcodes
from ... import tensor as mt
from ...config import options
from ...core import ENTITY_TYPE, EntityData, OutputType, get_output_types
from ...core.operator import Operator
from ...serialization.serializables import (
    BoolField,
    DataTypeField,
    KeyField,
    StringField,
)
from ...tensor.core import TensorOrder
from ..core import LearnOperatorMixin

try:
    from sklearn import get_config as get_sklearn_config
except ImportError:  # pragma: no cover
    get_sklearn_config = None


class CheckBase(Operator, LearnOperatorMixin):
    input = KeyField("input")
    value = KeyField("value", default=None)
    err_msg = StringField("err_msg", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _set_inputs(cls, op: "CheckBase", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if op.input is not None:
            op.input = op._inputs[0]
        if op.value is not None:
            op.value = op._inputs[-1]

    def __call__(self, x, value=None):
        # output input if value not specified
        self.value = value = value if value is not None else x
        self.output_types = get_output_types(value)
        return self.new_tileable([x, value], kws=[value.params])


class CheckNonNegative(CheckBase):
    _op_type_ = opcodes.CHECK_NON_NEGATIVE

    whom = StringField("whom", default=None)

    def __init__(self, gpu=None, output_types=None, **kw):
        super().__init__(output_types=output_types, gpu=gpu, **kw)
        if self.err_msg is None and self.whom is not None:
            self.err_msg = f"Negative values in data passed to {self.whom}"


def check_non_negative_then_return_value(to_check, value, whom):
    op = CheckNonNegative(
        input=to_check, value=value, whom=whom, sparse=to_check.is_sparse()
    )
    return op(to_check, value)


class AssertAllFinite(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.ASSERT_ALL_FINITE

    x = KeyField("x")
    allow_nan = BoolField("allow_nan", default=None)
    msg_dtype = DataTypeField("msg_dtype", default=None)
    check_only = BoolField("check_only", default=None)
    input_name = StringField("input_name", default=None)
    # chunks
    is_finite = KeyField("is_finite", default=None)
    check_nan = KeyField("check_nan", default=None)

    def __init__(self, x=None, output_types=None, **kw):
        super().__init__(x=x, _output_types=output_types, **kw)

    @classmethod
    def _set_inputs(cls, op: "AssertAllFinite", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op.inputs)
        for attr in ("x", "is_finite", "check_nan"):
            if getattr(op, attr) is not None:
                setattr(op, attr, next(inputs_iter))

    @classmethod
    def _assume_finite(cls):
        assume_finite = options.learn.assume_finite
        if assume_finite is None and get_sklearn_config is not None:
            # get config from scikit-learn
            assume_finite = get_sklearn_config()["assume_finite"]
        if assume_finite is None:  # pragma: no cover
            assume_finite = False

        return assume_finite

    def __call__(self, x):
        if self._assume_finite():
            # skip check
            if self.check_only:
                return
            else:
                return x

        if self.check_only:
            return self.new_tileable(
                [x], dtype=np.dtype(bool), shape=(), order=TensorOrder.C_ORDER
            )
        else:
            return self.new_tileable([x], kws=[x.params])


def assert_all_finite(
    X, allow_nan=False, msg_dtype=None, check_only=True, input_name=""
):
    if not isinstance(X, ENTITY_TYPE):
        X = mt.asarray(X)

    if (
        isinstance(X.op, AssertAllFinite)
        and X.op.allow_nan == allow_nan
        and X.op.msg_dtype == msg_dtype
        and X.op.check_only == check_only
    ):
        return X

    if check_only:
        output_types = [OutputType.tensor]
        sparse = False
    else:
        output_types = get_output_types(X)
        sparse = X.issparse()

    op = AssertAllFinite(
        x=X,
        allow_nan=allow_nan,
        msg_dtype=msg_dtype,
        check_only=check_only,
        sparse=sparse,
        output_types=output_types,
        input_name=input_name,
    )
    return op(X)
