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

from typing import Type

from ... import opcodes
from ...core import ENTITY_TYPE, OutputType
from ...core.operator import ObjectOperator, ObjectOperatorMixin
from ...serialization.serializables import (
    AnyField,
    DictField,
    FunctionField,
    TupleField,
)
from ...udf import BuiltinFunction
from ...utils import find_objects, replace_objects
from ..core import Model, ModelData


class ModelWithEvalData(ModelData):
    __slots__ = ("_evals_result",)

    _evals_result: dict

    def __init__(self, *args, evals_result=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._evals_result = evals_result if evals_result is not None else dict()

    def execute(self, session=None, **kw):
        # The evals_result should be fetched when BoosterData.execute() is called.
        result = super().execute(session=session, **kw)
        if (
            getattr(self.op, "has_evals_result", None)
            and self.key == self.op.outputs[0].key
        ):
            self._evals_result.update(self.op.outputs[1].fetch(session=session))
        return result


class ModelWithEval(Model):
    pass


class ModelDataSource(ObjectOperator, ObjectOperatorMixin):
    _op_type_ = opcodes.MODEL_DATA_SOURCE

    data = AnyField("data")

    def __call__(self, model_cls: Type[ModelWithEval]):
        self._output_types = [OutputType.object]
        return self.new_tileable(None, object_class=model_cls)


class ModelApplyChunk(ObjectOperator, ObjectOperatorMixin):
    _op_module_ = "maxframe.learn.contrib.models"
    _op_type_ = opcodes.APPLY_CHUNK

    func = FunctionField("func")
    args = TupleField("args")
    kwargs = DictField("kwargs")

    def __init__(self, output_types=None, **kwargs):
        if not isinstance(output_types, (tuple, list)):
            output_types = [output_types]
        self._output_types = list(output_types)
        super().__init__(**kwargs)

    def has_custom_code(self) -> bool:
        return not isinstance(self.func, BuiltinFunction)

    @classmethod
    def _set_inputs(cls, op: "ModelApplyChunk", inputs):
        super()._set_inputs(op, inputs)
        old_inputs = find_objects(op.args, ENTITY_TYPE) + find_objects(
            op.kwargs, ENTITY_TYPE
        )
        mapping = {o: n for o, n in zip(old_inputs, op._inputs[1:])}
        op.args = replace_objects(op.args, mapping)
        op.kwargs = replace_objects(op.kwargs, mapping)

    @property
    def output_limit(self) -> int:
        return len(self._output_types)

    def __call__(self, t, output_kws, args=None, **kwargs):
        self.args = args or ()
        self.kwargs = kwargs
        inputs = (
            [t]
            + find_objects(self.args, ENTITY_TYPE)
            + find_objects(self.kwargs, ENTITY_TYPE)
        )
        return self.new_tileables(inputs, kws=output_kws)


def to_remote_model(model, model_cls: Type[ModelWithEval]) -> ModelWithEval:
    op = ModelDataSource(data=model)
    return op(model_cls)
