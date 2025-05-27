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

import itertools
import logging
from typing import List

from .... import opcodes
from ....core import OutputType
from ....core.operator import ObjectOperator, ObjectOperatorMixin
from ....serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FieldTypes,
    FunctionField,
    Int32Field,
    KeyField,
    ListField,
    StringField,
)
from ....typing_ import EntityType
from ..models import to_remote_model
from ..utils import TrainingCallback
from .core import Booster, BoosterData, LGBMScikitLearnBase

logger = logging.getLogger(__name__)


class LGBMTrain(ObjectOperator, ObjectOperatorMixin):
    _op_type_ = opcodes.LGBM_TRAIN
    _op_module_ = "learn.contrib.lightgbm"

    params = DictField("params", key_type=FieldTypes.string, default_factory=dict)
    train_set = KeyField("train_set", default=None)
    num_boost_round = Int32Field("num_boost_round", default=None)
    valid_sets = ListField("valid_sets", FieldTypes.key, default_factory=list)
    valid_names = ListField("valid_names", FieldTypes.string, default_factory=list)
    feval = FunctionField("feval", default=None)
    init_model = KeyField("init_model", default=None)
    feature_name = AnyField("feature_name", default=None)
    categorical_feature = AnyField("categorical_feature", default=None)
    keep_training_booster = BoolField("keep_training_booster", default=False)
    callbacks = ListField(
        "callbacks",
        field_type=FunctionField.field_type,
        default=None,
        on_serialize=TrainingCallback.from_local,
    )
    tree_learner = StringField("tree_learner", default=None)
    timeout = Int32Field("timeout", default=None)
    # indicating shape of the predicted data of the model
    num_class = Int32Field("num_class", default=None)

    def __init__(self, gpu=None, **kw):
        super().__init__(gpu=gpu, **kw)
        if self.output_types is None:
            self.output_types = [OutputType.object]
        if self.has_evals_result:
            self.output_types.append(OutputType.object)

    @classmethod
    def _set_inputs(cls, op: "LGBMTrain", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        input_it = iter(op.inputs)
        op.train_set = next(input_it)
        op.valid_sets = list(itertools.islice(input_it, len(op.valid_sets or [])))
        if op.init_model is not None:
            op.init_model = next(input_it)

    def __call__(self, evals_result):
        inputs = [self.train_set]
        if self.has_evals_result:
            inputs.extend(self.valid_sets)
        if self.init_model is not None:
            inputs.append(self.init_model)
        kws = [{"object_class": Booster}, {}]
        return self.new_tileables(inputs, kws=kws, evals_result=evals_result)[0]

    @property
    def output_limit(self):
        return 2 if self.has_evals_result else 1

    def has_custom_code(self) -> bool:
        if not self.callbacks:
            return False
        return any(
            not isinstance(cb, TrainingCallback) or cb.has_custom_code()
            for cb in self.callbacks
        )

    @property
    def has_evals_result(self) -> bool:
        return bool(self.valid_sets)


def _get_lgbm_booster(init_model):
    import lightgbm

    if isinstance(init_model, (LGBMScikitLearnBase, lightgbm.LGBMModel)):
        init_model = init_model.booster_

    if isinstance(init_model, (Booster, BoosterData)):
        return init_model
    elif isinstance(init_model, lightgbm.Booster):
        return to_remote_model(init_model, model_cls=Booster)
    raise ValueError(f"Cannot use {type(init_model)} as init_model")


def train(
    params,
    train_set,
    num_boost_round=100,
    valid_sets=None,
    valid_names=None,
    feval=None,
    init_model=None,
    keep_training_booster=False,
    callbacks=None,
    num_class=2,
    evals_result=None,
    **kwargs,
):
    evals_result = evals_result if evals_result is not None else dict()
    timeout = kwargs.pop("timeout", 120)
    session = kwargs.pop("session", None)
    run_kwargs = kwargs.pop("run_kwargs", dict())
    if valid_sets and valid_names:
        if not isinstance(valid_names, list):
            raise TypeError("valid_names must be a list of strings")
        if len(valid_names) != len(valid_sets):
            raise ValueError("size of valid_names must match size of valid_sets")
    if init_model is not None:
        init_model = _get_lgbm_booster(init_model)
    data = LGBMTrain(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        feval=feval,
        init_model=init_model,
        keep_training_booster=keep_training_booster,
        callbacks=callbacks,
        num_class=num_class,
        evals_result=evals_result,
        timeout=timeout,
        **kwargs,
    )(evals_result)
    if valid_sets:
        data.execute(session=session, **run_kwargs)
    return data
