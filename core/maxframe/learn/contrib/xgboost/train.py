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

import logging
from collections import OrderedDict
from typing import List

from .... import opcodes
from ....core import EntityData, OutputType
from ....core.operator import ObjectOperator, ObjectOperatorMixin
from ....serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FieldTypes,
    FunctionField,
    Int16Field,
    Int64Field,
    KeyField,
    ListField,
)
from ..models import to_remote_model
from ..utils import TrainingCallback
from .core import Booster, BoosterData, XGBScikitLearnBase
from .dmatrix import ToDMatrix, to_dmatrix

logger = logging.getLogger(__name__)


def _on_serialize_evals(evals_val):
    if evals_val is None:
        return None
    return [list(x) for x in evals_val]


class XGBTrain(ObjectOperator, ObjectOperatorMixin):
    _op_type_ = opcodes.XGBOOST_TRAIN

    params = DictField("params", key_type=FieldTypes.string, default=None)
    dtrain = KeyField("dtrain", default=None)
    evals = ListField("evals", on_serialize=_on_serialize_evals, default=None)
    obj = FunctionField("obj", default=None)
    feval = FunctionField("feval", default=None)
    maximize = BoolField("maximize", default=None)
    early_stopping_rounds = Int64Field("early_stopping_rounds", default=None)
    verbose_eval = AnyField("verbose_eval", default=None)
    xgb_model = KeyField("xgb_model", default=None)
    callbacks = ListField(
        "callbacks",
        field_type=FunctionField.field_type,
        default=None,
        on_serialize=TrainingCallback.from_local,
    )
    custom_metric = FunctionField("custom_metric", default=None)
    num_boost_round = Int64Field("num_boost_round", default=10)
    num_class = Int64Field("num_class", default=None)
    _has_evals_result = BoolField("has_evals_result", default=False)
    output_ndim = Int16Field("output_ndim", default=None)

    def __init__(self, gpu=None, **kw):
        if kw.get("evals_result") is not None:
            kw["_has_evals_result"] = True
        super().__init__(gpu=gpu, **kw)
        if self.output_types is None:
            self.output_types = [OutputType.object]
        if self.has_evals_result:
            self.output_types.append(OutputType.object)

    def has_custom_code(self) -> bool:
        if not self.callbacks:
            return False
        return any(
            not isinstance(cb, TrainingCallback) or cb.has_custom_code()
            for cb in self.callbacks
        )

    @classmethod
    def _set_inputs(cls, op: "XGBTrain", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        input_it = iter(op._inputs)
        op.dtrain = next(input_it)
        if op.evals:
            evals_dict = OrderedDict(op.evals)
            new_evals_dict = OrderedDict()
            for val in evals_dict.values():
                new_key = next(input_it)
                new_evals_dict[new_key] = val
            op.evals = list(new_evals_dict.items())
        if op.xgb_model:
            op.xgb_model = next(input_it)

    def __call__(self, evals_result):
        inputs = [self.dtrain]
        if self.has_evals_result:
            inputs.extend(e[0] for e in self.evals)
        if self.xgb_model is not None:
            inputs.append(self.xgb_model)
        kws = [{"object_class": Booster}, {}]
        return self.new_tileables(inputs, kws=kws, evals_result=evals_result)[0]

    @property
    def output_limit(self):
        return 2 if self.has_evals_result else 1

    @property
    def has_evals_result(self) -> bool:
        return self._has_evals_result or self.evals


def _get_xgb_booster(xgb_model):
    import xgboost

    if isinstance(xgb_model, (XGBScikitLearnBase, xgboost.XGBModel)):
        xgb_model = xgb_model.get_booster()

    if isinstance(xgb_model, (Booster, BoosterData)):
        return xgb_model
    elif isinstance(xgb_model, xgboost.Booster):
        return to_remote_model(xgb_model, model_cls=Booster)
    raise ValueError(f"Cannot use {type(xgb_model)} as xgb_model")


def train(
    params,
    dtrain,
    evals=None,
    evals_result=None,
    xgb_model=None,
    num_class=None,
    **kwargs,
):
    """
    Train XGBoost model in MaxFrame manner.

    Parameters
    ----------
    Parameters are the same as `xgboost.train`. Note that train is an eager-execution
    API if evals is passed, thus the call will be blocked until training finished.

    Returns
    -------
    results: Booster
    """

    evals_result = evals_result if evals_result is not None else dict()
    processed_evals = []
    session = kwargs.pop("session", None)
    run_kwargs = kwargs.pop("run_kwargs", dict())
    if evals:
        for eval_dmatrix, name in evals:
            if not isinstance(name, str):
                raise TypeError("evals must a list of pairs (DMatrix, string)")
            if hasattr(eval_dmatrix, "op") and isinstance(eval_dmatrix.op, ToDMatrix):
                processed_evals.append((eval_dmatrix, name))
            else:
                processed_evals.append((to_dmatrix(eval_dmatrix), name))
    if xgb_model is not None:
        xgb_model = _get_xgb_booster(xgb_model)
    data = XGBTrain(
        params=params,
        dtrain=dtrain,
        evals=processed_evals,
        evals_result=evals_result,
        xgb_model=xgb_model,
        num_class=num_class,
        **kwargs,
    )(evals_result)
    if evals:
        data.execute(session=session, **run_kwargs)
    return data
