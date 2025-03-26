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

from .... import opcodes
from ....core import OutputType
from ....core.operator.base import Operator
from ....core.operator.core import TileableOperatorMixin
from ....serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FieldTypes,
    FunctionField,
    Int64Field,
    KeyField,
    ListField,
)
from .core import Booster
from .dmatrix import ToDMatrix, to_dmatrix

logger = logging.getLogger(__name__)


def _on_serialize_evals(evals_val):
    if evals_val is None:
        return None
    return [list(x) for x in evals_val]


class XGBTrain(Operator, TileableOperatorMixin):
    _op_type_ = opcodes.XGBOOST_TRAIN

    params = DictField("params", key_type=FieldTypes.string, default=None)
    dtrain = KeyField("dtrain", default=None)
    evals = ListField("evals", on_serialize=_on_serialize_evals, default=None)
    obj = FunctionField("obj", default=None)
    feval = FunctionField("obj", default=None)
    maximize = BoolField("maximize", default=None)
    early_stopping_rounds = Int64Field("early_stopping_rounds", default=None)
    verbose_eval = AnyField("verbose_eval", default=None)
    xgb_model = AnyField("xgb_model", default=None)
    callbacks = ListField(
        "callbacks", field_type=FunctionField.field_type, default=None
    )
    custom_metric = FunctionField("custom_metric", default=None)
    num_boost_round = Int64Field("num_boost_round", default=10)
    num_class = Int64Field("num_class", default=None)

    def __init__(self, gpu=None, **kw):
        super().__init__(gpu=gpu, **kw)
        if self.output_types is None:
            self.output_types = [OutputType.object]
        if self.has_evals_result:
            self.output_types.append(OutputType.object)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self.dtrain = self._inputs[0]
        rest = self._inputs[1:]
        if self.has_evals_result:
            evals_dict = OrderedDict(self.evals)
            new_evals_dict = OrderedDict()
            for new_key, val in zip(rest, evals_dict.values()):
                new_evals_dict[new_key] = val
            self.evals = list(new_evals_dict.items())

    def __call__(self, evals_result):
        inputs = [self.dtrain]
        if self.has_evals_result:
            inputs.extend(e[0] for e in self.evals)
        kws = [{"object_class": Booster}, {}]
        return self.new_tileables(inputs, kws=kws, evals_result=evals_result)[0]

    @property
    def output_limit(self):
        return 2 if self.has_evals_result else 1

    @property
    def has_evals_result(self) -> bool:
        return self.evals


def train(params, dtrain, evals=None, evals_result=None, num_class=None, **kwargs):
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
    data = XGBTrain(
        params=params,
        dtrain=dtrain,
        evals=processed_evals,
        evals_result=evals_result,
        num_class=num_class,
        **kwargs,
    )(evals_result)
    if evals:
        data.execute(session=session, **run_kwargs)
    return data
