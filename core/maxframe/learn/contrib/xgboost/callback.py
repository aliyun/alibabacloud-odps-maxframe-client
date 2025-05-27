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

from typing import Callable, Optional, Sequence, Union

from ....serialization.serializables import (
    AnyField,
    BoolField,
    Float32Field,
    Int32Field,
    StringField,
)
from ....udf import BuiltinFunction
from ..utils import TrainingCallback

try:
    from xgboost.callback import EarlyStopping as _EarlyStopping
    from xgboost.callback import LearningRateScheduler as _LearningRateScheduler
except ImportError:
    _LearningRateScheduler = _EarlyStopping = None


class XGBTrainingCallback(TrainingCallback):
    _local_to_remote = {}

    @classmethod
    def from_local(cls, callback_obj):
        cls._load_local_to_remote_mapping(globals())
        return super().from_local(callback_obj)


class LearningRateScheduler(XGBTrainingCallback):
    _local_cls = _LearningRateScheduler

    learning_rates = AnyField("learning_rates", default=None)

    def __init__(
        self, learning_rates: Union[Callable[[int], float], Sequence[float]], **kw
    ) -> None:
        super().__init__(learning_rates=learning_rates, **kw)

    def has_custom_code(self) -> bool:
        return not isinstance(self.learning_rates, (tuple, list, BuiltinFunction))


class EarlyStopping(XGBTrainingCallback):
    _local_cls = _EarlyStopping

    rounds = Int32Field("rounds")
    metric_name = StringField("metric_name", default=None)
    data_name = StringField("data_name", default=None)
    maximize = BoolField("maximize", default=None)
    save_best = BoolField("save_best", default=None)
    min_delta = Float32Field("min_delta", default=None)

    def __init__(
        self,
        *,
        rounds: int,
        metric_name: Optional[str] = None,
        data_name: Optional[str] = None,
        maximize: Optional[bool] = None,
        save_best: Optional[bool] = False,
        min_delta: float = 0.0,
        **kw
    ) -> None:
        super().__init__(
            rounds=rounds,
            metric_name=metric_name,
            data_name=data_name,
            maximize=maximize,
            save_best=save_best,
            min_delta=min_delta,
            **kw
        )
