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

import inspect
from typing import Callable, List, Union

from ....serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FieldTypes,
    Int32Field,
)
from ....udf import BuiltinFunction
from ..utils import TrainingCallback

try:
    from lightgbm.callback import _EarlyStoppingCallback, _ResetParameterCallback

    _use_lgbm_callback_classes = True
except ImportError:
    try:
        from lightgbm.callback import early_stopping as _EarlyStoppingCallback
        from lightgbm.callback import reset_parameter as _ResetParameterCallback

        _use_lgbm_callback_classes = False
    except ImportError:
        _EarlyStoppingCallback = _ResetParameterCallback = None


class LGBMTrainingCallback(TrainingCallback):
    _local_to_remote = {}

    @classmethod
    def _load_local_to_remote_mapping(cls, globals_dict):
        if isinstance(_EarlyStoppingCallback, type):
            return super()._load_local_to_remote_mapping(globals_dict)
        # lightgbm<4.0: callbacks returning closures
        for v in globals_dict.values():
            if isinstance(v, type) and issubclass(v, cls) and v._local_cls is not None:
                cls._local_to_remote[v._local_cls.__name__] = v

    @classmethod
    def from_local(cls, callback_obj):
        cls._load_local_to_remote_mapping(globals())
        if _use_lgbm_callback_classes:
            return super().from_local(callback_obj)
        cb_name = callback_obj.__qualname__.split(".", 1)[0]
        if cb_name not in cls._local_to_remote:
            return callback_obj

        remote_cls = cls._local_to_remote[cb_name]
        closure_vars = inspect.getclosurevars(callback_obj).nonlocals
        kw = {k: v for k, v in closure_vars.items() if k in remote_cls._FIELDS}
        return remote_cls(**kw)


class EarlyStoppingCallback(LGBMTrainingCallback):
    _local_cls = _EarlyStoppingCallback

    stopping_rounds = Int32Field("stopping_rounds")
    first_metric_only = BoolField("first_metric_only")
    verbose = BoolField("verbose")
    min_delta = AnyField("min_delta")

    def _extract_kw(self) -> dict:
        kw = super()._extract_kw()
        if not _use_lgbm_callback_classes:
            kw.pop("min_delta", None)
        return kw


def early_stopping(
    stopping_rounds: int,
    first_metric_only: bool = False,
    verbose: bool = True,
    min_delta: Union[float, List[float]] = 0.0,
) -> EarlyStoppingCallback:
    return EarlyStoppingCallback(
        stopping_rounds=stopping_rounds,
        first_metric_only=first_metric_only,
        verbose=verbose,
        min_delta=min_delta,
    )


class ResetParameterCallback(LGBMTrainingCallback):
    _local_cls = _ResetParameterCallback

    kwargs = DictField("kwargs", key_type=FieldTypes.string, default_factory=dict)

    def has_custom_code(self) -> bool:
        return any(
            not isinstance(v, (list, BuiltinFunction)) for v in self.kwargs.values()
        )

    def _extract_kw(self) -> dict:
        # _ResetParameterCallback in lightgbm accepts kwargs as **kwargs
        return self.kwargs


def reset_parameter(**kwargs: Union[list, Callable]) -> Callable:
    return ResetParameterCallback(kwargs=kwargs)
