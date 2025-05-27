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

import pytest

from ..callback import (
    EarlyStoppingCallback,
    LGBMTrainingCallback,
    early_stopping,
    reset_parameter,
)

try:
    from lightgbm.callback import _EarlyStoppingCallback
except ImportError:
    try:
        from lightgbm.callback import early_stopping as _EarlyStoppingCallback
    except ImportError:
        pytestmark = pytest.mark.skip("Need lightgbm to run the test")


def test_lgbm_training_callbacks():
    early_stopping_cb = early_stopping(stopping_rounds=5)
    assert not early_stopping_cb.has_custom_code()

    local_early_stopping_cb = early_stopping_cb.to_local()
    if isinstance(_EarlyStoppingCallback, type):
        assert isinstance(local_early_stopping_cb, _EarlyStoppingCallback)
        assert local_early_stopping_cb.stopping_rounds == 5
    else:
        assert local_early_stopping_cb.__qualname__.startswith(
            _EarlyStoppingCallback.__name__
        )
        cb_vars = inspect.getclosurevars(local_early_stopping_cb).nonlocals
        assert cb_vars["stopping_rounds"] == 5

    rmt_early_stopping_cb = LGBMTrainingCallback.from_local(local_early_stopping_cb)
    assert isinstance(rmt_early_stopping_cb, EarlyStoppingCallback)
    assert rmt_early_stopping_cb.stopping_rounds == 5

    lr_sch_cb = reset_parameter(a=[0.5, 0.4, 0.1])
    assert not lr_sch_cb.has_custom_code()

    lr_sch_cb = reset_parameter(a=lambda x: x * 0.1)
    assert lr_sch_cb.has_custom_code()
