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

import pytest

from ..callback import EarlyStopping, LearningRateScheduler, XGBTrainingCallback

try:
    from xgboost.callback import EarlyStopping as _EarlyStopping
except ImportError:
    pytestmark = pytest.mark.skip("Need xgboost to run the test")


def test_xgb_training_callbacks():
    early_stopping_cb = EarlyStopping(rounds=5)
    assert not early_stopping_cb.has_custom_code()

    local_early_stopping_cb = early_stopping_cb.to_local()
    assert isinstance(local_early_stopping_cb, _EarlyStopping)
    assert local_early_stopping_cb.rounds == 5

    rmt_early_stopping_cb = XGBTrainingCallback.from_local(local_early_stopping_cb)
    assert isinstance(rmt_early_stopping_cb, EarlyStopping)
    assert rmt_early_stopping_cb.rounds == 5

    lr_sch_cb = LearningRateScheduler([0.5, 0.4, 0.1])
    assert not lr_sch_cb.has_custom_code()

    lr_sch_cb = LearningRateScheduler(lambda x: x * 0.1)
    assert lr_sch_cb.has_custom_code()
