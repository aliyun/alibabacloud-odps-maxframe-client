# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

import numpy as np
import pytest

import maxframe.tensor as mt
from maxframe.learn.contrib.lightgbm.dataset import to_lgbm_dataset


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "auto"),
        ({"categorical_feature": "auto"}, "auto"),
        ({"categorical_feature": None}, None),
        ({"categorical_feature": ["cat_a", "cat_b"]}, ["cat_a", "cat_b"]),
        ({"categorical_feature": [0, 2]}, [0, 2]),
    ],
    ids=["default", "auto", "none", "names", "indices"],
)
def test_dataset_keeps_categorical_feature(kwargs, expected):
    # "auto" (infer from pandas category dtypes) and None (no categorical
    # feature) are different lightgbm values, so the operator must not merge them
    data = mt.tensor(np.random.rand(10, 3))

    assert to_lgbm_dataset(data, **kwargs).op.categorical_feature == expected
