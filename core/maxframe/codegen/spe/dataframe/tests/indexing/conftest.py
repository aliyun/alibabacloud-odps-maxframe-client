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

import numpy as np
import pandas as pd
import pytest

from ...... import dataframe as md


@pytest.fixture
def multi_index():
    arrays = [["m1", "m2", "m3"], ["n1", "n2", "n3"]]
    return pd.MultiIndex.from_arrays(arrays, names=("idx_1", "idx_2"))


@pytest.fixture
def df1():
    return md.DataFrame(
        np.random.randn(6, 4),
        index=md.Index(["1", "2", "3", "4", "5", "6"], name="test_idx"),
        columns=list("ABCD"),
    )


@pytest.fixture
def df2():
    return md.DataFrame(
        np.random.randn(6, 4),
        index=md.Index(["6", "5", "4", "3", "2", "1"], name="test_idx"),
        columns=list("ABCD"),
    )


@pytest.fixture
def s1():
    return md.Series([1, 2, 3], index=list("abc"))


@pytest.fixture()
def s2(multi_index):
    return md.Series([1, 2, 3], index=multi_index)


@pytest.fixture
def idx1(multi_index):
    return md.Index(multi_index)
