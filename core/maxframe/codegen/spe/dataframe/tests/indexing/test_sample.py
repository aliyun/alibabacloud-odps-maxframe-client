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

from ....core import SPECodeContext
from ...indexing import DataFrameSampleAdapter


def test_dataframe_sample_size(s1):
    df = s1.sample(n=2)
    adapter = DataFrameSampleAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    assert results[0] == "var_1_rs = np.random.RandomState()"
    assert results[1].startswith("var_1_rs.set_state(")
    assert results[2] == (
        "var_1 = var_0.sample(n=2, frac=None, replace=False, weights=None,"
        " axis=0, random_state=var_1_rs)"
    )


def test_dataframe_sample_frac(s1):
    s1._shape = (np.NAN, np.NAN)
    df = s1.sample(frac=1.1, weights=[0.1, 0.2, 0.3, 0.4], replace=True, random_state=1)
    adapter = DataFrameSampleAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    assert results[0] == "var_1_rs = np.random.RandomState()"
    assert results[1].startswith("var_1_rs.set_state(")
    assert results[2] == (
        "var_1 = var_0.sample(n=None, frac=1.1, replace=True,"
        " weights=[0.1, 0.2, 0.3, 0.4], axis=0, random_state=var_1_rs)"
    )
