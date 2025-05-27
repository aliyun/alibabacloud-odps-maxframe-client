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

from ...... import dataframe as md
from ...... import tensor as mt
from ......learn.utils import shuffle
from ....core import SPECodeContext
from ..shuffle import LearnShuffleAdapter


def test_shuffle():
    X = mt.array([[1.0, 0.0], [2.0, 1.0], [0.0, 0.0]])
    y = mt.array([0, 1, 2])
    df = md.DataFrame([["a", 1], ["b", 2], ["c", 3]], columns=["a", "b"])

    rx, ry, rdf = shuffle(X, y, df, random_state=0)
    adapter = LearnShuffleAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(rx.op, context)
    expected_results = [
        "var_3_rs = np.random.RandomState(2357136044)",
        "var_3, var_4, var_5 = sk_shuffle("
        "var_0, var_1, var_2, random_state=var_3_rs)",
    ]
    assert results == expected_results

    rx, ry, rdf = shuffle(X, y, df, random_state=0, axes=(1,))
    adapter = LearnShuffleAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(rx.op, context)
    expected_results = [
        "var_0_rs1 = np.random.RandomState(2357136044)",
        "var_0_axis1 = var_0_rs1.randint(0, var_1.shape[1], var_1.shape[1])",
        "var_0 = var_1[:, var_0_axis1]",
        "var_3 = var_2[:]",
        "var_0_axis1 = var_0_rs1.randint(0, var_4.shape[1], var_4.shape[1])",
        "var_5 = var_4.iloc[:, var_0_axis1]",
    ]
    assert results == expected_results
