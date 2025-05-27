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

from ..... import dataframe as md
from ...core import SPECodeContext
from ..value_counts import DataFrameValueCountsAdapter


@pytest.fixture
def s1():
    return md.Series([1, 2, 3], index=list("abc"))


def test_value_counts_drop_na(s1):
    s = s1.value_counts(dropna=True)
    adapter = DataFrameValueCountsAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s.op, context)
    expected_results = [
        "var_1 = var_0.value_counts("
        "ascending=False, bins=None, dropna=True, normalize=False, sort=True)"
    ]
    assert results == expected_results


def test_value_counts_asc_order(s1):
    s = s1.value_counts(dropna=False, sort=True, ascending=True)
    adapter = DataFrameValueCountsAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s.op, context)
    expected_results = [
        "var_1 = var_0.value_counts("
        "ascending=True, bins=None, dropna=False, normalize=False, sort=True)"
    ]
    assert results == expected_results


def test_value_counts_normalize(s1):
    s = s1.value_counts(normalize=True)
    adapter = DataFrameValueCountsAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(s.op, context)
    expected_results = [
        "var_1 = var_0.value_counts("
        "ascending=False, bins=None, dropna=True, normalize=True, sort=True)"
    ]
    assert results == expected_results
