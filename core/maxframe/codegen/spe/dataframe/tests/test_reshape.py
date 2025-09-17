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

from ..... import dataframe as md
from ...core import SPECodeContext
from ..reshape import (
    DataFrameMeltAdapter,
    DataFramePivotTableAdapter,
    DataFrameStackAdapter,
)


@pytest.fixture
def df1():
    return md.DataFrame(
        np.random.randn(6, 4),
        index=md.Index(["1", "2", "3", "4", "5", "6"], name="test_idx"),
        columns=list("ABCD"),
    )


@pytest.fixture
def midx_df1():
    return md.DataFrame(
        [
            [1, 10, 101],
            [9, 90, 909],
            [2, 20, 202],
        ],
        index=pd.MultiIndex.from_arrays(
            [[1, 9, 2], [11, 99, 22], [111, 999, 222]],
            names=("idx_0", "idx_1", "idx_2"),
        ),
        columns=list("ABC"),
    )


def test_melt(df1):
    context = SPECodeContext()

    v1 = df1.melt(id_vars=["A"], value_vars=["B"])
    results = DataFrameMeltAdapter().generate_code(v1.op, context)
    assert (
        results[0]
        == "var_1 = var_0.melt(id_vars=['A'], value_vars=['B'], value_name='value')"
    )


def test_pivot_table(midx_df1):
    df = midx_df1.pivot_table(columns="B", values="C")
    context = SPECodeContext()
    results = DataFramePivotTableAdapter().generate_code(df.op, context)
    assert results[0] == (
        "var_1 = var_0.pivot_table(values='C', columns='B', "
        "aggfunc='mean', margins=False, dropna=True, margins_name='All', "
        "sort=True)"
    )


def test_stack(midx_df1):
    df = midx_df1.stack([0])
    context = SPECodeContext()
    results = DataFrameStackAdapter().generate_code(df.op, context)
    assert results[0] == "var_1 = var_0.stack(level=[0], dropna=True)"
