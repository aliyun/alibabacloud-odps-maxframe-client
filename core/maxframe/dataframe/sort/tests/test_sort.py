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

from ...initializer import DataFrame
from ..sort_index import DataFrameSortIndex, sort_index
from ..sort_values import DataFrameSortValues, dataframe_sort_values


def test_sort_values():
    raw = pd.DataFrame(
        {
            "a": np.random.rand(10),
            "b": np.random.randint(1000, size=10),
            "c": np.random.rand(10),
            "d": [np.random.bytes(10) for _ in range(10)],
            "e": [pd.Timestamp(f"201{i}") for i in range(10)],
            "f": [pd.Timedelta(f"{i} days") for i in range(10)],
        },
    )
    df = DataFrame(raw)
    sorted_df = dataframe_sort_values(df, by="c")

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortValues)

    df = DataFrame(raw, chunk_size=6)
    sorted_df = dataframe_sort_values(df, by="c")

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortValues)

    df = DataFrame(raw, chunk_size=3)
    sorted_df = dataframe_sort_values(df, by=["a", "c"])

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortValues)


def test_sort_index():
    raw = pd.DataFrame(
        np.random.rand(10, 10), columns=np.random.rand(10), index=np.random.rand(10)
    )
    df = DataFrame(raw)
    sorted_df = sort_index(df)

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortIndex)
    assert sorted_df.op.level is None

    df = DataFrame(raw, chunk_size=6)
    sorted_df = sort_index(df)

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortIndex)

    df = DataFrame(raw, chunk_size=3)
    sorted_df = sort_index(df)

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortIndex)

    # support on axis 1
    df = DataFrame(raw, chunk_size=4)
    sorted_df = sort_index(df, axis=1)

    assert sorted_df.shape == raw.shape
    assert isinstance(sorted_df.op, DataFrameSortIndex)
