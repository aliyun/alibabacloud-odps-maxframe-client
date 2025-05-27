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

import pandas as pd
import pytest

from ... import DataFrame, Index
from ..to_odps import to_odps_table


@pytest.fixture
def df():
    return DataFrame({"A": [1, 2], "B": [3, 4]})


@pytest.fixture
def df_with_named_index():
    return DataFrame({"A": [1, 2], "B": [3, 4]}, index=Index([1, 2], name="A"))


@pytest.fixture
def df_with_named_multi_indexes():
    arrays = [
        ["c1", "c2"],
        ["d1", "d2"],
        ["e1", "e2"],
    ]
    multi_index = pd.MultiIndex.from_arrays(arrays, names=("C", "D", "E"))
    return DataFrame(
        {"A": [1, 2], "B": [3, 4]},
        index=multi_index,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"partition_col": ["A", "C"]},
        {"partition_col": "C"},
        {"partition": "a=1,C=2"},
    ],
)
def test_to_odps_table_validation(df, kwargs):
    with pytest.raises(ValueError):
        to_odps_table(df, "test_table", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"partition_col": ["a", "B"]},
        {"partition_col": "a"},
        {"partition": "C=1,d=2"},
    ],
)
def test_to_odps_table_vaild(df, kwargs):
    to_odps_table(df, "test_table", **kwargs)


def test_to_odps_table_column_conflicts(
    df, df_with_named_index, df_with_named_multi_indexes
):
    to_odps_table(df.reset_index(), "test_table", index=False)
    to_odps_table(df.reset_index(), "test_table", index_label="C")
    with pytest.raises(ValueError):
        to_odps_table(df.reset_index(), "test_table")

    to_odps_table(df_with_named_index, "test_table", index=False)
    to_odps_table(df_with_named_index, "test_table", index_label="C")
    with pytest.raises(ValueError):
        to_odps_table(df_with_named_index, "test_table")

    to_odps_table(df, "test_table", partition="C='1'")
    with pytest.raises(ValueError):
        to_odps_table(df, "test_table", partition="A='1'")

    with pytest.raises(ValueError):
        to_odps_table(df, "test_table", partition="A='1'")

    to_odps_table(df_with_named_multi_indexes, "test_table")
    to_odps_table(
        df_with_named_multi_indexes, "test_table", partition="C='1'", index=False
    )
    with pytest.raises(ValueError):
        to_odps_table(df_with_named_multi_indexes, "test_table", partition="C='1'")

    df_with_named_multi_indexes.index.names = ["C1", "D1", "E1"]
    to_odps_table(df_with_named_multi_indexes, "test_table", partition="C='1'")
