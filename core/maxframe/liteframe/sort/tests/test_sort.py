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

import pandas as pd
import pytest

from maxframe.liteframe.core import RANGE_COL_NAME
from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.initializer import LiteFrame as LF
from maxframe.liteframe.operators.sort import LiteFrameSort
from maxframe.protocol import DefaultIndexType


@pytest.mark.parametrize("column", ["NoSuchCol", "MixedCase"])
def test_sort_invalid_column_raises_key_error(column):
    """sort() with a non-existent column name raises KeyError with the original name."""
    lf = LF({"a": [3, 1, 2], "b": [1, 2, 3]})
    with pytest.raises(KeyError, match=column):
        lf.sort(by=column)


def test_sort_invalid_keep_kind_raises_value_error():
    """LiteFrameSort with an invalid keep_kind raises ValueError."""
    lf = LF({"a": [3, 1, 2]})
    op = LiteFrameSort(by=["a"], keep_kind="invalid")
    with pytest.raises(ValueError, match="Invalid keep_kind"):
        op(lf)


@pytest.mark.parametrize("na_position", ["middle", "first_last", ""])
def test_sort_invalid_na_position_raises_value_error(na_position):
    """sort() with an invalid na_position raises ValueError."""
    lf = LF({"a": [3, 1, 2]})
    with pytest.raises(ValueError, match="Invalid na_position"):
        lf.sort(by="a", na_position=na_position)


@pytest.mark.parametrize("psrs_kinds", [["a", "b"], ["a"], ["a", "b", "c", "d"]])
def test_sort_invalid_psrs_kinds_raises_value_error(psrs_kinds):
    """sort() with psrs_kinds of wrong length raises ValueError."""
    lf = LF({"a": [3, 1, 2]})
    with pytest.raises(ValueError, match="psrs_kinds must have exactly 3 elements"):
        lf.sort(by="a", psrs_kinds=psrs_kinds)


def test_sort_invalid_parallel_kind_raises_value_error():
    """sort() with an unsupported parallel_kind raises ValueError."""
    lf = LF({"a": [3, 1, 2]})
    with pytest.raises(ValueError, match="parallel_kind.*is not supported"):
        lf.sort(by="a", parallel_kind="other")


def test_sort_ascending_length_mismatch_raises_value_error():
    """sort() with ascending list length != by length raises ValueError."""
    lf = LF({"a": [3, 1, 2], "b": [1, 2, 3]})
    with pytest.raises(
        ValueError, match="ascending length 1 does not match by length 2"
    ):
        lf.sort(by=["a", "b"], ascending=[True])


def test_sort_metadata_column_order_with_range_columns():
    """sort() on a LiteFrame with range columns places range columns first
    in physical_dtypes, matching the tiler's materialization order.
    """
    raw = pd.DataFrame({"a": [3, 1, 2], "b": [1, 2, 3]})
    lf = from_local_df(raw, default_index_type=DefaultIndexType.range)
    result = lf.sort(by="a")

    dtypes_index = list(result._data._physical_dtypes.index)
    assert dtypes_index[0] == RANGE_COL_NAME
    assert set(dtypes_index[1:]) == {"a", "b"}
