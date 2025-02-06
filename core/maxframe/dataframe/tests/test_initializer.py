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

from ..core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE
from ..initializer import DataFrame, Series, read_pandas


def test_read_pandas():
    df_data = pd.DataFrame([["a", 1], ["b", 2]], columns=["a", "b"])
    assert isinstance(read_pandas(df_data), DATAFRAME_TYPE)

    s_data = pd.Series(["a", "b"])
    assert isinstance(read_pandas(s_data), SERIES_TYPE)

    idx_data = pd.Index(["a", "b"])
    assert isinstance(read_pandas(idx_data), INDEX_TYPE)


def test_init_dataframe_from_maxframe_series():
    s = Series([1, 2, 3, 4], index=[1, 2, 3, 4])

    df = DataFrame(s, index=s.index, columns=["col1"])

    assert isinstance(df, DATAFRAME_TYPE)
    assert df.dtypes.index == ["col1"]

    with pytest.raises(ValueError):
        DataFrame(s, index=s.index, columns=[])

    with pytest.raises(ValueError):
        DataFrame(s, index=s.index, columns="col1")

    with pytest.raises(ValueError):
        DataFrame(s, index=s.index, columns="col2")


def test_init_dataframe_from_maxframe_dataframe():
    df1 = DataFrame({"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]}, index=[1, 2, 3, 4])

    df2 = DataFrame(df1, index=df1.index, columns=["col1", "col2"])

    assert isinstance(df2, DATAFRAME_TYPE)
    assert list(df2.dtypes.index) == ["col1", "col2"]

    with pytest.raises(ValueError):
        DataFrame(df1, index=df1.index, columns=["col1", "col2", "col3"])
