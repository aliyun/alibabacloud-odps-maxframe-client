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

import random

import numpy as np
import pandas as pd
import pytest

from .... import dataframe as md
from ...core import MultiIndex
from .. import isna, isnull, notna, notnull


def test_fill_na():
    df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list("ABCDEFGHIJ"))
    for _ in range(20):
        df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
    series_raw = pd.Series(np.nan, index=range(20))
    for _ in range(3):
        series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)

    df = md.DataFrame(df_raw)
    series = md.Series(series_raw)

    # when nothing supplied, raise
    with pytest.raises(ValueError):
        df.fillna()
    # when both values and methods supplied, raises
    with pytest.raises(ValueError):
        df.fillna(value=1, method="ffill")
    # when call on series, cannot supply DataFrames
    with pytest.raises(ValueError):
        series.fillna(value=df)
    with pytest.raises(ValueError):
        series.fillna(value=df_raw)
    with pytest.raises(NotImplementedError):
        series.fillna(value=series_raw, downcast="infer")
    with pytest.raises(NotImplementedError):
        series.ffill(limit=1)


@pytest.mark.parametrize("check_func", [isna, isnull, notna, notnull])
def test_isna(check_func):
    # multi index
    with pytest.raises(NotImplementedError):
        midx = MultiIndex()
        check_func(midx)


def test_drop_na():
    # dataframe cases
    df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list("ABCDEFGHIJ"))
    for _ in range(30):
        df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
    for _ in range(random.randint(1, 5)):
        row = random.randint(0, 19)
        for idx in range(0, 10):
            df_raw.iloc[row, idx] = random.randint(0, 99)

    # not supporting drop with axis=1
    with pytest.raises(NotImplementedError):
        md.DataFrame(df_raw).dropna(axis=1)


def test_replace():
    # dataframe cases
    df_raw = pd.DataFrame(-1, index=range(0, 20), columns=list("ABCDEFGHIJ"))
    for _ in range(30):
        df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
    for _ in range(random.randint(1, 5)):
        row = random.randint(0, 19)
        for idx in range(0, 10):
            df_raw.iloc[row, idx] = random.randint(0, 99)

    # not supporting fill with limit
    df = md.DataFrame(df_raw, chunk_size=4)
    with pytest.raises(NotImplementedError):
        df.replace(-1, method="ffill", limit=5)
