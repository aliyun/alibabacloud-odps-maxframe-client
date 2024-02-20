# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

from .... import dataframe as md
from ...core import IndexValue
from ..reshuffle import DataFrameReshuffle


def test_reshuffle():
    df = pd.DataFrame(
        {"a": [3, 4, 5, 3, 5, 4, 1, 2, 3], "b": [1, 3, 4, 5, 6, 5, 4, 4, 4]}
    )
    mdf = md.DataFrame(df, chunk_size=2)

    with pytest.raises(ValueError):
        mdf.mf.reshuffle(sort_by="b")

    r = mdf.mf.reshuffle()
    assert isinstance(r.op, DataFrameReshuffle)
    assert isinstance(r.index_value.value, IndexValue.Int64Index)

    r = mdf.mf.reshuffle(ignore_index=True)
    assert isinstance(r.op, DataFrameReshuffle)
    assert isinstance(r.index_value.value, IndexValue.RangeIndex)
