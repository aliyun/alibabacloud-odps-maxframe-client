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

from .....core import OutputType
from ....datasource.series import from_pandas as from_pandas_series


def test_datetime_method():
    s = pd.Series(
        [pd.Timestamp("2020-1-1"), pd.Timestamp("2020-2-1"), pd.Timestamp("2020-3-1")],
        name="ss",
    )
    series = from_pandas_series(s, chunk_size=2)

    r = series.dt.year
    assert r.dtype == s.dt.year.dtype
    pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
    assert r.shape == s.shape
    assert r.op.output_types[0] == OutputType.series
    assert r.name == s.dt.year.name

    with pytest.raises(AttributeError):
        _ = from_pandas_series(pd.Series([1])).dt
    with pytest.raises(AttributeError):
        _ = series.dt.non_exist

    assert "ceil" in dir(series.dt)
