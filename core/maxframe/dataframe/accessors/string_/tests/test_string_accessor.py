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

from ....core import OutputType
from ....datasource.series import from_pandas as from_pandas_series


def test_string_method():
    s = pd.Series(["a", "b", "c"], name="s")
    series = from_pandas_series(s, chunk_size=2)

    with pytest.raises(AttributeError):
        _ = series.str.non_exist

    r = series.str.contains("c")
    assert r.dtype == np.bool_
    assert r.name == s.name
    pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
    assert r.shape == s.shape

    r = series.str.split(",", expand=True, n=1)
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.shape == (3, 2)
    pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
    pd.testing.assert_index_equal(r.columns_value.to_pandas(), pd.RangeIndex(2))

    with pytest.raises(TypeError):
        _ = series.str.cat([["1", "2"]])

    with pytest.raises(ValueError):
        _ = series.str.cat(["1", "2"])

    with pytest.raises(ValueError):
        _ = series.str.cat(",")

    with pytest.raises(TypeError):
        _ = series.str.cat({"1", "2", "3"})

    r = series.str.cat(sep=",")
    assert r.op.output_types[0] == OutputType.scalar
    assert r.dtype == s.dtype

    r = series.str.extract(r"[ab](\d)", expand=False)
    assert r.op.output_types[0] == OutputType.series
    assert r.dtype == s.dtype

    r = series.str.extract(r"[ab](\d)", expand=True)
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.shape == (3, 1)
    pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
    pd.testing.assert_index_equal(r.columns_value.to_pandas(), pd.RangeIndex(1))

    assert "lstrip" in dir(series.str)

    r = series.str[1:10:2]
    assert r.op.method == "slice"
    assert r.op.method_args == ()
    assert r.op.method_kwargs == {"start": 1, "stop": 10, "step": 2}
