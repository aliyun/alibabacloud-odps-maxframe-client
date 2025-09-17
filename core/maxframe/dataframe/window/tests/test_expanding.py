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

from .... import dataframe as md


def test_expanding():
    df = pd.DataFrame(np.random.rand(4, 3), columns=list("abc"))
    df2 = md.DataFrame(df)

    r = df2.expanding(3)
    expected = df.expanding(3)
    assert repr(r).split(",", 1)[0] == repr(expected).split(",", 1)[0]

    assert "b" in dir(r)

    with pytest.raises(AttributeError):
        _ = r.d

    with pytest.raises(KeyError):
        _ = r["d"]

    with pytest.raises(KeyError):
        _ = r["a", "d"]

    assert "a" not in dir(r.a)
    assert "c" not in dir(r["a", "b"])


def test_expanding_agg():
    df = pd.DataFrame(np.random.rand(4, 3), columns=list("abc"))
    df2 = md.DataFrame(df, chunk_size=3)

    r = df2.expanding(3).agg("max")
    expected = df.expanding(3).agg("max")

    assert r.shape == df.shape
    assert r.index_value is df2.index_value
    pd.testing.assert_index_equal(r.columns_value.to_pandas(), expected.columns)
    pd.testing.assert_series_equal(r.dtypes, df2.dtypes)

    aggs = ["sum", "count", "min", "max", "mean", "var", "std"]
    for a in aggs:
        r = getattr(df2.expanding(3), a)()
        assert r.op.func == a
