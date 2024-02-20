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

import numpy as np
import pandas as pd

from ..arrow import arrow_to_pandas, pandas_to_arrow


def test_dataframe_convert():
    pd_data = pd.DataFrame(np.random.rand(100, 5), columns=list("ABCDE"))
    arrow_data, meta = pandas_to_arrow(pd_data)
    assert arrow_data.column_names == ["_idx_0", "a", "b", "c", "d", "e"]

    pd.testing.assert_index_equal(pd.Index(arrow_data.columns[0]), pd_data.index)
    pd.testing.assert_frame_equal(
        arrow_data.select(list("abcde")).to_pandas().set_axis(list("ABCDE"), axis=1),
        pd_data.reset_index(drop=True),
    )

    pd_res = arrow_to_pandas(arrow_data, meta)
    pd.testing.assert_frame_equal(pd_data, pd_res)

    # test DataFrame with MultiIndex as columns
    pd_data.columns = pd.MultiIndex.from_tuples(
        [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B"), ("B", "C")]
    )
    arrow_data, meta = pandas_to_arrow(pd_data)
    pd_res = arrow_to_pandas(arrow_data, meta)
    pd.testing.assert_frame_equal(pd_data, pd_res)


def test_series_convert():
    pd_data = pd.Series(np.random.rand(100), name="series_name")
    arrow_data, meta = pandas_to_arrow(pd_data)
    assert arrow_data.column_names == ["_idx_0", "series_name"]

    pd.testing.assert_index_equal(pd.Index(arrow_data.columns[0]), pd_data.index)
    pd.testing.assert_series_equal(
        arrow_data.select(["series_name"]).to_pandas().iloc[:, 0],
        pd_data.reset_index(drop=True),
    )

    pd_res = arrow_to_pandas(arrow_data, meta)
    pd.testing.assert_series_equal(pd_data, pd_res)


def test_index_convert():
    pd_data = pd.Index(np.random.rand(100), name="idx_name")
    arrow_data, meta = pandas_to_arrow(pd_data)
    assert arrow_data.column_names == ["_idx_0"]

    pd.testing.assert_index_equal(
        pd.Index(arrow_data.columns[0], name="idx_name"), pd_data
    )

    pd_res = arrow_to_pandas(arrow_data, meta)
    pd.testing.assert_index_equal(pd_data, pd_res)

    # test MultiIndex
    pd_data = pd.MultiIndex.from_arrays(
        [np.random.choice(list("ABC"), 100), np.random.randint(0, 10, 100)]
    )
    arrow_data, meta = pandas_to_arrow(pd_data)
    pd_res = arrow_to_pandas(arrow_data, meta)
    pd.testing.assert_index_equal(pd_data, pd_res)


def test_scalar_convert():
    scalar_data = 12.3456
    arrow_data, meta = pandas_to_arrow(scalar_data)
    assert arrow_data.column_names == ["_idx_0"]

    assert arrow_data[0][0].as_py() == scalar_data

    scalar_res = arrow_to_pandas(arrow_data, meta)
    assert scalar_data == scalar_res
