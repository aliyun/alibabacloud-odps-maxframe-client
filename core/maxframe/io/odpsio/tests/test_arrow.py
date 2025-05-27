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
import pyarrow as pa
import pytest

from ....lib.dtypes_extension import dict_
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


@pytest.mark.skipif(
    pa is None or not hasattr(pd, "ArrowDtype"),
    reason="pandas doesn't support ArrowDtype",
)
def test_map_convert():
    pd_data = pd.DataFrame(
        {
            "A": pd.Series(
                [(("k1", "v1"), ("k2", "v2"))], dtype=dict_(pa.string(), pa.string())
            ),
            "B": pd.Series([{"k1": 1, "k2": 2}], dtype=dict_(pa.string(), pa.int64())),
        },
    )
    arrow_data, meta = pandas_to_arrow(pd_data)
    assert arrow_data.column_names == ["_idx_0", "a", "b"]
    pd.testing.assert_series_equal(
        meta.pd_column_dtypes,
        pd.Series(
            [dict_(pa.string(), pa.string()), dict_(pa.string(), pa.int64())],
            index=["A", "B"],
        ),
    )
    pd_result = arrow_to_pandas(arrow_data, meta)
    pd.testing.assert_frame_equal(pd_data, pd_result)


def test_datetime_with_tz_convert():
    pd_data = pd.DataFrame(
        {
            "a": pd.to_datetime(
                pd.Series([1609459200, 1609545600], index=[0, 1]), unit="s"
            ),
        },
    )

    arrow_data, meta = pandas_to_arrow(pd_data)
    assert arrow_data.column_names == ["_idx_0", "a"]
    pd_result = arrow_to_pandas(arrow_data, meta)
    pd.testing.assert_frame_equal(pd_data, pd_result)
