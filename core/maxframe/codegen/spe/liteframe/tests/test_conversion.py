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

import numpy as np
import pandas as pd

from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.liteframe.datasource import LiteFrameFromTensorAdapter
from maxframe.codegen.spe.tensor.datasource import TensorFromLiteFrameAdapter
from maxframe.tensor.datasource.from_liteframe import (
    TensorFromLiteFrame,
    from_liteframe,
)


def test_from_tensor_adapter_2d():
    """LiteFrameFromTensorAdapter converts a 2-d array into columns."""
    data = np.arange(12).reshape(4, 3)
    col_names = ["a", "b", "c"]
    df = LiteFrameFromTensorAdapter._from_tensor(data, col_names)
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == 4
    np.testing.assert_array_equal(df.values, data)


def test_from_tensor_adapter_1d():
    """LiteFrameFromTensorAdapter converts a 1-d array into one column."""
    data = np.arange(5, dtype=np.int64)
    df = LiteFrameFromTensorAdapter._from_tensor(data, ["val"])
    assert list(df.columns) == ["val"]
    assert df["val"].tolist() == [0, 1, 2, 3, 4]


def test_from_liteframe_adapter_basic():
    """TensorFromLiteFrameAdapter converts a DataFrame to an F-ordered array."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    arr = TensorFromLiteFrameAdapter._from_liteframe(
        df, ["a", "b"], np.dtype(np.int64), None
    )
    assert arr.shape == (3, 2)
    assert arr.flags["F_CONTIGUOUS"]
    np.testing.assert_array_equal(arr, df.values)


def test_from_liteframe_adapter_selects_visible_columns():
    """Visible columns are selected/reordered before to_numpy."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    arr = TensorFromLiteFrameAdapter._from_liteframe(
        df, ["c", "a"], np.dtype(np.int64), None
    )
    assert arr.shape == (2, 2)
    np.testing.assert_array_equal(arr[:, 0], [5, 6])
    np.testing.assert_array_equal(arr[:, 1], [1, 2])


def test_from_liteframe_adapter_materializes_range_columns():
    """Virtual range columns are materialized before conversion."""
    df = pd.DataFrame({"a": [10, 20, 30]})
    range_columns = {"__lf_range_index__": (0, 3, 1)}
    arr = TensorFromLiteFrameAdapter._from_liteframe(
        df,
        ["__lf_range_index__", "a"],
        np.dtype(np.int64),
        range_columns,
    )
    assert arr.shape == (3, 2)
    assert arr[:, 0].tolist() == [0, 1, 2]
    assert arr[:, 1].tolist() == [10, 20, 30]


def test_from_liteframe_adapter_codegen(make_liteframe):
    """generate_code emits a _from_liteframe call with the right args."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    lf = make_liteframe(df)

    tensor = from_liteframe(lf)
    op = tensor.op
    assert isinstance(op, TensorFromLiteFrame)
    adapter = TensorFromLiteFrameAdapter()
    context = SPECodeContext()
    code = adapter.generate_code(op, context)
    assert len(code) == 1
    assert "TensorFromLiteFrameAdapter._from_liteframe" in code[0]
