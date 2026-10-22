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
import pyarrow as pa
import pytest

from maxframe.liteframe.core import LiteFrameData
from maxframe.liteframe.operators.datasource import LiteFrameFromLocal
from maxframe.utils.datatypes import is_arrow_dtype_supported


@pytest.fixture
def test_df():
    """A small DataFrame used by the expression-evaluator tests."""
    return pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})


@pytest.fixture
def io_test_df():
    """A DataFrame covering int / float / string columns for IO round-trips."""
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})


@pytest.fixture
def make_tileable_data():
    """Return a factory building a minimal ``LiteFrameData`` for testing."""

    def _make(df):
        dtypes = pd.Series({c: df[c].dtype for c in df.columns})
        data = LiteFrameData(
            shape=(len(df), len(df.columns)),
            physical_dtypes=dtypes,
        )
        data._key = f"test_{id(df)}"
        return data

    return _make


@pytest.fixture
def make_liteframe():
    """Return a factory building a ``LiteFrame`` entity for testing.

    The created entity is backed by a ``LiteFrameFromLocal`` source so it has
    a proper ``_data`` and populated outputs — required by adapter codegen
    tests that read ``op.inputs`` / ``op.outputs``.
    """

    def _make(df):
        if not is_arrow_dtype_supported():
            pytest.skip("pandas ArrowDtype is not supported in current environment")
        dtypes = pd.Series({c: pd.ArrowDtype(pa.int64()) for c in df.columns})
        op = LiteFrameFromLocal(data=df, dtypes=dtypes, nrows=len(df))
        return op._new_liteframe_from_source(shape=(len(df), len(df.columns)))

    return _make
