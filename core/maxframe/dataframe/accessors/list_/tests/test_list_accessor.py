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

from ..... import dataframe as md
from .....lib.dtypes_extension import list_
from .....utils import ARROW_DTYPE_NOT_SUPPORTED
from ..core import SeriesListMethod

pytestmark = pytest.mark.skipif(
    ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported"
)


@pytest.fixture
def df():
    return md.DataFrame(
        {
            "A": pd.Series([[5, 3, 2]], dtype=list_(pa.int32())),
            "B": pd.Series([["ab", "cd"]], dtype=list_(pa.string())),
            "C": pd.Series([1], dtype=np.dtype("int64")),
        },
        index=[1],
    )


def test_invalid_dtype(df):
    with pytest.raises(AttributeError):
        df["C"].list.len()


def test_getitem(df):
    s1 = df["A"].list[1]
    assert isinstance(s1, md.Series)
    assert s1.dtype == pd.ArrowDtype(pa.int32())
    assert s1.shape == (1,)
    assert s1.index_value == df.index_value
    op = s1.op
    assert isinstance(op, SeriesListMethod)
    assert op.method == "getitem"
    assert op.method_kwargs["query_index"] == 1
    assert op.method_kwargs["ignore_index_error"] is False


def test_getitem_ignore_index_err(df):
    s1 = df["B"].list.get(1)
    assert isinstance(s1, md.Series)
    assert s1.dtype == pd.ArrowDtype(pa.string())
    assert s1.shape == (1,)
    assert s1.index_value == df.index_value
    op = s1.op
    assert isinstance(op, SeriesListMethod)
    assert op.method == "getitem"
    assert op.method_kwargs["query_index"] == 1
    assert op.method_kwargs["ignore_index_error"] is True


def test_length(df):
    s1 = df["A"].list.len()
    assert isinstance(s1, md.Series)
    assert s1.dtype == pd.ArrowDtype(pa.int64())
    assert s1.shape == (1,)
    assert s1.index_value == df.index_value
    op = s1.op
    assert isinstance(op, SeriesListMethod)
    assert op.method == "len"
