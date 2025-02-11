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
from .....lib.dtypes_extension import dict_
from .....utils import ARROW_DTYPE_NOT_SUPPORTED
from ..contains import SeriesDictContainsOperator
from ..getitem import SeriesDictGetItemOperator
from ..length import SeriesDictLengthOperator
from ..remove import SeriesDictRemoveOperator
from ..setitem import SeriesDictSetItemOperator


@pytest.fixture
def df():
    return md.DataFrame(
        {
            "A": pd.Series(
                [(("k1", "v1"), ("k2", "v2"))], dtype=dict_(pa.string(), pa.string())
            ),
            "B": pd.Series([{"k1": 1, "k2": 2}], dtype=dict_(pa.string(), pa.int64())),
            "C": pd.Series([1], dtype=np.dtype("int64")),
        },
        index=[1],
    )


@pytest.mark.skipif(ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported")
def test_invalid_dtype(df):
    with pytest.raises(AttributeError):
        df["C"].dict.contains("k1")


@pytest.mark.skipif(ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported")
def test_getitem(df):
    s1 = df["A"].dict["k1"]
    assert isinstance(s1, md.Series)
    assert s1.name == "k1"
    assert s1.dtype == pd.ArrowDtype(pa.string())
    assert s1.shape == (1,)
    assert s1.index_value == df.index_value
    op = s1.op
    assert isinstance(op, SeriesDictGetItemOperator)
    assert op.query_key == "k1"
    assert op.default_value is None
    assert op.ignore_key_error is False


@pytest.mark.skipif(ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported")
def test_getitem_with_default_value(df):
    s1 = df["B"].dict.get("k1", 1)
    assert isinstance(s1, md.Series)
    assert s1.name == "k1"
    assert s1.dtype == pd.ArrowDtype(pa.int64())
    assert s1.shape == (1,)
    assert s1.index_value == df.index_value
    op = s1.op
    assert isinstance(op, SeriesDictGetItemOperator)
    assert op.query_key == "k1"
    assert op.default_value == 1
    assert op.ignore_key_error is True


@pytest.mark.skipif(ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported")
def test_setitem(df):
    s1 = df["A"]
    s1.dict["k1"] = "v3"
    assert isinstance(s1, md.Series)
    assert s1.name == "A"
    assert s1.dtype == dict_(pa.string(), pa.string())
    assert s1.index_value == df.index_value
    assert s1.shape == (1,)
    op = s1.op
    assert isinstance(op, SeriesDictSetItemOperator)
    assert op.query_key == "k1"
    assert op.value == "v3"


@pytest.mark.skipif(ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported")
def test_length(df):
    s1 = df["A"].dict.len()
    assert isinstance(s1, md.Series)
    assert s1.name is None
    assert s1.dtype == pd.ArrowDtype(pa.int64())
    assert s1.shape == (1,)
    assert s1.index_value == df.index_value
    op = s1.op
    assert isinstance(op, SeriesDictLengthOperator)


@pytest.mark.skipif(ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported")
def test_remove(df):
    s1 = df["A"].dict.remove("k1", ignore_key_error=True)
    assert isinstance(s1, md.Series)
    assert s1.name == "A"
    assert s1.dtype == dict_(pa.string(), pa.string())
    assert s1.index_value == df.index_value
    assert s1.shape == (1,)
    op = s1.op
    assert isinstance(op, SeriesDictRemoveOperator)
    assert op.query_key == "k1"
    assert op.ignore_key_error is True


@pytest.mark.skipif(ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported")
def test_contains(df):
    s1 = df["A"].dict.contains("k1")
    assert isinstance(s1, md.Series)
    assert s1.name is None
    assert s1.dtype == pd.ArrowDtype(pa.bool_())
    assert s1.index_value == df.index_value
    assert s1.shape == (1,)
    op = s1.op
    assert isinstance(op, SeriesDictContainsOperator)
    assert op.query_key == "k1"
