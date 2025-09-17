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
from .....core import OutputType, TileableGraph, build_fetch
from .....lib.dtypes_extension import dict_
from .....utils import (
    ARROW_DTYPE_NOT_SUPPORTED,
    deserialize_serializable,
    serialize_serializable,
)
from ..core import SeriesDictMethod
from ..getitem import SeriesDictGetItemOperator

pytestmark = pytest.mark.skipif(
    ARROW_DTYPE_NOT_SUPPORTED, reason="Arrow Dtype is not supported"
)


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


def test_invalid_dtype(df):
    with pytest.raises(AttributeError):
        df["C"].dict.contains("k1")


def test_getitem(df):
    s1 = df["A"].dict["k1"]
    assert isinstance(s1, md.Series)
    assert s1.name == "k1"
    assert s1.dtype == pd.ArrowDtype(pa.string())
    assert s1.shape == (1,)
    assert s1.index_value == df.index_value
    op = s1.op
    assert isinstance(op, SeriesDictMethod)
    assert op.method == "getitem"
    assert op.method_kwargs["query_key"] == "k1"
    assert op.method_kwargs["default_value"] is None
    assert op.method_kwargs["ignore_key_error"] is False


def test_getitem_with_default_value(df):
    s1 = df["B"].dict.get("k1", 1)
    assert isinstance(s1, md.Series)
    assert s1.name == "k1"
    assert s1.dtype == pd.ArrowDtype(pa.int64())
    assert s1.shape == (1,)
    assert s1.index_value == df.index_value
    op = s1.op
    assert isinstance(op, SeriesDictMethod)
    assert op.method == "getitem"
    assert op.method_kwargs["query_key"] == "k1"
    assert op.method_kwargs["default_value"] == 1
    assert op.method_kwargs["ignore_key_error"] is True


def test_setitem(df):
    s1 = df["A"]
    s1.dict["k1"] = "v3"
    assert isinstance(s1, md.Series)
    assert s1.name == "A"
    assert s1.dtype == dict_(pa.string(), pa.string())
    assert s1.index_value == df.index_value
    assert s1.shape == (1,)
    op = s1.op
    assert isinstance(op, SeriesDictMethod)
    assert op.method == "setitem"
    assert op.method_kwargs["query_key"] == "k1"
    assert op.method_kwargs["value"] == "v3"


def test_length(df):
    s1 = df["A"].dict.len()
    assert isinstance(s1, md.Series)
    assert s1.name is None
    assert s1.dtype == pd.ArrowDtype(pa.int64())
    assert s1.shape == (1,)
    assert s1.index_value == df.index_value
    op = s1.op
    assert op.method == "len"
    assert isinstance(op, SeriesDictMethod)


def test_remove(df):
    s1 = df["A"].dict.remove("k1", ignore_key_error=True)
    assert isinstance(s1, md.Series)
    assert s1.name == "A"
    assert s1.dtype == dict_(pa.string(), pa.string())
    assert s1.index_value == df.index_value
    assert s1.shape == (1,)
    op = s1.op
    assert isinstance(op, SeriesDictMethod)
    assert op.method == "remove"
    assert op.method_kwargs["query_key"] == "k1"
    assert op.method_kwargs["ignore_key_error"] is True


def test_contains(df):
    s1 = df["A"].dict.contains("k1")
    assert isinstance(s1, md.Series)
    assert s1.name is None
    assert s1.dtype == pd.ArrowDtype(pa.bool_())
    assert s1.index_value == df.index_value
    assert s1.shape == (1,)
    op = s1.op
    assert isinstance(op, SeriesDictMethod)
    assert op.method == "contains"
    assert op.method_kwargs["query_key"] == "k1"


def test_legacy_compatibility(df):
    in_series = df["A"]
    legacy_op = SeriesDictGetItemOperator(
        query_key="k1",
        default_value=1,
        ignore_key_error=True,
        _output_types=[OutputType.series],
    )
    out = legacy_op.new_tileable(
        [in_series],
        shape=in_series.shape,
        index_value=in_series.index_value,
        name="k1",
        dtype=pd.ArrowDtype(pa.int64()),
    )
    fetch_node = build_fetch(in_series).data
    out.op.inputs = [fetch_node]

    dag = TileableGraph([out.data])
    dag.add_node(fetch_node)
    dag.add_node(out.data)
    dag.add_edge(fetch_node, out.data)
    dag_out = deserialize_serializable(serialize_serializable(dag))

    out_op = dag_out.results[0].op
    assert isinstance(out_op, SeriesDictMethod)
    assert out_op.method == "getitem"
    assert out_op.method_kwargs["ignore_key_error"] == True
    assert out_op.method_kwargs["default_value"] == 1
    assert out_op.method_kwargs["query_key"] == "k1"
