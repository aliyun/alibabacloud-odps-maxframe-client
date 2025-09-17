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

from ....udf import MarkedFunction
from ... import DataFrame
from ...core import DATAFRAME_TYPE, SERIES_TYPE


@pytest.fixture
def df1():
    return DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})


@pytest.fixture
def df2():
    return DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])


@pytest.fixture
def df3():
    return DataFrame(
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        columns=["a", "b", "c"],
        index=pd.MultiIndex.from_arrays([[1, 2, 3], [1, 2, 3]], names=["A", "B"]),
    )


def test_apply_chunk_infer_dtypes_and_index(df1, df2, df3):
    # dataframe -> dataframe filter
    result = df3.mf.apply_chunk(
        lambda data: data.query("A > 1"), batch_rows=2, output_type="dataframe"
    )
    assert isinstance(result, DATAFRAME_TYPE)
    assert df3.index_value.key != result.index_value.key
    assert df3.index_value.to_pandas().names == result.index_value.to_pandas().names

    # dataframe -> dataframe keep same
    result = df1.mf.apply_chunk(
        lambda data: data, batch_rows=2, output_type="dataframe"
    )
    assert isinstance(result, DATAFRAME_TYPE)
    assert result.index_value is df1.index_value

    # dataframe -> dataframe ufunc with arguments
    result = df1.mf.apply_chunk(
        np.add, batch_rows=2, args=(2,), output_type="dataframe"
    )
    assert isinstance(result, DATAFRAME_TYPE)
    assert result.index_value is df1.index_value
    assert result.dtypes.equals(df1.dtypes)
    assert result.shape == df1.shape

    # dataframe -> series ufunc return series
    result = df1.mf.apply_chunk(np.sum, batch_rows=2)
    assert isinstance(result, SERIES_TYPE)
    assert result.index_value is not df1.index_value

    # series -> series
    result = df3.a.mf.apply_chunk(lambda data: data, batch_rows=2, output_type="series")
    assert isinstance(result, SERIES_TYPE)
    assert df3.a.index_value is result.index_value

    result = df3.a.mf.apply_chunk(
        np.sum, batch_rows=2, output_type="series", dtype=np.int64, name="sum"
    )
    assert isinstance(result, SERIES_TYPE)
    assert isinstance(result.index_value.to_pandas(), pd.RangeIndex)

    # general functions
    def process(data, param, k):
        return data * param * k

    result = df2.mf.apply_chunk(
        process, batch_rows=3, output_type="dataframe", args=(4,), k=1
    )
    assert result.index_value is df2.index_value
    assert result.dtypes.equals(df2.dtypes)

    def process(data, param, k) -> pd.DataFrame[df2.dtypes]:
        return data * param * k

    result = df2.mf.apply_chunk(process, batch_rows=3, args=(4,), k=1)
    assert result.index_value is df2.index_value
    assert result.dtypes.equals(df2.dtypes)

    # mark functions
    from ....udf import with_python_requirements, with_resources

    @with_resources("empty.txt")
    @with_python_requirements("numpy")
    def process(data, k) -> pd.DataFrame[df1.dtypes]:
        return data

    result = df1.mf.apply_chunk(process, batch_rows=3, k=1)
    assert result.index_value is df1.index_value
    assert result.dtypes.equals(df1.dtypes)
    assert isinstance(result.op.func, MarkedFunction)
    assert result.op.func is process
    assert result.op.func.resources is process.resources
    assert result.op.func.pythonpacks is process.pythonpacks

    def func_series_ret_series(data):
        return pd.DataFrame([data, data])

    result = df3.a.mf.apply_chunk(
        func_series_ret_series, batch_rows=2, output_type="dataframe"
    )
    assert isinstance(result, DATAFRAME_TYPE)
    assert result.op.func is func_series_ret_series


def test_apply_test(df1):
    def process(x, param):
        return x * param

    result = df1.a.mf.apply_chunk(
        process, batch_rows=2, output_type="series", args=(5,)
    )
    assert isinstance(result, SERIES_TYPE)


def test_apply_chunk(df1):
    keys = [1, 2]

    def f(x, keys):
        if x["a"] in keys:
            return [1, 0]
        else:
            return [0, 1]

    result = df1[["a"]].mf.apply_chunk(
        f,
        output_type="dataframe",
        dtypes=pd.Series(["int64", "int64"]),
        batch_rows=5,
        keys=keys,
    )
    assert result.shape == (np.nan, 2)
    assert df1.index_value.key != result.index_value.key

    # dataframe return series
    result = df1.mf.apply_chunk(
        lambda x: x.a,
        output_type="series",
        dtype="int64",
        batch_rows=5,
    )
    assert result.shape == (np.nan,)
    assert df1.index_value.key == result.index_value.key
    assert df1.a.index_value.key == result.index_value.key

    # return dataframe with given dtypes
    result = df1.a.mf.apply_chunk(
        lambda x: pd.concat([x, x], axis=1),
        output_type="dataframe",
        dtypes=pd.Series(["int64", "int64"]),
        batch_rows=5,
    )
    assert result.shape == (np.nan, 2)
    assert df1.a.index_value.key != result.index_value.key

    # return series but as dataframe
    result = df1.a.mf.apply_chunk(
        lambda x: pd.concat([x, x], axis=0),
        output_type="dataframe",
        dtypes={"c": np.int_},
        batch_rows=5,
    )
    assert result.shape == (np.nan, 1)


def test_apply_chunk_exception(df1):
    with pytest.raises(ValueError):
        df1.mf.apply_chunk(lambda data: data, batch_rows=-1, output_type="dataframe")

    with pytest.raises(TypeError):
        df1.mf.apply_chunk(
            lambda data: data, batch_rows=object(), output_type="dataframe"
        )
