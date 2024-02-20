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
import pytest

from .... import opcodes
from ....core import OutputType
from ....tensor.core import TENSOR_TYPE
from ... import eval as maxframe_eval
from ... import get_dummies, to_numeric
from ...core import CATEGORICAL_TYPE, DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE
from ...datasource.dataframe import from_pandas as from_pandas_df
from ...datasource.index import from_pandas as from_pandas_index
from ...datasource.series import from_pandas as from_pandas_series
from .. import astype, cut


def test_transform():
    cols = [chr(ord("A") + i) for i in range(10)]
    df_raw = pd.DataFrame(dict((c, [i**2 for i in range(20)]) for c in cols))
    df = from_pandas_df(df_raw, chunk_size=5)

    idxes = [chr(ord("A") + i) for i in range(20)]
    s_raw = pd.Series([i**2 for i in range(20)], index=idxes)
    series = from_pandas_series(s_raw, chunk_size=5)

    def rename_fn(f, new_name):
        f.__name__ = new_name
        return f

    # DATAFRAME CASES

    # test transform with infer failure
    def transform_df_with_err(v):
        assert len(v) > 2
        return v.sort_values()

    with pytest.raises(TypeError):
        df.transform(transform_df_with_err)

    r = df.transform(transform_df_with_err, dtypes=df_raw.dtypes)
    assert r.shape == df.shape
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe

    # test transform scenarios on data frames
    r = df.transform(lambda x: list(range(len(x))))
    assert all(v == np.dtype("int64") for v in r.dtypes) is True
    assert r.shape == df.shape
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe

    r = df.transform(lambda x: list(range(len(x))), axis=1)
    assert all(v == np.dtype("int64") for v in r.dtypes) is True
    assert r.shape == df.shape
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe

    r = df.transform(["cumsum", "cummax", lambda x: x + 1])
    assert all(v == np.dtype("int64") for v in r.dtypes) is True
    assert r.shape == (df.shape[0], df.shape[1] * 3)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe

    r = df.transform({"A": "cumsum", "D": ["cumsum", "cummax"], "F": lambda x: x + 1})
    assert all(v == np.dtype("int64") for v in r.dtypes) is True
    assert r.shape == (df.shape[0], 4)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe

    # test agg scenarios on series
    r = df.transform(lambda x: x.iloc[:-1], _call_agg=True)
    assert all(v == np.dtype("int64") for v in r.dtypes) is True
    assert r.shape == (np.nan, df.shape[1])
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe

    r = df.transform(lambda x: x.iloc[:-1], axis=1, _call_agg=True)
    assert all(v == np.dtype("int64") for v in r.dtypes) is True
    assert r.shape == (df.shape[0], np.nan)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe

    fn_list = [
        rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), "f1"),
        lambda x: x.iloc[:-1].reset_index(drop=True),
    ]
    r = df.transform(fn_list, _call_agg=True)
    assert all(v == np.dtype("int64") for v in r.dtypes) is True
    assert r.shape == (np.nan, df.shape[1] * 2)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe

    r = df.transform(lambda x: x.sum(), _call_agg=True)
    assert r.dtype == np.dtype("int64")
    assert r.shape == (df.shape[1],)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.series

    fn_dict = {
        "A": rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), "f1"),
        "D": [
            rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), "f1"),
            lambda x: x.iloc[:-1].reset_index(drop=True),
        ],
        "F": lambda x: x.iloc[:-1].reset_index(drop=True),
    }
    r = df.transform(fn_dict, _call_agg=True)
    assert all(v == np.dtype("int64") for v in r.dtypes) is True
    assert r.shape == (np.nan, 4)
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.dataframe

    # SERIES CASES
    # test transform scenarios on series
    r = series.transform(lambda x: x + 1)
    assert np.dtype("int64") == r.dtype
    assert r.shape == series.shape
    assert r.op._op_type_ == opcodes.TRANSFORM
    assert r.op.output_types[0] == OutputType.series


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


def test_datetime_method():
    s = pd.Series(
        [pd.Timestamp("2020-1-1"), pd.Timestamp("2020-2-1"), pd.Timestamp("2020-3-1")],
        name="ss",
    )
    series = from_pandas_series(s, chunk_size=2)

    r = series.dt.year
    assert r.dtype == s.dt.year.dtype
    pd.testing.assert_index_equal(r.index_value.to_pandas(), s.index)
    assert r.shape == s.shape
    assert r.op.output_types[0] == OutputType.series
    assert r.name == s.dt.year.name

    with pytest.raises(AttributeError):
        _ = from_pandas_series(pd.Series([1])).dt
    with pytest.raises(AttributeError):
        _ = series.dt.non_exist

    assert "ceil" in dir(series.dt)


def test_series_isin():
    # one chunk in multiple chunks
    a = from_pandas_series(pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), chunk_size=10)
    b = from_pandas_series(pd.Series([2, 1, 9, 3]), chunk_size=2)

    r = a.isin(b)
    assert r.dtype == np.dtype("bool")
    assert r.shape == (10,)
    assert len(r.op.inputs) == 2
    assert r.op.output_types[0] == OutputType.series

    with pytest.raises(TypeError):
        _ = a.isin("sth")

    with pytest.raises(TypeError):
        _ = a.to_frame().isin("sth")


def test_astype():
    s = from_pandas_series(pd.Series([1, 2, 1, 2], name="a"), chunk_size=2)
    with pytest.raises(KeyError):
        astype(s, {"b": "str"})

    df = from_pandas_df(
        pd.DataFrame({"a": [1, 2, 1, 2], "b": ["a", "b", "a", "b"]}), chunk_size=2
    )

    with pytest.raises(KeyError):
        astype(df, {"c": "str", "a": "str"})


def test_eval_query():
    rs = np.random.RandomState(0)
    raw = pd.DataFrame({"a": rs.rand(100), "b": rs.rand(100), "c c": rs.rand(100)})
    df = from_pandas_df(raw, chunk_size=(10, 2))

    with pytest.raises(NotImplementedError):
        maxframe_eval("df.a * 2", engine="numexpr")
    with pytest.raises(NotImplementedError):
        maxframe_eval("df.a * 2", parser="pandas")
    with pytest.raises(TypeError):
        df.eval(df)
    with pytest.raises(SyntaxError):
        df.query(
            """
        a + b
        a + `c c`
        """
        )
    with pytest.raises(SyntaxError):
        df.eval(
            """
        def a():
            return v
        a()
        """
        )
    with pytest.raises(SyntaxError):
        df.eval("a + `c")
    with pytest.raises(KeyError):
        df.eval("a + c")
    with pytest.raises(ValueError):
        df.eval("p, q = a + c")
    with pytest.raises(ValueError):
        df.query("p = a + c")


def test_cut():
    s = from_pandas_series(pd.Series([1.0, 2.0, 3.0, 4.0]), chunk_size=2)

    with pytest.raises(ValueError):
        _ = cut(s, -1)

    with pytest.raises(ValueError):
        _ = cut([[1, 2], [3, 4]], 3)

    with pytest.raises(ValueError):
        _ = cut([], 3)

    r, b = cut(s, [1.5, 2.5], retbins=True)
    assert isinstance(r, SERIES_TYPE)
    assert isinstance(b, TENSOR_TYPE)

    r = cut(s.to_tensor(), [1.5, 2.5])
    assert isinstance(r, CATEGORICAL_TYPE)
    assert len(r) == len(s)
    assert "Categorical" in repr(r)

    r = cut([0, 1, 1, 2], bins=4, labels=False)
    assert isinstance(r, TENSOR_TYPE)
    e = pd.cut([0, 1, 1, 2], bins=4, labels=False)
    assert r.dtype == e.dtype


def test_drop():
    # test dataframe drop
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(20, 8)), columns=["c" + str(i + 1) for i in range(8)]
    )

    df = from_pandas_df(raw, chunk_size=8)

    with pytest.raises(KeyError):
        df.drop(columns=["c9"])
    with pytest.raises(NotImplementedError):
        df.drop(columns=from_pandas_series(pd.Series(["c9"])))

    r = df.drop(columns=["c1"])
    pd.testing.assert_index_equal(r.index_value.to_pandas(), raw.index)

    df = from_pandas_df(raw, chunk_size=3)

    columns = ["c2", "c4", "c5", "c6"]
    index = [3, 6, 7]
    r = df.drop(columns=columns, index=index)
    assert isinstance(r, DATAFRAME_TYPE)

    # test series drop
    raw = pd.Series(rs.randint(1000, size=(20,)))
    series = from_pandas_series(raw, chunk_size=3)

    r = series.drop(index=index)
    assert isinstance(r, SERIES_TYPE)

    # test index drop
    ser = pd.Series(range(20))
    rs.shuffle(ser)
    raw = pd.Index(ser)

    idx = from_pandas_index(raw)

    r = idx.drop(index)
    assert isinstance(r, INDEX_TYPE)


def test_drop_duplicates():
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(20, 7)), columns=["c" + str(i + 1) for i in range(7)]
    )
    raw["c7"] = [f"s{j}" for j in range(20)]

    df = from_pandas_df(raw, chunk_size=10)
    with pytest.raises(ValueError):
        df.drop_duplicates(keep=True)
    with pytest.raises(ValueError):
        df.drop_duplicates(method="unknown")
    with pytest.raises(KeyError):
        df.drop_duplicates(subset="c8")

    s = df["c7"]
    with pytest.raises(ValueError):
        s.drop_duplicates(method="unknown")
    with pytest.raises(ValueError):
        s.drop_duplicates(keep=True)


def test_get_dummies():
    raw = pd.DataFrame(
        {
            "a": [1.1, 2.1, 3.1],
            "b": ["5", "-6", "-7"],
            "c": [1, 2, 3],
            "d": ["2", "3", "4"],
        }
    )
    df = from_pandas_df(raw, chunk_size=2)

    with pytest.raises(TypeError):
        _ = get_dummies(df, columns="a")

    with pytest.raises(ValueError):
        _ = get_dummies(df, prefix=["col1"])

    with pytest.raises(ValueError):
        _ = get_dummies(df, columns=["a"], prefix={"a": "col1", "c": "col2"})

    with pytest.raises(KeyError):
        _ = get_dummies(df, columns=["a", "b"], prefix={"a": "col1", "c": "col2"})

    r = get_dummies(df)
    assert isinstance(r, DATAFRAME_TYPE)


def test_to_numeric():
    raw = pd.DataFrame({"a": [1.0, 2, 3, -3]})
    df = from_pandas_df(raw, chunk_size=2)

    with pytest.raises(ValueError):
        _ = to_numeric(df)

    with pytest.raises(ValueError):
        _ = to_numeric([["1.0", 1]])

    with pytest.raises(ValueError):
        _ = to_numeric([])
