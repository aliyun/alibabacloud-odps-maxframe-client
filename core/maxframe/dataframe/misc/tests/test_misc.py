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

from maxframe import options

from .... import opcodes
from ....core import OutputType
from ....dataframe import DataFrame
from ....tensor.core import TENSOR_TYPE
from ....udf import ODPSFunction, with_running_options
from ... import eval as maxframe_eval
from ... import get_dummies, to_numeric
from ...arithmetic import DataFrameGreater, DataFrameLess
from ...core import CATEGORICAL_TYPE, DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE
from ...datasource.dataframe import from_pandas as from_pandas_df
from ...datasource.index import from_pandas as from_pandas_index
from ...datasource.series import from_pandas as from_pandas_series
from .. import astype, cut


def test_dataframe_apply():
    cols = [chr(ord("A") + i) for i in range(10)]
    df_raw = pd.DataFrame(dict((c, [i**2 for i in range(20)]) for c in cols))

    df = from_pandas_df(df_raw, chunk_size=5)

    def df_func_with_err(v):
        assert len(v) > 2
        return v.sort_values()

    def df_series_func_with_err(v):
        assert len(v) > 2
        return 0

    with pytest.raises(TypeError):
        df.apply(df_func_with_err)

    r = df.apply(df_func_with_err, output_type="dataframe", dtypes=df_raw.dtypes)
    assert r.shape == (np.nan, df.shape[-1])
    assert r.op._op_type_ == opcodes.APPLY
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.op.elementwise is False

    r = df.apply(
        df_series_func_with_err, output_type="series", dtype=object, name="output"
    )
    assert r.dtype == np.dtype("O")
    assert r.shape == (df.shape[-1],)
    assert r.op._op_type_ == opcodes.APPLY
    assert r.op.output_types[0] == OutputType.series
    assert r.op.elementwise is False

    r = df.apply("ffill")
    assert r.op._op_type_ == opcodes.FILL_NA

    r = df.apply(np.sqrt)
    assert all(v == np.dtype("float64") for v in r.dtypes) is True
    assert r.shape == df.shape
    assert r.op._op_type_ == opcodes.APPLY
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.op.elementwise is True

    r = df.apply(lambda x: pd.Series([1, 2]))
    assert all(v == np.dtype("int64") for v in r.dtypes) is True
    assert r.shape == (np.nan, df.shape[1])
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.op.elementwise is False

    r = df.apply(np.sum, axis="index")
    assert np.dtype("int64") == r.dtype
    assert r.shape == (df.shape[1],)
    assert r.op.output_types[0] == OutputType.series
    assert r.op.elementwise is False

    r = df.apply(np.sum, axis="columns")
    assert np.dtype("int64") == r.dtype
    assert r.shape == (df.shape[0],)
    assert r.op.output_types[0] == OutputType.series
    assert r.op.elementwise is False

    r = df.apply(lambda x: pd.Series([1, 2], index=["foo", "bar"]), axis=1)
    assert all(v == np.dtype("int64") for v in r.dtypes) is True
    assert r.shape == (df.shape[0], 2)
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.op.elementwise is False

    r = df.apply(lambda x: [1, 2], axis=1, result_type="expand")
    assert all(v == np.dtype("int64") for v in r.dtypes) is True
    assert r.shape == (df.shape[0], 2)
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.op.elementwise is False

    r = df.apply(lambda x: list(range(10)), axis=1, result_type="reduce")
    assert np.dtype("object") == r.dtype
    assert r.shape == (df.shape[0],)
    assert r.op.output_types[0] == OutputType.series
    assert r.op.elementwise is False

    r = df.apply(lambda x: list(range(10)), axis=1, result_type="broadcast")
    assert all(v == np.dtype("int64") for v in r.dtypes) is True
    assert r.shape == (df.shape[0], 10)
    assert r.op.output_types[0] == OutputType.dataframe
    assert r.op.elementwise is False

    raw = pd.DataFrame({"a": [np.array([1, 2, 3]), np.array([4, 5, 6])]})
    df = from_pandas_df(raw)
    df2 = df.apply(
        lambda x: x["a"].astype(pd.Series),
        axis=1,
        output_type="dataframe",
        dtypes=pd.Series([np.dtype(float)] * 3),
    )
    assert df2.ndim == 2
    assert df2.op.expect_resources == options.function.default_running_options


def test_series_apply():
    idxes = [chr(ord("A") + i) for i in range(20)]
    s_raw = pd.Series([i**2 for i in range(20)], index=idxes)

    series = from_pandas_series(s_raw, chunk_size=5)

    r = series.apply("add", args=(1,))
    assert r.op._op_type_ == opcodes.ADD

    r = series.apply(np.sqrt)
    assert np.dtype("float64") == r.dtype
    assert r.shape == series.shape
    assert r.index_value is series.index_value
    assert r.op._op_type_ == opcodes.APPLY
    assert r.op.output_types[0] == OutputType.series

    r = series.apply("sqrt")
    assert np.dtype("float64") == r.dtype
    assert r.shape == series.shape
    assert r.op._op_type_ == opcodes.APPLY
    assert r.op.output_types[0] == OutputType.series

    r = series.apply(lambda x: [x, x + 1], convert_dtype=False)
    assert np.dtype("object") == r.dtype
    assert r.shape == series.shape
    assert r.op._op_type_ == opcodes.APPLY
    assert r.op.output_types[0] == OutputType.series

    s_raw2 = pd.Series([np.array([1, 2, 3]), np.array([4, 5, 6])])
    series = from_pandas_series(s_raw2)

    r = series.apply(np.sum)
    assert r.dtype == np.dtype(object)

    r = series.apply(lambda x: pd.Series([1]), output_type="dataframe")
    expected = s_raw2.apply(lambda x: pd.Series([1]))
    pd.testing.assert_series_equal(r.dtypes, expected.dtypes)

    dtypes = pd.Series([np.dtype(float)] * 3)
    r = series.apply(pd.Series, output_type="dataframe", dtypes=dtypes)
    assert r.ndim == 2
    pd.testing.assert_series_equal(r.dtypes, dtypes)
    assert r.shape == (2, 3)

    def apply_with_error(_):
        raise ValueError

    r = series.apply(apply_with_error, output_type="dataframe", dtypes=dtypes)
    assert r.ndim == 2

    r = series.apply(
        pd.Series, output_type="dataframe", dtypes=dtypes, index=pd.RangeIndex(2)
    )
    assert r.ndim == 2
    assert r.op.expect_resources == options.function.default_running_options

    pd.testing.assert_series_equal(r.dtypes, dtypes)
    assert r.shape == (2, 3)

    with pytest.raises(AttributeError, match="abc"):
        series.apply("abc")

    with pytest.raises(TypeError):
        # dtypes not provided
        series.apply(lambda x: x.tolist(), output_type="dataframe")


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

    def transform_df_with_param(row, param, k):
        assert param == 5
        assert k == "6"
        return row

    r = df.transform(transform_df_with_param, 1, 5, k="6")
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
    assert r.op.expect_resources == options.function.default_running_options


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
        rs.randint(1000, size=(20, 7)),
        columns=["c" + str(i + 1) for i in range(7)],
        index=pd.Index(range(20), name="idx"),
    )
    raw["c7"] = [f"s{j}" for j in range(20)]

    df = from_pandas_df(raw, chunk_size=10)
    with pytest.raises(ValueError):
        df.drop_duplicates(keep=True)
    with pytest.raises(ValueError):
        df.drop_duplicates(method="unknown")
    with pytest.raises(KeyError):
        df.drop_duplicates(subset="c8")

    # check index
    distinct_df = df.drop_duplicates()
    assert distinct_df.index_value.name == df.index_value.name
    assert isinstance(df.index_value.to_pandas(), pd.RangeIndex)
    assert not isinstance(distinct_df.index_value.to_pandas(), pd.RangeIndex)

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

    raw = pd.Series(["a", "a", "b", "c"])
    ms = from_pandas_series(raw, chunk_size=2)
    r = get_dummies(ms)
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


def test_case_when():
    rs = np.random.RandomState(0)
    raw = pd.DataFrame(
        rs.randint(1000, size=(20, 8)), columns=["c" + str(i + 1) for i in range(8)]
    )
    df = from_pandas_df(raw, chunk_size=8)

    with pytest.raises(TypeError):
        df.c1.case_when(df.c2)
    with pytest.raises(ValueError):
        df.c1.case_when([])
    with pytest.raises(TypeError):
        df.c1.case_when([[]])
    with pytest.raises(ValueError):
        df.c1.case_when([()])

    col = df.c1.case_when([(df.c2 < 10, 10), (df.c2 > 20, df.c3)])
    assert len(col.inputs) == 4
    assert isinstance(col.inputs[1].op, DataFrameLess)
    assert isinstance(col.inputs[2].op, DataFrameGreater)


def test_apply():
    df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})

    keys = [1, 2]

    @with_running_options(engine="spe", memory="40GB")
    def f(x, keys):
        if x["a"] in keys:
            return [1, 0]
        else:
            return [0, 1]

    apply_df = df[["a"]].apply(
        f,
        output_type="dataframe",
        dtypes=pd.Series(["int64", "int64"]),
        axis=1,
        result_type="expand",
        keys=keys,
    )
    assert apply_df.shape == (3, 2)
    assert apply_df.op.expect_engine == "SPE"
    assert apply_df.op.expect_resources == {
        "cpu": 4,
        "memory": "40GB",
        "gpu": 0,
        "gu_quota": None,
    }


def test_pivot_table():
    from ...groupby.aggregation import DataFrameGroupByAgg
    from ...reshape.pivot_table import DataFramePivotTable

    raw = pd.DataFrame(
        {
            "A": "foo foo foo foo foo bar bar bar bar".split(),
            "B": "one one one two two one one two two".split(),
            "C": "small large large small small large small small large".split(),
            "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
            "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
        }
    )
    df = from_pandas_df(raw, chunk_size=8)
    with pytest.raises(ValueError):
        df.pivot_table(index=123)
    with pytest.raises(ValueError):
        df.pivot_table(index=["F"])
    with pytest.raises(ValueError):
        df.pivot_table(values=["D", "E"], aggfunc="sum")

    t = df.pivot_table(index=["A", "B", "C"])
    assert isinstance(t.op, DataFrameGroupByAgg)
    t = df.pivot_table(index="A", values=["D", "E"], aggfunc="sum")
    assert isinstance(t.op, DataFrameGroupByAgg)

    t = df.pivot_table(index=["A", "B"], values=["D", "E"], aggfunc="sum", margins=True)
    assert isinstance(t.op, DataFramePivotTable)

    t = df.pivot_table(index="A", columns=["B", "C"], aggfunc="sum")
    assert isinstance(t.op, DataFramePivotTable)
    assert t.shape == (np.nan, np.nan)

    t = df.pivot_table(index=["A", "B"], columns="C", aggfunc="sum")
    assert isinstance(t.op, DataFramePivotTable)
    assert t.shape == (np.nan, np.nan)


def test_map_with_functions():
    raw = pd.Series([1, 2, 3], name="s_name")
    series = from_pandas_series(raw, chunk_size=2)

    # inferred type may not be exact
    def fn1(val):
        return val

    with pytest.raises(ValueError, match="int type"):
        series.map(fn1)
    mapped = series.map(fn1, dtype="float64", skip_infer=True)
    assert mapped.dtype == np.dtype("float64")

    # test when type infer is valid
    def fn2(val):
        return val * 1.0

    mapped = series.map(fn2)
    assert mapped.dtype == np.dtype("float64")

    # test function with type annotations
    def fn3(val) -> int:
        return val

    mapped = series.map(fn3)
    assert mapped.dtype == np.dtype("int64")

    # test odps function
    odps_func = ODPSFunction("test_odps_udf", dtype=np.float64)
    mapped = series.map(odps_func)
    assert isinstance(mapped.op.arg, ODPSFunction)
    assert mapped.dtype == np.dtype("float64")
