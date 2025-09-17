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

import os
import uuid
from collections import OrderedDict
from math import isinf

import mock
import numpy as np
import pandas as pd
import pytest
from odps import ODPS
from odps import types as odps_types
from odps.errors import ODPSError

from .... import tensor as mt
from ....core import OutputType
from ....core.operator import estimate_size
from ....tests.utils import tn
from ....utils import lazy_import
from ... import read_odps_query, read_odps_table
from ...core import (
    DatetimeIndex,
    Float64Index,
    Index,
    IndexValue,
    Int64Index,
    MultiIndex,
)
from ..dataframe import from_pandas as from_pandas_df
from ..date_range import date_range
from ..from_tensor import (
    dataframe_from_1d_tileables,
    dataframe_from_tensor,
    series_from_tensor,
)
from ..index import from_pandas as from_pandas_index
from ..index import from_tileable
from ..read_odps_query import (
    ColumnSchema,
    _parse_full_explain,
    _parse_simple_explain,
    _resolve_query_schema,
    _resolve_task_sector,
)
from ..series import from_pandas as from_pandas_series

ray = lazy_import("ray")


def test_from_pandas_dataframe():
    data = pd.DataFrame(
        np.random.rand(10, 10), columns=["c" + str(i) for i in range(10)]
    )
    df = from_pandas_df(data, chunk_size=4)

    pd.testing.assert_series_equal(df.op.dtypes, data.dtypes)
    assert isinstance(df.index_value._index_value, IndexValue.RangeIndex)
    assert df.index_value._index_value._slice == slice(0, 10, 1)
    assert df.index_value.is_monotonic_increasing is True
    assert df.index_value.is_monotonic_decreasing is False
    assert df.index_value.is_unique is True
    assert df.index_value.min_val == 0
    assert df.index_value.max_val == 9
    np.testing.assert_equal(df.columns_value._index_value._data, data.columns.values)

    result_ctx = dict()
    estimate_size(result_ctx, df.op)
    assert result_ctx[df.key] > 0 and not isinf(result_ctx[df.key])

    data2 = data[::2]
    df2 = from_pandas_df(data2, chunk_size=4)

    pd.testing.assert_series_equal(df.op.dtypes, data2.dtypes)
    assert isinstance(df2.index_value._index_value, IndexValue.RangeIndex)
    assert df2.index_value._index_value._slice == slice(0, 10, 2)


def test_from_pandas_dataframe_with_multi_index():
    index = pd.MultiIndex.from_tuples([("k1", "v1")], names=["X", "Y"])
    data = np.random.randint(0, 100, size=(1, 3))
    pdf = pd.DataFrame(data, columns=["A", "B", "C"], index=index)
    df = from_pandas_df(pdf, chunk_size=4)
    assert isinstance(df.index, MultiIndex)
    assert df.index.names == ["X", "Y"]
    assert df.index.name is None


def test_from_pandas_series():
    data = pd.Series(np.random.rand(10), name="a")
    series = from_pandas_series(data, chunk_size=4)

    assert series.name == data.name
    assert isinstance(series.index_value._index_value, IndexValue.RangeIndex)
    assert series.index_value._index_value._slice == slice(0, 10, 1)
    assert series.index_value.is_monotonic_increasing is True
    assert series.index_value.is_monotonic_decreasing is False
    assert series.index_value.is_unique is True
    assert series.index_value.min_val == 0
    assert series.index_value.max_val == 9


def test_from_pandas_index():
    data = pd.date_range("2020-1-1", periods=10, name="date")
    index = from_pandas_index(data, chunk_size=4)

    assert isinstance(index, DatetimeIndex)
    assert index.name == data.name
    assert index.dtype == data.dtype
    assert isinstance(index.index_value.value, IndexValue.DatetimeIndex)


def test_from_tileable_index():
    t = mt.random.rand(10, 4)

    with pytest.raises(ValueError):
        from_tileable(t)

    pd_df = pd.DataFrame(
        np.random.rand(10, 4), index=np.arange(10, 0, -1).astype(np.int64)
    )
    pd_df.index.name = "ind"
    df = from_pandas_df(pd_df, chunk_size=6)

    for o in [df, df[0]]:
        index = o.index
        assert isinstance(index, (Index, Int64Index))
        assert index.dtype == np.int64
        assert index.name == pd_df.index.name
        assert isinstance(
            index.index_value.value, (IndexValue.Int64Index, IndexValue.Index)
        )

    t = mt.random.rand(10, chunk_size=6)
    index = from_tileable(t, name="new_name")

    assert isinstance(index, (Index, Float64Index))
    assert index.dtype == np.float64
    assert index.name == "new_name"
    assert isinstance(
        index.index_value.value, (IndexValue.Float64Index, IndexValue.Index)
    )


def test_from_tensor():
    tensor = mt.random.rand(10, 10, chunk_size=5)
    df = dataframe_from_tensor(tensor)
    assert isinstance(df.index_value._index_value, IndexValue.RangeIndex)
    assert df.dtypes[0] == tensor.dtype

    # test converted from scalar
    scalar = mt.array(1)
    np.testing.assert_equal(scalar.ndim, 0)
    with pytest.raises(TypeError):
        dataframe_from_tensor(scalar)

    # from tensor with given columns
    df = dataframe_from_tensor(tensor, columns=list("abcdefghij"))
    pd.testing.assert_index_equal(df.dtypes.index, pd.Index(list("abcdefghij")))

    # test series from tensor
    tensor = mt.random.rand(10, chunk_size=4)
    series = series_from_tensor(tensor, name="a")

    assert series.dtype == tensor.dtype
    assert series.name == "a"
    pd.testing.assert_index_equal(series.index_value.to_pandas(), pd.RangeIndex(10))

    d = OrderedDict(
        [(0, mt.tensor(np.random.rand(4))), (1, mt.tensor(np.random.rand(4)))]
    )
    df = dataframe_from_1d_tileables(d)
    pd.testing.assert_index_equal(df.columns_value.to_pandas(), pd.RangeIndex(2))

    # test axis parameter for dataframe_from_1d_tileables
    d = OrderedDict(
        [("a", mt.tensor(np.random.rand(4))), ("b", mt.tensor(np.random.rand(4)))]
    )

    # axis=1 (default behavior) - keys become columns
    df = dataframe_from_1d_tileables(d, axis=1)
    assert df.shape == (4, 2)
    pd.testing.assert_index_equal(df.columns_value.to_pandas(), pd.Index(["a", "b"]))
    pd.testing.assert_index_equal(df.index_value.to_pandas(), pd.RangeIndex(4))

    # axis=0 - keys become index (rows)
    df = dataframe_from_1d_tileables(d, axis=0)
    assert df.shape == (2, 4)
    pd.testing.assert_index_equal(df.index_value.to_pandas(), pd.Index(["a", "b"]))
    pd.testing.assert_index_equal(df.columns_value.to_pandas(), pd.RangeIndex(4))

    series = series_from_tensor(mt.random.rand(4))
    pd.testing.assert_index_equal(series.index_value.to_pandas(), pd.RangeIndex(4))

    series = series_from_tensor(mt.random.rand(4), index=[1, 2, 3])
    pd.testing.assert_index_equal(series.op.index, pd.Index([1, 2, 3]))

    series = series_from_tensor(
        mt.random.rand(4), index=pd.Index([1, 2, 3], name="my_index")
    )
    pd.testing.assert_index_equal(series.op.index, pd.Index([1, 2, 3], name="my_index"))
    assert series.index_value.name == "my_index"

    with pytest.raises(TypeError):
        series_from_tensor(mt.ones((10, 10)))

    # index has wrong shape
    with pytest.raises(ValueError):
        dataframe_from_tensor(mt.random.rand(4, 3), index=mt.random.rand(5))

    # columns have wrong shape
    with pytest.raises(ValueError):
        dataframe_from_tensor(mt.random.rand(4, 3), columns=["a", "b"])

    # 1-d tensors should have same shape
    with pytest.raises(ValueError):
        dataframe_from_1d_tileables(
            OrderedDict(
                [(0, mt.tensor(np.random.rand(3))), (1, mt.tensor(np.random.rand(2)))]
            )
        )

    # index has wrong shape
    with pytest.raises(ValueError):
        dataframe_from_1d_tileables(
            {0: mt.tensor(np.random.rand(3))}, index=mt.tensor(np.random.rand(2))
        )

    # columns have wrong shape
    with pytest.raises(ValueError):
        dataframe_from_1d_tileables(
            {0: mt.tensor(np.random.rand(3))}, columns=["a", "b"]
        )

    # index should be 1-d
    with pytest.raises(ValueError):
        dataframe_from_tensor(
            mt.tensor(np.random.rand(3, 2)), index=mt.tensor(np.random.rand(3, 2))
        )

    # 1-d tensors should have same shape
    with pytest.raises(ValueError):
        dataframe_from_1d_tileables(
            OrderedDict(
                [(0, mt.tensor(np.random.rand(3))), (1, mt.tensor(np.random.rand(2)))]
            )
        )

    # index has wrong shape
    with pytest.raises(ValueError):
        dataframe_from_1d_tileables(
            {0: mt.tensor(np.random.rand(3))}, index=mt.tensor(np.random.rand(2))
        )

    # columns have wrong shape
    with pytest.raises(ValueError):
        dataframe_from_1d_tileables(
            {0: mt.tensor(np.random.rand(3))}, columns=["a", "b"]
        )

    # index should be 1-d
    with pytest.raises(ValueError):
        series_from_tensor(mt.random.rand(4), index=mt.random.rand(4, 3))


def test_from_odps_table():
    odps_entry = ODPS.from_environments()
    table_name = tn("test_from_odps_table_src")
    odps_entry.delete_table(table_name, if_exists=True)
    test_table = odps_entry.create_table(
        table_name, "col1 string, col2 bigint, col3 double", lifecycle=1
    )

    part_table_name = tn("test_parted_from_odps_table_src")
    odps_entry.delete_table(part_table_name, if_exists=True)
    test_parted_table = odps_entry.create_table(
        part_table_name,
        ("col1 string, col2 bigint, col3 double", "pt string"),
        lifecycle=1,
    )

    df = read_odps_table(table_name)
    assert df.op.table_name == test_table.full_table_name
    assert df.index_value.name is None
    assert isinstance(df.index_value.value, IndexValue.RangeIndex)
    assert df.op.get_columns() == ["col1", "col2", "col3"]
    pd.testing.assert_series_equal(
        df.dtypes,
        pd.Series(
            [np.dtype("O"), np.dtype("int64"), np.dtype("float64")],
            index=["col1", "col2", "col3"],
        ),
    )

    result_ctx = dict()
    estimate_size(result_ctx, df.op)
    assert result_ctx[df.key] >= 0 and not isinf(result_ctx[df.key])

    with pytest.raises(ValueError):
        read_odps_table(test_table, columns=["col3", "col4"])
    with pytest.raises(ValueError):
        read_odps_table(test_table, index_col="col4")
    with pytest.raises(ValueError):
        read_odps_table(test_table, columns=["col1", "col2"], index_col="col2")

    df = read_odps_table(test_table, columns=["Col1", "col2"])
    assert df.op.table_name == test_table.full_table_name
    assert df.index_value.name is None
    assert isinstance(df.index_value.value, IndexValue.RangeIndex)
    assert df.op.get_columns() == ["col1", "col2"]
    pd.testing.assert_series_equal(
        df.dtypes,
        pd.Series([np.dtype("O"), np.dtype("int64")], index=["col1", "col2"]),
    )

    df = read_odps_table(test_table, index_col="col1")
    assert df.op.table_name == test_table.full_table_name
    assert df.index_value.name == "col1"
    assert isinstance(df.index_value.value, IndexValue.Index)
    assert df.index.dtype == np.dtype("O")
    assert df.op.get_columns() == ["col2", "col3"]
    pd.testing.assert_series_equal(
        df.dtypes,
        pd.Series([np.dtype("int64"), np.dtype("float64")], index=["col2", "col3"]),
    )

    with pytest.raises(ValueError):
        read_odps_table(test_table, append_partitions=True)

    df = read_odps_table(test_parted_table, append_partitions=True)
    assert df.op.append_partitions is True
    assert df.op.get_columns() == ["col1", "col2", "col3", "pt"]
    pd.testing.assert_series_equal(
        df.dtypes,
        pd.Series(
            [np.dtype("O"), np.dtype("int64"), np.dtype("float64"), np.dtype("O")],
            index=["col1", "col2", "col3", "pt"],
        ),
    )

    test_parted_table.create_partition("pt=20240103")
    df = read_odps_table(
        test_parted_table, columns=["col1", "col2", "pt"], partitions="pt=20240103"
    )
    assert df.op.append_partitions is True
    assert df.op.partitions == ["pt=20240103"]
    assert df.op.get_columns() == ["col1", "col2", "pt"]
    pd.testing.assert_series_equal(
        df.dtypes,
        pd.Series(
            [np.dtype("O"), np.dtype("int64"), np.dtype("O")],
            index=["col1", "col2", "pt"],
        ),
    )

    result_ctx = dict()
    estimate_size(result_ctx, df.op)
    assert result_ctx[df.key] >= 0 and not isinf(result_ctx[df.key])

    out_idx = read_odps_table(
        test_table,
        columns=[],
        index_col=["col1", "col2"],
        output_type=OutputType.index,
    )
    assert out_idx.names == ["col1", "col2"]
    assert out_idx.shape == (np.nan,)

    test_table.drop()
    test_parted_table.drop()


def test_from_odps_query():
    odps_entry = ODPS.from_environments()
    table1_name = tn("test_from_odps_query_src1")
    table2_name = tn("test_from_odps_query_src2")
    odps_entry.delete_table(table1_name, if_exists=True)
    odps_entry.delete_table(table2_name, if_exists=True)
    test_table = odps_entry.create_table(
        table1_name, "col1 string, col2 bigint, col3 double", lifecycle=1
    )
    # need some data to produce complicated plans
    odps_entry.write_table(test_table, [["A", 10, 3.5]])
    test_table2 = odps_entry.create_table(
        table2_name, "col1 string, col2 bigint, col3 double", lifecycle=1
    )
    odps_entry.write_table(test_table2, [["A", 10, 4.5]])

    with pytest.raises(ValueError) as err_info:
        read_odps_query(
            f"CREATE TABLE dummy_table_{uuid.uuid4().hex} LIFECYCLE 1 "
            f"AS SELECT * FROM {table1_name}"
        )
    assert "instant query" in err_info.value.args[0]

    # test simple select query
    query1 = f"SELECT * FROM {table1_name} WHERE col1 > 10"
    df = read_odps_query(query1, no_split_sql=True)
    assert df.op.query == query1
    assert df.op.extra_params.no_split_sql is True
    assert df.index_value.name is None
    assert isinstance(df.index_value.value, IndexValue.RangeIndex)
    pd.testing.assert_series_equal(
        df.dtypes,
        pd.Series(
            [np.dtype("O"), np.dtype("int64"), np.dtype("float64")],
            index=["col1", "col2", "col3"],
        ),
    )
    assert df.op.get_columns() == ["col1", "col2", "col3"]

    df = read_odps_query(query1, skip_schema=True)
    assert df.dtypes is None
    assert df.columns_value is None
    assert df.op.extra_params.no_split_sql is False

    df = read_odps_query(query1, index_col="col1")
    assert df.op.query == query1
    assert df.index_value.name == "col1"
    assert df.op.get_columns() == ["col2", "col3"]
    assert isinstance(df.index_value.value, IndexValue.Index)
    pd.testing.assert_series_equal(
        df.dtypes,
        pd.Series([np.dtype("int64"), np.dtype("float64")], index=["col2", "col3"]),
    )

    # test query with join
    query2 = f"""
    SELECT t1.col1, t1.col2, t1.col3 as c31, t2.col3 as c32
    FROM {table1_name} t1
    INNER JOIN {table2_name} t2
    ON t1.col1 = t2.col1 AND t1.col2 = t2.col2
    """
    df = read_odps_query(query2, index_col=["col1", "col2"])
    assert df.op.query == query2
    assert df.index_value.names == ["col1", "col2"]
    assert df.op.get_columns() == ["c31", "c32"]
    assert isinstance(df.index_value.value, IndexValue.MultiIndex)
    pd.testing.assert_series_equal(
        df.dtypes,
        pd.Series([np.dtype("float64"), np.dtype("float64")], index=["c31", "c32"]),
    )

    # test query with multiple statements
    query3 = f"""
    @val := SELECT t1.col1, t1.col2 as c1, t1.col3 as c31, t1.col3 as c32
    FROM {table1_name} t1;
    SELECT c1, c32 FROM @val;
    """
    df = read_odps_query(query3, index_col=["c1"])
    assert df.op.query == query3
    assert df.op.extra_params.no_split_sql is False
    assert df.index_value.names == ["c1"]
    assert df.op.get_columns() == ["c32"]
    pd.testing.assert_series_equal(
        df.dtypes,
        pd.Series([np.dtype("float64")], index=["c32"]),
    )

    test_table.drop()
    test_table2.drop()


def test_date_range():
    with pytest.raises(TypeError):
        _ = date_range("2020-1-1", periods="2")

    with pytest.raises(ValueError):
        _ = date_range("2020-1-1", "2020-1-10", periods=10, freq="D")

    with pytest.raises(ValueError):
        _ = date_range(pd.NaT, periods=10)

    expected = pd.date_range("2020-1-1", periods=9.0, name="date")

    dr = date_range("2020-1-1", periods=9.0, name="date", chunk_size=3)
    assert isinstance(dr, DatetimeIndex)
    assert dr.shape == (9,)
    assert dr.dtype == expected.dtype
    assert isinstance(dr.index_value.value, IndexValue.DatetimeIndex)
    assert dr.index_value.min_val == expected.min()
    assert dr.index_value.min_val_close is True
    assert dr.index_value.max_val == expected.max()
    assert dr.index_value.max_val_close is True
    assert dr.index_value.is_unique == expected.is_unique
    assert dr.index_value.is_monotonic_increasing == expected.is_monotonic_increasing
    assert dr.name == expected.name


def test_resolve_task_sector():
    input_path = os.path.join(
        os.path.dirname(__file__), "test-data", "task-input-full.txt"
    )
    with open(input_path, "r") as f:
        sector = f.read()
    actual_sector = _resolve_task_sector("job0", sector)

    assert actual_sector.job_name == "job0"
    assert actual_sector.task_name == "M1"
    assert actual_sector.output_target == "Screen"
    assert len(actual_sector.schema) == 78
    assert actual_sector.schema[0] == ColumnSchema("unnamed: 0", "bigint", "")
    assert actual_sector.schema[1] == ColumnSchema("id", "bigint", "id_alias")
    assert actual_sector.schema[2] == ColumnSchema("listing_url", "string", "")


def test_resolve_task_odps2():
    input_path = os.path.join(
        os.path.dirname(__file__), "test-data", "task-input-odps2.txt"
    )
    with open(input_path, "r") as f:
        sector = f.read()
    actual_sector = _resolve_task_sector("job0", sector)

    assert actual_sector.job_name == "job0"
    assert actual_sector.task_name == "M1"
    assert actual_sector.output_target == "Screen"
    assert len(actual_sector.schema) == 2
    assert actual_sector.schema[0] == ColumnSchema("key", "varchar(2048)", "")
    assert actual_sector.schema[1] == ColumnSchema("data", "binary", "")


def test_resolve_simple_explain():
    input_path = os.path.join(
        os.path.dirname(__file__), "test-data", "task-input-simple.txt"
    )
    with open(input_path, "r") as f:
        sector = f.read()

    schema = _parse_simple_explain(sector)
    assert schema.columns[0].name == "memberid"
    assert schema.columns[0].type == odps_types.string
    assert schema.columns[1].name == "createdate"
    assert schema.columns[1].type == odps_types.bigint


def test_resolve_conditional():
    input_path = os.path.join(
        os.path.dirname(__file__), "test-data", "task-input-multi-cond.txt"
    )
    with open(input_path, "r") as f:
        sector = f.read()

    expected_col_types = {
        "cs1": "string",
        "cs2": "string",
        "ci1": "bigint",
        "cs3": "string",
        "cs4": "string",
        "cs5": "string",
        "cs6": "string",
        "cs7": "string",
        "cs8": "string",
        "ci2": "int",
        "ci3": "bigint",
        "cs9": "string",
    }

    schema = _parse_full_explain(sector)
    for col, (exp_nm, exp_tp) in zip(schema.columns, expected_col_types.items()):
        assert col.name == exp_nm
        assert col.type == odps_types.validate_data_type(exp_tp)


def test_resolve_multi_join():
    input_path = os.path.join(
        os.path.dirname(__file__), "test-data", "task-input-multi-join.txt"
    )
    with open(input_path, "r") as f:
        sector = f.read()

    expected_col_types = {
        "cs1": "string",
        "cs2": "string",
        "ci1": "bigint",
        "cs3": "string",
        "cs4": "string",
        "cs5": "string",
        "cs6": "string",
        "cs7": "string",
        "cs8": "string",
        "ci2": "int",
        "ci3": "bigint",
        "cs9": "string",
    }

    schema = _parse_full_explain(sector)
    for col, (exp_nm, exp_tp) in zip(schema.columns, expected_col_types.items()):
        assert col.name == exp_nm
        assert col.type == odps_types.validate_data_type(exp_tp)


def test_resolve_break_lines():
    input_path = os.path.join(
        os.path.dirname(__file__), "test-data", "task-input-with-break-line.txt"
    )
    with open(input_path, "r") as f:
        sector = f.read()

    expected_col_types = {
        "key": "string",
        "value": "string",
    }

    schema = _parse_full_explain(sector)
    for col, (exp_nm, exp_tp) in zip(schema.columns, expected_col_types.items()):
        assert col.name == exp_nm
        assert col.type == odps_types.validate_data_type(exp_tp)


@pytest.mark.parametrize("use_explain_output", [None, False, True])
def test_explain_use_explain_output(use_explain_output):
    class MockInstance:
        @property
        def id(self):
            return "mock_id"

        def get_task_results(self):
            return {"pot": """{"columns":[{"name":"a_bigint","type":"BIGINT"}]}"""}

    old_execute_sql = ODPS.execute_sql
    exec_count = 0

    def new_execute_sql(self, sql, *args, **kw):
        nonlocal exec_count
        exec_count += 1

        if use_explain_output and sql.lower().startswith("explain output select"):
            return MockInstance()
        elif use_explain_output is None and sql.lower().startswith("explain output"):
            raise ODPSError("ODPS-0130161: mock error")
        return old_execute_sql(self, sql, *args, **kw)

    odps_entry = ODPS.from_environments()

    with mock.patch("odps.core.ODPS.execute_sql", new=new_execute_sql):
        with pytest.raises(ValueError):
            _resolve_query_schema(
                odps_entry, "not_a_sql", use_explain_output=use_explain_output
            )
        assert exec_count == (2 if use_explain_output is None else 1)

        exec_count = 0
        schema = _resolve_query_schema(
            odps_entry,
            "select cast(1 as bigint) as a_bigint",
            use_explain_output=use_explain_output,
        )
        assert schema.columns[0].name == "a_bigint"
        assert schema.columns[0].type == odps_types.bigint
        assert exec_count == (2 if use_explain_output is None else 1)
