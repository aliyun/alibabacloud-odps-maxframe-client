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

import mock
import numpy as np
import pandas as pd
import pyarrow as pa
from odps import ODPS

from maxframe import dataframe as md
from maxframe import tensor as mt
from maxframe.codegen.core import EngineAcceptance
from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.dataframe.datasource import (
    DataFrameDateRangeAdapter,
    DataFrameFromRecordsAdapter,
    DataFrameReadODPSQueryAdapter,
    DataFrameReadODPSTableAdapter,
    IndexDataSourceAdapter,
    SeriesFromTensorAdapter,
)
from maxframe.config import option_context
from maxframe.dataframe.datasource.index import from_pandas
from maxframe.dataframe.datasource.read_odps_query import DataFrameReadODPSQuery
from maxframe.io.odpsio import ODPSTableIO
from maxframe.tests.utils import flaky, tn


def _run_generated_code(code: str, ctx: SPECodeContext) -> dict:
    local_vars = ctx.constants.copy()
    local_vars["DataFrameReadODPSQueryAdapter"] = DataFrameReadODPSQueryAdapter
    local_vars["DataFrameReadODPSTableAdapter"] = DataFrameReadODPSTableAdapter
    exec(code, local_vars, local_vars)
    return local_vars


def test_date_range():
    rng = md.date_range(start="1/1/2018", end="1/08/2018")
    context = SPECodeContext()
    adapter = DataFrameDateRangeAdapter()
    results = adapter.generate_code(rng.op, context)
    assert "pd" in context.imports["pandas"]
    expected_results = [
        "var_0 = pd.date_range(start=pd.Timestamp('2018-01-01 00:00:00'), "
        "end=pd.Timestamp('2018-01-08 00:00:00'), periods=8, freq=const_0, "
        "normalize=False, inclusive='both')"
    ]
    assert results == expected_results


def test_from_records():
    data = mt.array(
        [(3, "a"), (2, "b"), (1, "c"), (0, "d")],
        dtype=[("col_1", "i4"), ("col_2", "U1")],
    )
    df = md.from_records(data)
    context = SPECodeContext()
    adapter = DataFrameFromRecordsAdapter()
    results = adapter.generate_code(df.op, context)
    assert "pd" in context.imports["pandas"]
    expected_results = ["var_1 = pd.DataFrame.from_records(var_0, coerce_float=False)"]
    assert results == expected_results


def test_series_from_tensor():
    s = md.Series(["a", "b", "c", "d"], index=md.Index(["p", "q", "r", "t"]))
    context = SPECodeContext()
    adapter = SeriesFromTensorAdapter()
    results = adapter.generate_code(s.op, context)
    assert "np" in context.imports["numpy"]
    assert "pd" in context.imports["pandas"]
    expected_results = [
        "var_2 = pd.Series(var_0, index=var_1, name=None, dtype=np.dtype('<U1'))"
    ]
    assert results == expected_results


@flaky(max_runs=3)
def test_read_odps_table_simple_table():
    odps_entry = ODPS.from_environments()
    halo_table_io = ODPSTableIO(odps_entry)
    table_name = tn("test_read_odps_table_simple_table_spe")

    odps_entry.delete_table(table_name, if_exists=True)
    table = odps_entry.create_table(
        table_name, ",".join(f"{c} double" for c in "abcde"), lifecycle=1
    )
    pd_data = pd.DataFrame(np.random.rand(100, 5), columns=list("abcde"))
    with halo_table_io.open_writer(table_name) as writer:
        writer.write(pa.Table.from_pandas(pd_data, preserve_index=False))

    df = md.read_odps_table(table_name)
    ctx = SPECodeContext()
    results = DataFrameReadODPSTableAdapter().generate_code(df.op, ctx)
    expected_results = [
        f"var_0 = DataFrameReadODPSTableAdapter._read_as_pandas("
        f"'{table.full_table_name}', None, None, None, False, const_0)"
    ]
    assert results == expected_results
    exec_results = _run_generated_code(results[0], ctx)
    pd.testing.assert_frame_equal(pd_data, exec_results["var_0"])

    df = md.read_odps_table(table_name, index_col="a")
    ctx = SPECodeContext()
    results = DataFrameReadODPSTableAdapter().generate_code(df.op, ctx)
    expected_results = [
        f"var_0 = DataFrameReadODPSTableAdapter._read_as_pandas("
        f"'{table.full_table_name}', None, ['b', 'c', 'd', 'e'], ['a'], False, const_0)"
    ]
    assert results == expected_results
    exec_results = _run_generated_code(results[0], ctx)
    pd.testing.assert_frame_equal(pd_data.set_index("a"), exec_results["var_0"])

    df = md.read_odps_table(table_name, columns=["a", "b"])
    ctx = SPECodeContext()
    results = DataFrameReadODPSTableAdapter().generate_code(df.op, ctx)
    expected_results = [
        f"var_0 = DataFrameReadODPSTableAdapter._read_as_pandas("
        f"'{table.full_table_name}', None, ['a', 'b'], None, False, const_0)"
    ]
    assert results == expected_results
    exec_results = _run_generated_code(results[0], ctx)
    pd.testing.assert_frame_equal(pd_data[["a", "b"]], exec_results["var_0"])

    table.drop()


@flaky(max_runs=3)
def test_read_odps_table_with_parts():
    odps_entry = ODPS.from_environments()
    halo_table_io = ODPSTableIO(odps_entry)
    table_name = tn("test_read_odps_table_with_parts_spe")

    odps_entry.delete_table(table_name, if_exists=True)
    table = odps_entry.create_table(
        table_name, (",".join(f"{c} double" for c in "abcde"), "pt string"), lifecycle=1
    )
    pd_data = pd.DataFrame(np.random.rand(100, 5), columns=list("abcde"))
    with halo_table_io.open_writer(table_name, partition="pt=20240309") as writer:
        writer.write(pa.Table.from_pandas(pd_data, preserve_index=False))

    df = md.read_odps_table(table_name, partition="pt=20240309")
    ctx = SPECodeContext()
    results = DataFrameReadODPSTableAdapter().generate_code(df.op, ctx)
    expected_results = [
        f"var_0 = DataFrameReadODPSTableAdapter._read_as_pandas("
        f"'{table.full_table_name}', ['pt=20240309'], None, None, False, const_0)"
    ]
    assert results == expected_results
    exec_results = _run_generated_code(results[0], ctx)
    pd.testing.assert_frame_equal(pd_data, exec_results["var_0"])

    df = md.read_odps_table(table_name, append_partitions=True)
    ctx = SPECodeContext()
    results = DataFrameReadODPSTableAdapter().generate_code(df.op, ctx)
    expected_results = [
        f"var_0 = DataFrameReadODPSTableAdapter._read_as_pandas("
        f"'{table.full_table_name}', None, None, None, True, const_0)"
    ]
    assert results == expected_results
    exec_results = _run_generated_code(results[0], ctx)
    expected = pd_data.copy()
    expected["pt"] = "20240309"
    pd.testing.assert_frame_equal(expected, exec_results["var_0"])

    table.drop()


def test_read_odps_query_generate_code():
    query = "select a, b, c from src_table"
    dtypes = pd.Series(
        [np.dtype("float64"), np.dtype("object")],
        index=["b", "c"],
    )
    index_dtypes = pd.Series([np.dtype("int64")], index=["a"])
    op = DataFrameReadODPSQuery(
        query=query,
        dtypes=dtypes,
        index_columns=["a"],
        index_dtypes=index_dtypes,
        column_renames={},
    )
    df = op()
    ctx = SPECodeContext()
    results = DataFrameReadODPSQueryAdapter().generate_code(df.op, ctx)
    expected_results = [
        "var_0 = DataFrameReadODPSQueryAdapter._read_as_pandas("
        "'select a, b, c from src_table', None, ['a'], {}, None, const_0)"
    ]
    assert results == expected_results


class _MockQueryReader:
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def to_pandas(self):
        return self._data.copy()


class _MockQueryInstance:
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def open_reader(self, tunnel=True):
        return _MockQueryReader(self._data)


class _MockODPS:
    def __init__(self, data: pd.DataFrame, schema_namespace_enabled: bool = False):
        self._data = data
        self._schema_namespace_enabled = schema_namespace_enabled
        self.query = None
        self.hints = None

    def is_schema_namespace_enabled(self):
        return self._schema_namespace_enabled

    def execute_sql(self, query: str, **kwargs):
        self.query = query
        self.hints = kwargs.get("hints")
        return _MockQueryInstance(self._data)


@mock.patch("maxframe.codegen.spe.dataframe.datasource.ODPS")
def test_read_odps_query_as_pandas_without_index(mock_odps):
    query_data = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    mock_odps_entry = _MockODPS(query_data)
    mock_odps.from_environments.return_value = mock_odps_entry

    op = DataFrameReadODPSQuery(
        query="select a, b from src_table",
        dtypes=query_data.dtypes,
        column_renames={},
    )
    df = op()
    table_meta = SPECodeContext().get_pandas_data_table_meta(df)
    result = DataFrameReadODPSQueryAdapter._read_as_pandas(
        df.op.query,
        df.op.columns,
        df.op.index_columns,
        df.op.column_renames,
        df.op.nrows,
        table_meta,
    )

    pd.testing.assert_frame_equal(query_data, result)
    assert mock_odps_entry.query == "select a, b from src_table"
    assert mock_odps_entry.hints == {
        "odps.sql.submit.mode": "script",
        "odps.sql.object.table.split.by.object.size.enabled": "false",
    }


@mock.patch("maxframe.codegen.spe.dataframe.datasource.ODPS")
def test_read_odps_query_as_pandas_with_index_and_columns(mock_odps):
    query_data = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0],
            "c": ["x", "y", "z"],
        }
    )
    mock_odps.from_environments.return_value = _MockODPS(query_data)

    dtypes = pd.Series([query_data.dtypes["c"]], index=["c"])
    index_dtypes = pd.Series([query_data.dtypes["a"]], index=["a"])
    op = DataFrameReadODPSQuery(
        query="select a, b, c from src_table",
        dtypes=dtypes,
        columns=["c"],
        index_columns=["a"],
        index_dtypes=index_dtypes,
        column_renames={},
    )
    df = op()
    table_meta = SPECodeContext().get_pandas_data_table_meta(df)
    result = DataFrameReadODPSQueryAdapter._read_as_pandas(
        df.op.query,
        df.op.columns,
        df.op.index_columns,
        df.op.column_renames,
        df.op.nrows,
        table_meta,
    )

    expected = query_data.set_index("a")[["c"]]
    pd.testing.assert_frame_equal(expected, result)


@mock.patch("maxframe.codegen.spe.dataframe.datasource.ODPS")
def test_read_odps_query_as_pandas_with_nrows(mock_odps):
    query_data = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    mock_odps.from_environments.return_value = _MockODPS(query_data)

    op = DataFrameReadODPSQuery(
        query="select a, b from src_table",
        dtypes=query_data.dtypes,
        column_renames={},
        nrows=2,
    )
    df = op()
    ctx = SPECodeContext()
    results = DataFrameReadODPSQueryAdapter().generate_code(df.op, ctx)
    expected_results = [
        "var_0 = DataFrameReadODPSQueryAdapter._read_as_pandas("
        "'select a, b from src_table', None, None, {}, 2, const_0)"
    ]
    assert results == expected_results

    exec_results = _run_generated_code(results[0], ctx)
    pd.testing.assert_frame_equal(query_data.iloc[:2], exec_results["var_0"])


def test_read_odps_query_build_hints_with_schema():
    mock_odps_entry = _MockODPS(pd.DataFrame(), schema_namespace_enabled=True)
    with option_context({"sql.settings": {"odps.sql.type.system.odps2": "true"}}):
        hints = DataFrameReadODPSQueryAdapter._build_hints(mock_odps_entry)

    assert hints == {
        "odps.sql.type.system.odps2": "true",
        "odps.namespace.schema": "true",
        "odps.sql.allow.namespace.schema": "true",
        "odps.sql.submit.mode": "script",
        "odps.sql.object.table.split.by.object.size.enabled": "false",
    }


def test_index_data_source():
    pd_idx = pd.MultiIndex.from_tuples([(1, 1), (2, 2), (3, 3)], names=["x", "y"])
    idx1 = from_pandas(pd_idx)
    context = SPECodeContext()
    adapter = IndexDataSourceAdapter()
    assert adapter.accepts(idx1.op) == EngineAcceptance.ACCEPT
    results = adapter.generate_code(idx1.op, context)
    expected_results = ["var_0 = const_0"]
    assert results == expected_results
    assert context.constants["const_0"] is pd_idx
