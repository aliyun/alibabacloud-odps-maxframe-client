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
from odps import ODPS

from ..... import dataframe as md
from ..... import tensor as mt
from .....dataframe.datasource.index import from_pandas
from .....io.odpsio import ODPSTableIO
from .....tests.utils import flaky, tn
from ....core import EngineAcceptance
from ...core import SPECodeContext
from ..datasource import (
    DataFrameDateRangeAdapter,
    DataFrameFromRecordsAdapter,
    DataFrameReadODPSTableAdapter,
    IndexDataSourceAdapter,
    SeriesFromTensorAdapter,
)


def _run_generated_code(code: str, ctx: SPECodeContext) -> dict:
    local_vars = ctx.constants.copy()
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
