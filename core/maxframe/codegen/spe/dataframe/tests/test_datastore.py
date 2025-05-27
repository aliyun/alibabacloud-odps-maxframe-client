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

from typing import Optional

import pandas as pd
import pytest
from odps import ODPS

from ..... import dataframe as md
from .....config import options
from .....io.odpsio import HaloTableIO
from .....tests.utils import flaky, tn
from ...core import SPECodeContext
from ..datastore import DataFrameToODPSTableAdapter


@pytest.fixture
def df1():
    return md.DataFrame(
        [[1, 10, "A"], [2, 20, "A"], [3, 30, "B"], [4, 40, "B"]],
        index=list("abcd"),
        columns=list("abc"),
    )


@pytest.fixture
def df2():
    return md.DataFrame(
        [[1, 10, "A"], [2, 20, "A"], [3, 30, "B"], [4, 40, "B"]],
        index=pd.MultiIndex.from_arrays([["A", "A", "B", "B"], [1, 2, 3, 4]]),
        columns=list("abc"),
    )


@pytest.fixture(autouse=True)
def auto_use_common_table():
    try:
        options.use_common_table = True
        yield
    finally:
        options.use_common_table = False


def _run_generated_code(
    code: str, ctx: SPECodeContext, input_val: pd.DataFrame
) -> dict:
    local_vars = ctx.constants.copy()
    local_vars["DataFrameToODPSTableAdapter"] = DataFrameToODPSTableAdapter
    local_vars["var_0"] = input_val
    exec(code, local_vars, local_vars)
    return local_vars


def _table_to_dataframe(
    table_name: str, partition: Optional[str] = None
) -> pd.DataFrame:
    odps_entry = ODPS.from_environments()
    halo_table_io = HaloTableIO(odps_entry)
    with halo_table_io.open_reader(table_name, partition) as reader:
        return reader.read_all().to_pandas()


@flaky(max_runs=3)
def test_to_odps_no_partition_no_index(df1):
    odps_entry = ODPS.from_environments()
    table_name = tn("test_to_odps_no_part_spe")
    odps_entry.delete_table(table_name, if_exists=True)

    res = md.to_odps_table(df1, table_name, index=False)
    ctx = SPECodeContext()
    results = DataFrameToODPSTableAdapter().generate_code(res.op, ctx)
    expected_result = (
        f"DataFrameToODPSTableAdapter._write_as_table(var_0, '{table_name}', "
        f"None, None, False, None, overwrite=False, lifecycle=None, table_properties=None)"
    )
    assert results[0] == expected_result
    _run_generated_code(results[0], ctx, df1.op.get_data())
    pd.testing.assert_frame_equal(
        df1.op.get_data().reset_index(drop=True), _table_to_dataframe(table_name)
    )
    odps_entry.delete_table(table_name, if_exists=True)


@flaky(max_runs=3)
def test_to_odps_no_partition_index(df1):
    odps_entry = ODPS.from_environments()
    table_name = tn("test_to_odps_no_part_idx_spe")
    odps_entry.delete_table(table_name, if_exists=True)

    res = md.to_odps_table(df1, table_name)
    ctx = SPECodeContext()
    results = DataFrameToODPSTableAdapter().generate_code(res.op, ctx)
    expected_result = (
        f"DataFrameToODPSTableAdapter._write_as_table(var_0, '{table_name}', "
        f"None, None, True, None, overwrite=False, lifecycle=None, table_properties=None)"
    )
    assert results[0] == expected_result
    _run_generated_code(results[0], ctx, df1.op.get_data())
    pd.testing.assert_frame_equal(
        df1.op.get_data(),
        _table_to_dataframe(table_name).set_index("index").rename_axis(None),
    )
    odps_entry.delete_table(table_name, if_exists=True)


@flaky(max_runs=3)
def test_to_odps_no_partition_multi_index(df2):
    odps_entry = ODPS.from_environments()
    table_name = tn("test_to_odps_no_part_multi_idx_spe")
    odps_entry.delete_table(table_name, if_exists=True)

    res = md.to_odps_table(df2, table_name)
    ctx = SPECodeContext()
    results = DataFrameToODPSTableAdapter().generate_code(res.op, ctx)
    expected_result = (
        f"DataFrameToODPSTableAdapter._write_as_table(var_0, '{table_name}', "
        f"None, None, True, None, overwrite=False, lifecycle=None, table_properties=None)"
    )
    assert results[0] == expected_result
    _run_generated_code(results[0], ctx, df2.op.get_data())
    pd.testing.assert_frame_equal(
        df2.op.get_data(),
        _table_to_dataframe(table_name)
        .set_index(["level_0", "level_1"])
        .rename_axis([None, None]),
    )
    odps_entry.delete_table(table_name, if_exists=True)


@pytest.mark.parametrize("create", [False, True])
@flaky(max_runs=3)
def test_to_odps_part_index(df1, create):
    odps_entry = ODPS.from_environments()
    existing_tag = "existing_" if create else ""
    table_name = tn(f"test_to_odps_part_index_{existing_tag}spe")
    odps_entry.delete_table(table_name, if_exists=True)

    if create:
        odps_entry.create_table(
            table_name, ("index string, a bigint, b bigint", "pt string, c string")
        )

    res = md.to_odps_table(df1, table_name, partition="pt=20240109", partition_col="c")
    ctx = SPECodeContext()
    results = DataFrameToODPSTableAdapter().generate_code(res.op, ctx)
    expected_result = (
        f"DataFrameToODPSTableAdapter._write_as_table(var_0, '{table_name}', "
        f"'pt=20240109', ['c'], True, None, overwrite=False, lifecycle=None, table_properties=None)"
    )
    assert results[0] == expected_result
    _run_generated_code(results[0], ctx, df1.op.get_data())

    data_parts = []
    for c_value in df1.op.get_data().c.unique():
        result_df = _table_to_dataframe(table_name, f"pt=20240109,c={c_value}")
        result_df["c"] = c_value
        data_parts.append(result_df)
    result_df = pd.concat(data_parts)

    pd.testing.assert_frame_equal(
        df1.op.get_data(), result_df.set_index("index").rename_axis(None)
    )
    odps_entry.delete_table(table_name, if_exists=True)


@pytest.mark.parametrize(
    "idx, col_def, pt_def",
    [
        (0, "index string, a bigint, b bigint", "pt string"),
        (1, "index string, a bigint, b bigint, d bigint", "pt string, c string"),
    ],
)
@flaky(max_runs=3)
def test_schema_mismatch(df1, idx, col_def, pt_def):
    odps_entry = ODPS.from_environments()
    table_name = tn(f"test_schema_mismatch_{idx}")
    odps_entry.delete_table(table_name, if_exists=True)

    odps_entry.create_table(table_name, (col_def, pt_def))
    with pytest.raises(ValueError):
        res = md.to_odps_table(
            df1, table_name, partition="pt=20240109", partition_col="c"
        )
        ctx = SPECodeContext()
        results = DataFrameToODPSTableAdapter().generate_code(res.op, ctx)
        _run_generated_code(results[0], ctx, df1.op.get_data())

    odps_entry.delete_table(table_name, if_exists=True)
