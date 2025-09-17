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

import uuid

import mock
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from odps import ODPS
from odps.errors import ODPSError

import maxframe.dataframe as md
from maxframe.config import options
from maxframe.io.odpsio import ODPSTableIO
from maxframe.protocol import ODPSTableResultInfo, ResultType
from maxframe.tests.utils import tn

from ..fetcher import ODPSTableFetcher


@pytest.fixture
def switch_table_io(request):
    old_use_common_table = options.use_common_table
    try:
        options.use_common_table = request.param
        yield
    finally:
        options.use_common_table = old_use_common_table


@pytest.mark.parametrize("switch_table_io", [False, True], indirect=True)
async def test_fetch_table_data(switch_table_io):
    odps_entry = ODPS.from_environments()
    table_io = ODPSTableIO(odps_entry)
    fetcher = ODPSTableFetcher(odps_entry)

    data = pd.DataFrame(
        {
            "_idx_0": np.arange(1000),
            "a": np.random.rand(1000),
            "b": np.random.randint(0, 10, 1000),
            "c": np.random.choice(list("ABC"), 1000),
        }
    )

    table_name = tn("mf_test_groupby_table_" + str(uuid.uuid4().hex))
    odps_entry.delete_table(table_name, if_exists=True)

    odps_entry.create_table(
        table_name, "_idx_0 bigint, a double, b bigint, c string", lifecycle=1
    )
    with table_io.open_writer(table_name) as writer:
        writer.write(pa.Table.from_pandas(data, preserve_index=False))

    raw_data = data[list("abc")]

    tileable = md.read_pandas(data[list("abc")])
    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [None, None])
    assert len(fetched) == 1000
    pd.testing.assert_frame_equal(raw_data, fetched)

    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [slice(None, 2000), None])
    assert len(fetched) == 1000
    pd.testing.assert_frame_equal(raw_data, fetched)

    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [2, None])
    assert len(fetched) == 1
    pd.testing.assert_frame_equal(raw_data.iloc[2:3], fetched)

    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [slice(None, 10), None])
    assert len(fetched) == 10
    pd.testing.assert_frame_equal(raw_data.iloc[:10], fetched)

    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [slice(None, 10, 3), None])
    assert len(fetched) == 4
    pd.testing.assert_frame_equal(raw_data.iloc[:10:3], fetched)

    tileable = md.read_pandas(data[list("ab")])
    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [slice(-5, None), slice(2)])
    assert len(fetched) == 5
    pd.testing.assert_frame_equal(raw_data.iloc[-5:, :2], fetched)

    tileable = md.read_pandas(data[list("a")])
    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [slice(-1, -6, -1), 0])
    assert len(fetched) == 5
    assert len(fetched.columns) == 1
    pd.testing.assert_frame_equal(raw_data.iloc[-1:-6:-1, :1], fetched)

    odps_entry.delete_table(table_name, if_exists=True)


async def test_fetch_data_shape():
    odps_entry = ODPS.from_environments()
    table_io = ODPSTableIO(odps_entry)
    fetcher = ODPSTableFetcher(odps_entry)

    data = pd.DataFrame(
        {
            "_idx_0": np.arange(1000),
            "a": np.random.rand(1000),
            "b": np.random.randint(0, 10, 1000),
            "c": np.random.choice(list("ABC"), 1000),
        }
    )

    table_names = [
        tn("mf_test_groupby_table_" + str(uuid.uuid4().hex)) for _ in range(3)
    ]
    pt_specs = [None, None, ["pt='2020-01-01'"]]
    for tn_to_check in table_names:
        odps_entry.delete_table(tn_to_check, if_exists=True)

    odps_entry.create_table(
        table_names[0], "_idx_0 bigint, a double, b bigint, c string", lifecycle=1
    )
    odps_entry.create_table(
        table_names[1], "_idx_0 bigint, a double, b bigint, c string", lifecycle=1
    )
    odps_entry.create_table(
        table_names[2],
        ("_idx_0 bigint, a double, b bigint, c string", "pt string"),
        lifecycle=1,
    )
    with table_io.open_writer(table_names[0]) as writer:
        writer.write(pa.Table.from_pandas(data, preserve_index=False))

    odps_entry.execute_sql(
        f"insert overwrite table {table_names[1]} select * from {table_names[0]}"
    )
    odps_entry.execute_sql(
        f"insert overwrite table {table_names[2]} partition ({pt_specs[-1][0]}) "
        f"select * from {table_names[0]}"
    )

    for tn_to_check, pt_spec in zip(table_names, pt_specs):
        tileable = md.read_pandas(data[list("abc")])
        tileable._shape = (np.nan, 3)
        result_info = ODPSTableResultInfo(
            ResultType.ODPS_TABLE, full_table_name=tn_to_check, partition_specs=pt_spec
        )
        await fetcher.update_tileable_meta(tileable, result_info)
        assert tileable.shape == (1000, 3)

    def _create_session_with_error(*_, **__):
        raise ODPSError("StatusConflict: mock")

    with mock.patch(
        "maxframe.io.odpsio.tableio.TunnelTableIO.create_download_sessions",
        new=_create_session_with_error,
    ):
        tileable = md.read_pandas(data[list("abc")])
        tileable._shape = (np.nan, 3)
        result_info = ODPSTableResultInfo(
            ResultType.ODPS_TABLE, full_table_name=table_names[0]
        )
        await fetcher.update_tileable_meta(tileable, result_info)
        assert tileable.shape == (np.nan, 3)

    for tn_to_check in table_names:
        odps_entry.delete_table(tn_to_check, if_exists=True, wait=False)
