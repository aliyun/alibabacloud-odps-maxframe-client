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

import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
from odps import ODPS

import maxframe.dataframe as md
from maxframe.odpsio import HaloTableIO
from maxframe.protocol import ODPSTableResultInfo, ResultType
from maxframe.tests.utils import tn

from ..fetcher import ODPSTableFetcher


async def test_table_fetcher():
    odps_entry = ODPS.from_environments()
    halo_table_io = HaloTableIO(odps_entry)
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
    with halo_table_io.open_writer(table_name) as writer:
        writer.write(pa.Table.from_pandas(data, preserve_index=False))

    raw_data = data[list("abc")]

    tileable = md.read_pandas(data[list("abc")])
    result_info = ODPSTableResultInfo(ResultType.ODPS_TABLE, full_table_name=table_name)
    fetched = await fetcher.fetch(tileable, result_info, [None, None])
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
