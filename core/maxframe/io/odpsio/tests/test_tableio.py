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

import datetime

import mock
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from odps import ODPS
from odps.errors import TableModified
from odps.models import Table

from ....config import options
from ....tests.utils import flaky, tn
from ....utils import config_odps_default_options
from .. import TunnelTableIO
from ..tableio import ODPSTableIO


@pytest.fixture
def switch_table_io(request):
    old_use_common_table = options.use_common_table
    try:
        options.use_common_table = request.param
        yield request.param
    finally:
        options.use_common_table = old_use_common_table


@flaky(max_runs=3)
@pytest.mark.parametrize("switch_table_io", [False, True], indirect=True)
def test_empty_table_io(switch_table_io):
    config_odps_default_options()

    o = ODPS.from_environments()
    table_io = ODPSTableIO(o)

    # test read from empty table
    empty_table_name = tn("test_empty_table_halo_read_" + str(switch_table_io).lower())
    o.delete_table(empty_table_name, if_exists=True)
    tb = o.create_table(empty_table_name, "col1 string", lifecycle=1)

    try:
        with table_io.open_reader(empty_table_name) as reader:
            assert len(reader.read_all()) == 0
    finally:
        tb.drop()


@flaky(max_runs=3)
@pytest.mark.parametrize("switch_table_io", [False, True], indirect=True)
def test_table_io_without_parts(switch_table_io):
    config_odps_default_options()

    o = ODPS.from_environments()
    table_io = ODPSTableIO(o)

    # test read and write tables without partition
    no_part_table_name = tn("test_no_part_halo_write_" + str(switch_table_io).lower())
    o.delete_table(no_part_table_name, if_exists=True)
    col_desc = ",".join(f"{c} double" for c in "abcde") + ", f datetime"
    tb = o.create_table(no_part_table_name, col_desc, lifecycle=1)

    try:
        pd_data = pd.DataFrame(np.random.rand(100, 5), columns=list("abcde"))
        date_val = [
            (
                datetime.datetime.now().replace(microsecond=0)
                + datetime.timedelta(seconds=i)
            )
            for i in range(100)
        ]
        pd_data["f"] = pd.Series(date_val, dtype="datetime64[ms]").dt.tz_localize(
            options.local_timezone
        )
        with table_io.open_writer(no_part_table_name) as writer:
            writer.write(pa.Table.from_pandas(pd_data, preserve_index=False))
        with table_io.open_reader(no_part_table_name) as reader:
            pd.testing.assert_frame_equal(reader.read_all().to_pandas(), pd_data)
    finally:
        tb.drop()


@flaky(max_runs=3)
@pytest.mark.parametrize("switch_table_io", [False, True], indirect=True)
def test_table_io_with_range_reader(switch_table_io):
    config_odps_default_options()

    o = ODPS.from_environments()
    table_io = ODPSTableIO(o)

    # test read and write tables without partition
    no_part_table_name = tn("test_halo_write_range_" + str(switch_table_io).lower())
    o.delete_table(no_part_table_name, if_exists=True)
    tb = o.create_table(
        no_part_table_name, ",".join(f"{c} double" for c in "abcde"), lifecycle=1
    )

    try:
        pd_data = pd.DataFrame(np.random.rand(100, 5), columns=list("abcde"))
        with table_io.open_writer(no_part_table_name) as writer:
            writer.write(pa.Table.from_pandas(pd_data, preserve_index=False))

        with table_io.open_reader(
            no_part_table_name, start=None, stop=100, row_batch_size=10
        ) as reader:
            pd.testing.assert_frame_equal(reader.read_all().to_pandas(), pd_data)

        with table_io.open_reader(
            no_part_table_name,
            start=-2,
            stop=-52,
            reverse_range=True,
            row_batch_size=10,
        ) as reader:
            pd.testing.assert_frame_equal(
                reader.read_all().to_pandas(),
                pd_data.iloc[-51:-1].reset_index(drop=True),
            )
    finally:
        tb.drop()


@flaky(max_runs=3)
@pytest.mark.parametrize("switch_table_io", [False, True], indirect=True)
def test_table_io_with_parts(switch_table_io):
    config_odps_default_options()

    o = ODPS.from_environments()
    table_io = ODPSTableIO(o)

    # test read and write tables with partition
    parted_table_name = tn("test_parted_halo_write_" + str(switch_table_io).lower())
    o.delete_table(parted_table_name, if_exists=True)
    tb = o.create_table(
        parted_table_name,
        (",".join(f"{c} double" for c in "abcde"), "pt string"),
        lifecycle=1,
    )

    try:
        pd_data = pd.DataFrame(np.random.rand(100, 5), columns=list("abcde"))
        with table_io.open_writer(parted_table_name, "pt=test") as writer:
            writer.write(pa.Table.from_pandas(pd_data, preserve_index=False))
        with table_io.open_reader(parted_table_name, "pt=test") as reader:
            pd.testing.assert_frame_equal(reader.read_all().to_pandas(), pd_data)
        with table_io.open_reader(
            parted_table_name, "pt=test", partition_columns=True
        ) as reader:
            expected_data = pd_data.copy()
            expected_data["pt"] = "test"
            pd.testing.assert_frame_equal(reader.read_all().to_pandas(), expected_data)
    finally:
        tb.drop()


def test_tunnel_table_io_with_modified():
    config_odps_default_options()

    o = ODPS.from_environments()
    table_io = TunnelTableIO(o)

    # test read and write tables with partition
    parted_table_name = tn("test_tunnel_write_modified")
    o.delete_table(parted_table_name, if_exists=True)
    tb = o.create_table(
        parted_table_name,
        (",".join(f"{c} double" for c in "abcde"), "pt string"),
        lifecycle=1,
    )

    raised = False
    raw_open_reader = Table.open_reader

    def _new_open_reader(self, *args, **kwargs):
        nonlocal raised
        if not raised:
            raised = True
            raise TableModified("Intentional error")
        return raw_open_reader(self, *args, **kwargs)

    try:
        pd_data = pd.DataFrame(np.random.rand(100, 5), columns=list("abcde"))
        with table_io.open_writer(parted_table_name, "pt=test") as writer:
            writer.write(pa.Table.from_pandas(pd_data, preserve_index=False))

        with mock.patch(
            "odps.models.table.Table.open_reader", new=_new_open_reader
        ), table_io.open_reader(parted_table_name, "pt=test") as reader:
            pd.testing.assert_frame_equal(reader.read_all().to_pandas(), pd_data)
    finally:
        tb.drop()
