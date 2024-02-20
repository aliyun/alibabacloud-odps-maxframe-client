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
import pyarrow as pa
from odps import ODPS

from ...tests.utils import flaky, tn
from ...utils import config_odps_default_options
from ..tableio import HaloTableIO


@flaky(max_runs=3)
def test_empty_table_io():
    config_odps_default_options()

    o = ODPS.from_environments()
    halo_table_io = HaloTableIO(o)

    # test read from empty table
    empty_table_name = tn("test_empty_table_halo_read")
    o.delete_table(empty_table_name, if_exists=True)
    tb = o.create_table(empty_table_name, "col1 string", lifecycle=1)

    try:
        with halo_table_io.open_reader(empty_table_name) as reader:
            assert len(reader.read_all()) == 0
    finally:
        tb.drop()


@flaky(max_runs=3)
def test_table_io_without_parts():
    config_odps_default_options()

    o = ODPS.from_environments()
    halo_table_io = HaloTableIO(o)

    # test read and write tables without partition
    no_part_table_name = tn("test_no_part_halo_write")
    o.delete_table(no_part_table_name, if_exists=True)
    tb = o.create_table(
        no_part_table_name, ",".join(f"{c} double" for c in "abcde"), lifecycle=1
    )

    try:
        pd_data = pd.DataFrame(np.random.rand(100, 5), columns=list("abcde"))
        with halo_table_io.open_writer(no_part_table_name) as writer:
            writer.write(pa.Table.from_pandas(pd_data, preserve_index=False))
        with halo_table_io.open_reader(no_part_table_name) as reader:
            pd.testing.assert_frame_equal(reader.read_all().to_pandas(), pd_data)
    finally:
        tb.drop()


@flaky(max_runs=3)
def test_table_io_with_range_reader():
    config_odps_default_options()

    o = ODPS.from_environments()
    halo_table_io = HaloTableIO(o)

    # test read and write tables without partition
    no_part_table_name = tn("test_no_part_halo_write")
    o.delete_table(no_part_table_name, if_exists=True)
    tb = o.create_table(
        no_part_table_name, ",".join(f"{c} double" for c in "abcde"), lifecycle=1
    )

    try:
        pd_data = pd.DataFrame(np.random.rand(100, 5), columns=list("abcde"))
        with halo_table_io.open_writer(no_part_table_name) as writer:
            writer.write(pa.Table.from_pandas(pd_data, preserve_index=False))

        with halo_table_io.open_reader(
            no_part_table_name, start=None, stop=100, row_batch_size=10
        ) as reader:
            pd.testing.assert_frame_equal(reader.read_all().to_pandas(), pd_data)

        with halo_table_io.open_reader(
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
def test_table_io_with_parts():
    config_odps_default_options()

    o = ODPS.from_environments()
    halo_table_io = HaloTableIO(o)

    # test read and write tables with partition
    parted_table_name = tn("test_parted_halo_write")
    o.delete_table(parted_table_name, if_exists=True)
    tb = o.create_table(
        parted_table_name,
        (",".join(f"{c} double" for c in "abcde"), "pt string"),
        lifecycle=1,
    )

    try:
        pd_data = pd.DataFrame(np.random.rand(100, 5), columns=list("abcde"))
        with halo_table_io.open_writer(parted_table_name, "pt=test") as writer:
            writer.write(pa.Table.from_pandas(pd_data, preserve_index=False))
        with halo_table_io.open_reader(parted_table_name, "pt=test") as reader:
            pd.testing.assert_frame_equal(reader.read_all().to_pandas(), pd_data)
        with halo_table_io.open_reader(
            parted_table_name, "pt=test", partition_columns=True
        ) as reader:
            expected_data = pd_data.copy()
            expected_data["pt"] = "test"
            pd.testing.assert_frame_equal(reader.read_all().to_pandas(), expected_data)
    finally:
        tb.drop()
