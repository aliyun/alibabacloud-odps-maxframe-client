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

import time
from typing import Dict

import mock
import numpy as np
import pandas as pd
import pytest
from odps import ODPS

import maxframe.dataframe as md
import maxframe.remote as mr
from maxframe.core import ExecutableTuple, TileableGraph
from maxframe.protocol import ResultInfo
from maxframe.serialization import RemoteException
from maxframe.session import new_session
from maxframe.utils import build_temp_table_name
from maxframe_framedriver.app.tests.test_framedriver_webapp import (  # noqa: F401
    framedriver_app,
)

from ..clients.framedriver import FrameDriverClient

pytestmark = pytest.mark.maxframe_engine(["MCSQL", "SPE"])


@pytest.fixture
def start_mock_session(framedriver_app):  # noqa: F811
    odps_entry = ODPS.from_environments()
    framedriver_addr = f"mf://localhost:{framedriver_app.port}"
    session = new_session(framedriver_addr, odps_entry=odps_entry)

    session_id = session._isolated_session.session_id
    try:
        yield session_id
    finally:
        if hasattr(session, "destroy"):
            session.destroy()
            time.sleep(5)  # Wait for temp table deleted
        else:
            session.reset_default()


def test_simple_run_dataframe(start_mock_session):
    odps_entry = ODPS.from_environments()

    pd_df = pd.DataFrame(np.random.rand(1000, 5), columns=list("ABCDE"))
    df = md.DataFrame(pd_df)
    df["F"] = df["A"] + df["B"]
    df["G"] = df["C"] * 2 + df["D"]

    intermediate_key = df.key
    result = df.execute().fetch()
    assert len(result) == 1000
    assert len(result.columns) == 7

    pd_result = pd_df.copy()
    pd_result["F"] = pd_result["A"] + pd_result["B"]
    pd_result["G"] = pd_result["C"] * 2 + pd_result["D"]
    pd.testing.assert_frame_equal(pd_result, result)

    original_submit_dag = FrameDriverClient.submit_dag

    async def patched_submit_dag(
        self,
        session_id: str,
        dag: TileableGraph,
        managed_input_infos: Dict[str, ResultInfo] = None,
    ):
        assert len(dag) == 2
        return await original_submit_dag(self, session_id, dag, managed_input_infos)

    df["H"] = "extra_content"

    with mock.patch(
        "maxframe_client.clients.framedriver.FrameDriverClient.submit_dag",
        new=patched_submit_dag,
    ):
        result = df.execute().fetch()
        assert len(result) == 1000
        assert len(result.columns) == 8

    corner_top, corner_bottom = ExecutableTuple([df.iloc[:10], df.iloc[-10:]]).fetch()
    assert len(corner_top) == len(corner_bottom) == 10

    # check ellipsis mark in DataFrame errors
    df_str_repr = str(df)
    assert ".." in df_str_repr

    key = df.key
    assert odps_entry.exist_table(
        build_temp_table_name(start_mock_session, intermediate_key)
    )
    assert odps_entry.exist_table(build_temp_table_name(start_mock_session, key))
    del df
    time.sleep(5)
    assert not odps_entry.exist_table(
        build_temp_table_name(start_mock_session, intermediate_key)
    )
    assert not odps_entry.exist_table(build_temp_table_name(start_mock_session, key))


def test_run_dataframe_with_pd_source(start_mock_session):
    odps_entry = ODPS.from_environments()

    pd_df = pd.DataFrame(np.random.rand(1000, 5), columns=list("ABCDE"))
    df = md.DataFrame(pd_df)
    src_key = df.key
    df["F"] = df["A"] + df["B"]
    df["G"] = df["C"] * 2 + df["D"]

    try:
        result = df.execute().fetch()
        assert len(result) == 1000
        assert len(result.columns) == 7

        del df
        time.sleep(5)
        assert not odps_entry.exist_table(
            build_temp_table_name(start_mock_session, src_key)
        )
    finally:
        odps_entry.delete_table(
            build_temp_table_name(start_mock_session, src_key), if_exists=True
        )


def test_run_dataframe_from_to_odps_table(start_mock_session):
    odps_entry = ODPS.from_environments()

    pd_df = pd.DataFrame(np.random.rand(10, 5), columns=list("ABCDE"))
    table_name = build_temp_table_name(start_mock_session, "tmp_save")
    table_obj = odps_entry.get_table(table_name)
    try:
        md.to_odps_table(md.DataFrame(pd_df), table_obj).execute().fetch()
        with table_obj.open_reader() as reader:
            result_df = reader.to_pandas()
        assert len(result_df) == 10
        assert len(result_df.columns) == 6

        df = md.read_odps_table(table_obj, index_col="index").head(10).execute().fetch()
        assert len(df) == 10
        assert len(df.columns) == 5
    finally:
        odps_entry.delete_table(table_name, if_exists=True)


def test_run_and_fetch_series(start_mock_session):
    odps_entry = ODPS.from_environments()

    pd_s1 = pd.Series(np.random.rand(1000))
    s1 = md.Series(pd_s1)
    src_key = s1.key
    s1 = s1 * 10 + 5

    try:
        result = s1.execute().fetch()
        assert len(result) == 1000

        pd_result = pd_s1 * 10 + 5
        pd.testing.assert_series_equal(pd_result, result)

        del s1
        time.sleep(5)
        assert not odps_entry.exist_table(
            build_temp_table_name(start_mock_session, src_key)
        )
    finally:
        odps_entry.delete_table(
            build_temp_table_name(start_mock_session, src_key), if_exists=True
        )


def test_run_remote_success(start_mock_session):
    def func(a, b):
        return a + b

    v = mr.spawn(func, args=(1, 5))
    v2 = mr.spawn(func, args=(v, 15))

    result = v2.execute().fetch()
    assert result == 21


def test_run_remote_error(start_mock_session):
    def func():
        raise ValueError

    v = mr.spawn(func)

    with pytest.raises((ValueError, RemoteException)):
        v.execute()
