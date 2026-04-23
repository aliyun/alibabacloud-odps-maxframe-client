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

import base64
import json
import os

import mock
import msgpack
import pytest
from defusedxml import ElementTree
from odps import ODPS
from odps import options as odps_options

from maxframe import dataframe as md
from maxframe import options
from maxframe.config import option_context
from maxframe.protocol import SessionInfo
from maxframe.tests.utils import tn

from ...session.task import MaxFrameInstanceCaller, MaxFrameTask, MaxFrameTaskSession
from ..consts import MAXFRAME_OUTPUT_JSON_FORMAT, MAXFRAME_OUTPUT_MSGPACK_FORMAT

expected_file_dir = os.path.join(os.path.dirname(__file__), "expected-data")


def test_maxframe_instance_caller_creating_session():
    o = ODPS.from_environments()

    def create_caller(**kwargs):
        kw = dict(
            odps_entry=o,
            task_name="task_test",
            major_version="test_version",
            output_format=MAXFRAME_OUTPUT_JSON_FORMAT,
            running_cluster="test_cluster",
        )
        kw.update(**kwargs)
        return MaxFrameInstanceCaller(**kw)

    def mock_create(self, task: MaxFrameTask, priority=None, running_cluster=None):
        assert priority == 100
        assert running_cluster == "test_cluster"
        root = ElementTree.parse(
            os.path.join(expected_file_dir, "create_session.xml")
        ).getroot()
        assert root.tag == "MaxFrame"
        assert root.find("Name").text == "task_test"
        assert root.find("Command").text == "CREATE_SESSION"
        property_node = root.find("Config").find("Property")
        assert property_node.find("Name").text == "settings"
        setting_dict = json.loads(property_node.find("Value").text)
        assert setting_dict["odps.task.major.version"] == "test_version"

        assert setting_dict["odps.maxframe.output_format"] == "json"
        maxframe_setting_dict = json.loads(setting_dict["odps.maxframe.settings"])
        assert maxframe_setting_dict["session.max_alive_seconds"] == 259200

    with mock.patch.multiple(
        target="maxframe_client.session.task.MaxFrameInstanceCaller",
        _wait_instance_task_ready=mock.DEFAULT,
        get_session=mock.DEFAULT,
    ), mock.patch("odps.models.instances.BaseInstances.create", mock_create):
        task_caller = create_caller(priority=100)
        task_caller.create_session()

        old_priority = odps_options.priority
        old_get_priority = odps_options.get_priority
        try:
            task_caller = create_caller(priority=100)
            odps_options.priority = 100
            task_caller.create_session()

            odps_options.priority = None
            odps_options.get_priority = lambda _: 100
            task_caller.create_session()
        finally:
            odps_options.priority = old_priority
            odps_options.get_priority = old_get_priority


@pytest.mark.asyncio
async def test_session_quota_flag_valid():
    def mock_create(self, task: MaxFrameTask, **kwargs):
        assert task.properties["settings"]
        task_settings = json.loads(task.properties["settings"])
        assert task_settings["odps.task.wlm.quota"] == "session_quota"

    with mock.patch.multiple(
        target="maxframe_client.session.task.MaxFrameInstanceCaller",
        _wait_instance_task_ready=mock.DEFAULT,
        get_session=mock.DEFAULT,
        get_logview_address=mock.DEFAULT,
    ), mock.patch("odps.models.instances.BaseInstances.create", mock_create):
        with option_context({"session.quota_name": "session_quota"}):
            with pytest.raises(ValueError):
                options.sql.settings["odps.task.wlm.quota"] = "session_quota2"
                await MaxFrameTaskSession.init(
                    address="test", odps_entry=ODPS.from_environments()
                )
            options.sql.settings["odps.task.wlm.quota"] = "session_quota"
            mf_task_session = await MaxFrameTaskSession.init(
                address="test", odps_entry=ODPS.from_environments()
            )
            with pytest.raises(ValueError):
                options.sql.settings["odps.task.wlm.quota"] = "session_quota2"
                mf_task_session._get_diff_settings()
            options.sql.settings["odps.task.wlm.quota"] = "session_quota"
            mf_task_session._get_diff_settings()


def test_collect_explain_instances():
    odps_entry = ODPS.from_environments()
    test_table = tn("test_collect_explain_table")
    odps_entry.delete_table(test_table, if_exists=True)
    tb = odps_entry.create_table(test_table, "col string", lifecycle=1)

    t = md.read_odps_query(f"select * from {test_table}")
    settings = MaxFrameInstanceCaller._collect_explain_instances(t.build_graph())
    inst_id = settings["dag.settings"]["client.explain_instances"]
    inst = odps_entry.get_instance(inst_id)
    assert "explain" in inst.get_sql_query().lower()
    tb.drop()


def test_deserial_task_info_result():
    # Create a mock caller instance
    caller = MaxFrameInstanceCaller.__new__(MaxFrameInstanceCaller)

    # Test case 1: When result is None and session is not deleted
    caller._deleted = False
    caller._instance = mock.MagicMock()
    caller._instance.id = "test_instance_id"

    with pytest.raises(
        Exception
    ):  # SessionAlreadyClosedError is wrapped in parse_instance_error
        caller._deserial_task_info_result({}, mock.MagicMock())

    # Test case 2: When result is None and session is deleted
    caller._deleted = True
    result = caller._deserial_task_info_result({}, mock.MagicMock())
    assert result is None

    # Test case 3: With valid JSON format result
    caller._output_format = MAXFRAME_OUTPUT_JSON_FORMAT
    session_info = SessionInfo(session_id="test_session_id")
    json_data = session_info.to_json()
    encoded_result = base64.b64encode(json.dumps(json_data).encode()).decode()

    # Mock the from_json method to return our test object
    with mock.patch.object(SessionInfo, "from_json", return_value=session_info):
        result = caller._deserial_task_info_result(
            {"result": encoded_result}, SessionInfo
        )
        assert isinstance(result, SessionInfo)
        assert result.session_id == "test_session_id"

    # Test case 4: With valid msgpack format result
    caller._output_format = MAXFRAME_OUTPUT_MSGPACK_FORMAT
    session_info = SessionInfo(session_id="test_session_id_msgpack")
    json_data = session_info.to_json()
    packed_data = msgpack.dumps(json_data)
    encoded_result = base64.b64encode(packed_data).decode()

    with mock.patch.object(SessionInfo, "from_json", return_value=session_info):
        result = caller._deserial_task_info_result(
            {"result": encoded_result}, SessionInfo
        )
        assert isinstance(result, SessionInfo)
        assert result.session_id == "test_session_id_msgpack"

    # Test case 5: Unsupported format
    caller._output_format = "unsupported_format"
    session_info = SessionInfo(session_id="test_session_id")
    json_data = session_info.to_json()
    encoded_result = base64.b64encode(json.dumps(json_data).encode()).decode()

    # Should raise some kind of exception due to the except clause wrapping everything
    with pytest.raises(Exception):
        caller._deserial_task_info_result({"result": encoded_result}, SessionInfo)
