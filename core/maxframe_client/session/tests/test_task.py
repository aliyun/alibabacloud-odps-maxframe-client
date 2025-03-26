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

import json
import os

import mock
import pytest
from defusedxml import ElementTree
from odps import ODPS
from odps import options as odps_options

from maxframe import options
from maxframe.config import option_context

from ...session.consts import MAXFRAME_OUTPUT_JSON_FORMAT
from ...session.task import MaxFrameInstanceCaller, MaxFrameTask, MaxFrameTaskSession

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
