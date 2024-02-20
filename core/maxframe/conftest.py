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

import faulthandler
import os
from configparser import ConfigParser, NoOptionError

import pytest
from odps import ODPS

faulthandler.enable(all_threads=True)
_test_conf_file_name = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tests", "test.conf"
)


@pytest.fixture(scope="session")
def test_config():
    config = ConfigParser()
    config.read(_test_conf_file_name)
    return config


@pytest.fixture(scope="session", autouse=True)
def odps_envs(test_config):
    access_id = test_config.get("odps", "access_id")
    secret_access_key = test_config.get("odps", "secret_access_key")
    project = test_config.get("odps", "project")
    endpoint = test_config.get("odps", "endpoint")
    try:
        tunnel_endpoint = test_config.get("odps", "tunnel_endpoint")
    except NoOptionError:
        tunnel_endpoint = None

    entry = ODPS(
        access_id, secret_access_key, project, endpoint, overwrite_global=False
    )
    policy = {
        "Version": "1",
        "Statement": [
            {"Action": ["odps:*"], "Resource": "acs:odps:*:*", "Effect": "Allow"}
        ],
    }
    token = entry.get_project().generate_auth_token(policy, "bearer", 5)

    os.environ["ODPS_BEARER_TOKEN"] = token
    os.environ["ODPS_PROJECT_NAME"] = project
    os.environ["ODPS_ENDPOINT"] = endpoint
    if tunnel_endpoint:
        os.environ["ODPS_TUNNEL_ENDPOINT"] = tunnel_endpoint

    try:
        yield
    finally:
        os.environ.pop("ODPS_BEARER_TOKEN", None)
        os.environ.pop("ODPS_PROJECT_NAME", None)
        os.environ.pop("ODPS_ENDPOINT", None)
        os.environ.pop("ODPS_TUNNEL_ENDPOINT", None)

        from .tests.utils import _test_tables_to_drop

        for table_name in _test_tables_to_drop:
            try:
                entry.delete_table(table_name, wait=False)
            except:
                pass


@pytest.fixture
def oss_config():
    config = ConfigParser()
    config.read(_test_conf_file_name)

    try:
        oss_access_id = config.get("oss", "access_id")
        oss_secret_access_key = config.get("oss", "secret_access_key")
        oss_bucket_name = config.get("oss", "bucket_name")
        oss_endpoint = config.get("oss", "endpoint")

        config.oss_config = (
            oss_access_id,
            oss_secret_access_key,
            oss_bucket_name,
            oss_endpoint,
        )

        import oss2

        auth = oss2.Auth(oss_access_id, oss_secret_access_key)
        config.oss_bucket = oss2.Bucket(auth, oss_endpoint, oss_bucket_name)
        return config
    except (ConfigParser.NoSectionError, ConfigParser.NoOptionError, ImportError):
        return None


@pytest.fixture(autouse=True)
def apply_engine_selection(request):
    try:
        from maxframe_framedriver.services.analyzer import DagAnalyzer
    except ImportError:
        DagAnalyzer = None
    try:
        marks = list(request.node.iter_markers())
        if request.node.parent:
            marks.extend(request.node.parent.iter_markers())
        marks = [m for m in marks if m.name == "maxframe_engine"]

        if DagAnalyzer:
            engines = set()
            for mark in marks:
                engines.update(mark.args[0])
            DagAnalyzer._enabled_engines = set(engines)
        yield
    finally:
        if DagAnalyzer:
            DagAnalyzer._enabled_engines = set()


@pytest.fixture
def local_test_envs():
    spe_launcher_env = "MAXFRAME_SPE_LAUNCHER"
    old_value = os.getenv(spe_launcher_env)
    os.environ[spe_launcher_env] = "local"
    yield
    if old_value is not None:
        os.environ[spe_launcher_env] = old_value
    else:
        del os.environ[spe_launcher_env]
