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

import contextlib
import faulthandler
import os
from configparser import ConfigParser, NoOptionError, NoSectionError

import pytest
from odps import ODPS
from odps.accounts import BearerTokenAccount

from .config import options

faulthandler.enable(all_threads=True)
_test_conf_file_name = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tests", "test.conf"
)


@pytest.fixture(scope="session")
def test_config():
    config = ConfigParser()
    config.read(_test_conf_file_name)
    return config


def _get_account_env(test_config: ConfigParser, section_name: str) -> ODPS:
    try:
        access_id = test_config.get(section_name, "access_id")
    except NoOptionError:
        access_id = test_config.get("odps", "access_id")
    if not access_id:
        access_id = os.getenv("ACCESS_ID")
    try:
        secret_access_key = test_config.get(section_name, "secret_access_key")
    except NoOptionError:
        secret_access_key = test_config.get("odps", "secret_access_key")
    if not secret_access_key:
        secret_access_key = os.getenv("SECRET_ACCESS_KEY")
    try:
        project = test_config.get(section_name, "project")
    except NoOptionError:
        project = test_config.get("odps", "project")
    try:
        endpoint = test_config.get(section_name, "endpoint")
    except NoOptionError:
        endpoint = test_config.get("odps", "endpoint")
    try:
        tunnel_endpoint = test_config.get("odps", "tunnel_endpoint")
    except NoOptionError:
        tunnel_endpoint = None
    try:
        namespace = test_config.get("odps", "namespace")
    except NoOptionError:
        namespace = None
    return ODPS(
        access_id,
        secret_access_key,
        project,
        endpoint,
        tunnel_endpoint=tunnel_endpoint,
        overwrite_global=False,
        namespace=namespace,
    )


def _get_bearer_token_env(test_config: ConfigParser, section_name: str) -> ODPS:
    entry = _get_account_env(test_config, section_name)
    policy = {
        "Version": "1",
        "Statement": [
            {"Action": ["odps:*"], "Resource": "acs:odps:*:*", "Effect": "Allow"}
        ],
    }
    token = entry.get_project().generate_auth_token(policy, "bearer", 5)
    return ODPS(
        account=BearerTokenAccount(token, 5),
        project=entry.project,
        endpoint=entry.endpoint,
        tunnel_endpoint=entry.tunnel_endpoint,
        namespace=entry.namespace,
    )


@contextlib.contextmanager
def _enter_odps_envs(entry, drop_temp_tables=True):
    stored_envs = {}
    for env_name in (
        "ODPS_BEARER_TOKEN",
        "ODPS_PROJECT_NAME",
        "ODPS_ENDPOINT",
        "RAY_ISOLATION_UT_ENV",
        "ODPS_TUNNEL_ENDPOINT",
        "ODPS_NAMESPACE",
    ):
        if env_name in os.environ:
            stored_envs[env_name] = os.environ[env_name]
            del os.environ[env_name]

    os.environ["ODPS_BEARER_TOKEN"] = entry.account.token
    os.environ["ODPS_PROJECT_NAME"] = entry.project
    os.environ["ODPS_ENDPOINT"] = entry.endpoint
    if entry.namespace:
        os.environ["ODPS_NAMESPACE"] = entry.namespace
    os.environ["RAY_ISOLATION_UT_ENV"] = "UT"
    if entry.tunnel_endpoint:
        os.environ["ODPS_TUNNEL_ENDPOINT"] = entry.tunnel_endpoint

    try:
        yield
    finally:
        os.environ.pop("ODPS_BEARER_TOKEN", None)
        os.environ.pop("ODPS_PROJECT_NAME", None)
        os.environ.pop("ODPS_ENDPOINT", None)
        os.environ.pop("ODPS_NAMESPACE", None)
        os.environ.pop("ODPS_TUNNEL_ENDPOINT", None)
        os.environ.pop("RAY_ISOLATION_UT_ENV", None)

        for env_name, val in stored_envs.items():
            os.environ[env_name] = val

        if drop_temp_tables:
            from .tests.utils import _test_tables_to_drop

            for table_name in _test_tables_to_drop:
                try:
                    entry.delete_table(table_name, wait=False)
                except:
                    pass


@pytest.fixture
def odps_with_schema(test_config, request):
    try:
        entry = _get_bearer_token_env(test_config, "odps_with_schema")
    except NoSectionError:
        pytest.skip("Need to specify odps_with_schema section in test.conf")
        raise

    with _enter_odps_envs(entry, drop_temp_tables=False):
        yield entry


@pytest.fixture(scope="session", autouse=True)
def odps_envs(test_config):
    entry = _get_bearer_token_env(test_config, "odps")

    with _enter_odps_envs(entry):
        yield


@pytest.fixture(scope="session")
def odps_account(test_config):
    return _get_account_env(test_config, "odps")


@pytest.fixture(scope="session")
def oss_config():
    config = ConfigParser()
    config.read(_test_conf_file_name)

    old_role_arn = options.service_role_arn
    old_cache_url = options.object_cache_url

    try:
        oss_access_id = config.get("oss", "access_id") or os.getenv("ACCESS_ID")
        oss_secret_access_key = config.get("oss", "secret_access_key") or os.getenv(
            "SECRET_ACCESS_KEY"
        )
        oss_bucket_name = config.get("oss", "bucket_name")
        oss_endpoint = config.get("oss", "endpoint")
        oss_rolearn = config.get("oss", "rolearn")

        options.service_role_arn = oss_rolearn
        if "test" in oss_endpoint:
            oss_svc_endpoint = oss_endpoint
        else:
            endpoint_parts = oss_endpoint.split(".", 1)
            if "-internal" not in endpoint_parts[0]:
                endpoint_parts[0] += "-internal"
            oss_svc_endpoint = ".".join(endpoint_parts)
        options.object_cache_url = f"oss://{oss_svc_endpoint}/{oss_bucket_name}"

        config.oss_config = (
            oss_access_id,
            oss_secret_access_key,
            oss_bucket_name,
            oss_endpoint,
        )

        import oss2

        auth = oss2.Auth(oss_access_id, oss_secret_access_key)
        config.oss_bucket = oss2.Bucket(auth, oss_endpoint, oss_bucket_name)
        config.oss_rolearn = oss_rolearn
        yield config
    except (NoSectionError, NoOptionError, ImportError):
        yield None
    finally:
        options.service_role_arn = old_role_arn
        options.object_cache_url = old_cache_url


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


@pytest.fixture
def enable_local_execution(request):
    old_enabled = options.local_execution.enabled
    old_limit = options.local_execution.size_limit
    try:
        options.local_execution.enabled = True
        options.local_execution.size_limit = getattr(request, "param", 0)
        yield
    finally:
        options.local_execution.enabled = old_enabled
        options.local_execution.size_limit = old_limit
