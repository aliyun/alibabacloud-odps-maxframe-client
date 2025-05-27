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
import time

import pytest

from ..lib import wrapped_pickle
from ..protocol import (
    DagInfo,
    ErrorInfo,
    ExecutionStatus,
    ODPSTableResultInfo,
    ODPSVolumeResultInfo,
    ResultInfo,
    SessionInfo,
)
from ..serialization import RemoteException
from ..utils import deserialize_serializable, serialize_serializable


def _json_round_trip(json_data: dict) -> dict:
    return json.loads(json.dumps(json_data))


def test_result_info_json_serialize():
    ri = ResultInfo.from_json(_json_round_trip(ResultInfo().to_json()))
    assert type(ri) is ResultInfo

    ri = ODPSTableResultInfo(
        full_table_name="table_name", partition_specs=["pt=partition"]
    )
    deserial_ri = ResultInfo.from_json(_json_round_trip(ri.to_json()))
    assert type(ri) is ODPSTableResultInfo
    assert ri.result_type == deserial_ri.result_type
    assert ri.full_table_name == deserial_ri.full_table_name
    assert ri.partition_specs == deserial_ri.partition_specs

    ri = ODPSTableResultInfo(full_table_name="table_name")
    deserial_ri = ResultInfo.from_json(_json_round_trip(ri.to_json()))
    assert type(ri) is ODPSTableResultInfo
    assert ri.result_type == deserial_ri.result_type
    assert ri.full_table_name == deserial_ri.full_table_name
    assert ri.partition_specs == deserial_ri.partition_specs

    ri = ODPSVolumeResultInfo(volume_name="vol_name", volume_path="vol_path")
    deserial_ri = ResultInfo.from_json(_json_round_trip(ri.to_json()))
    assert type(ri) is ODPSVolumeResultInfo
    assert ri.result_type == deserial_ri.result_type
    assert ri.volume_name == deserial_ri.volume_name
    assert ri.volume_path == deserial_ri.volume_path


def test_error_info_json_serialize():
    try:
        raise ValueError("ERR_DATA")
    except ValueError as ex:
        err_info = ErrorInfo.from_exception(ex)

    deserial_err_info = ErrorInfo.from_json(_json_round_trip(err_info.to_json()))
    assert deserial_err_info.error_messages == err_info.error_messages
    assert isinstance(deserial_err_info.raw_error_data, ValueError)

    with wrapped_pickle.switch_unpickle():
        mf_err_info = deserialize_serializable(serialize_serializable(err_info))
    assert isinstance(mf_err_info.raw_error_data, RemoteException)
    with pytest.raises(RemoteException):
        mf_err_info.reraise()

    deserial_err_info = ErrorInfo.from_json(_json_round_trip(mf_err_info.to_json()))
    assert isinstance(deserial_err_info.raw_error_data, ValueError)
    with pytest.raises(ValueError):
        deserial_err_info.reraise()


class CannotPickleException(Exception):
    def __reduce__(self):
        raise ValueError


class CannotUnpickleException(Exception):
    @classmethod
    def load_from_pk(cls, _):
        raise ValueError

    def __reduce__(self):
        return type(self).load_from_pk, (0,)


def test_error_info_fallback_json_serialize():
    try:
        raise CannotPickleException
    except CannotPickleException as ex:
        err_info1 = ErrorInfo.from_exception(ex)

    try:
        raise CannotUnpickleException
    except CannotUnpickleException as ex:
        err_info2 = ErrorInfo.from_exception(ex)

    for err_info in (err_info1, err_info2):
        deserial_err_info = ErrorInfo.from_json(err_info.to_json())
        assert deserial_err_info.raw_error_source is None
        assert deserial_err_info.raw_error_data is None

        with pytest.raises(RemoteException):
            deserial_err_info.reraise()


def test_dag_info_json_serialize():
    try:
        raise ValueError("ERR_DATA")
    except ValueError as ex:
        err_info = ErrorInfo.from_exception(ex)
    info = DagInfo(
        session_id="test_session_id",
        dag_id="test_dag_id",
        status=ExecutionStatus.FAILED,
        progress=0.65,
        tileable_to_result_infos={
            "tileable_key": ODPSTableResultInfo(full_table_name="table_name")
        },
        error_info=err_info,
    )
    json_info = info.to_json()
    json_info["non_existing_field"] = "non_existing"
    deserial_info = DagInfo.from_json(_json_round_trip(json_info))
    assert deserial_info.session_id == info.session_id
    assert deserial_info.dag_id == info.dag_id
    assert deserial_info.status == info.status
    assert deserial_info.progress == info.progress
    assert (
        deserial_info.tileable_to_result_infos["tileable_key"].full_table_name
        == info.tileable_to_result_infos["tileable_key"].full_table_name
    )
    with pytest.raises(ValueError):
        deserial_info.error_info.reraise()


def test_session_info_json_serialize():
    dag_info = DagInfo(
        session_id="test_session_id",
        dag_id="test_dag_id",
        status=ExecutionStatus.RUNNING,
        progress=0.65,
    )
    info = SessionInfo(
        session_id="test_session_id",
        settings={"sql.settings": {}},
        start_timestamp=time.time(),
        idle_timestamp=None,
        dag_infos={"test_dag_id": dag_info},
    )
    deserial_info = SessionInfo.from_json(_json_round_trip(info.to_json()))
    assert deserial_info.session_id == info.session_id
    assert deserial_info.settings == info.settings
    assert deserial_info.start_timestamp == info.start_timestamp
    assert deserial_info.idle_timestamp == info.idle_timestamp
    assert (
        deserial_info.dag_infos["test_dag_id"].status
        == info.dag_infos["test_dag_id"].status
    )
    assert (
        deserial_info.dag_infos["test_dag_id"].progress
        == info.dag_infos["test_dag_id"].progress
    )
