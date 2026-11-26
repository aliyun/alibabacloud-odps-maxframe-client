# Copyright 1999-2026 Alibaba Group Holding Ltd.
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
import pickle
import time
from typing import Any

import pandas as pd
import pytest

from maxframe.io.odpsio.arrow import pandas_to_arrow
from maxframe.lib import wrapped_pickle
from maxframe.protocol import (
    DagInfo,
    DisplayMessage,
    ErrorInfo,
    ErrorSource,
    ExecutionStatus,
    ODPSTableResultInfo,
    ODPSVolumeResultInfo,
    ResultInfo,
    SessionInfo,
    SubDagInfo,
)
from maxframe.serialization import RemoteException, pickle_buffers
from maxframe.utils import deserialize_serializable, serialize_serializable


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


class GeneratorHoldingException(Exception):
    def __init__(self):
        super().__init__("ERR_DATA")
        self.data_generator = (value for value in range(1))


def test_error_info_from_exception_drops_unpickleable_raw_error():
    try:
        raise GeneratorHoldingException()
    except GeneratorHoldingException as ex:
        err_info = ErrorInfo.from_exception(ex)

    assert err_info.raw_error_data is None
    assert err_info.raw_error_source is None
    assert err_info.error_messages == [
        f"{GeneratorHoldingException.__name__}: ERR_DATA"
    ]
    assert err_info.error_tracebacks[0]

    restored = pickle.loads(pickle.dumps(err_info))
    assert restored.raw_error_data is None
    assert restored.raw_error_source is None
    with pytest.raises(RemoteException, match="ERR_DATA"):
        restored.reraise()


_FAILURES = [RuntimeError, KeyboardInterrupt, SystemExit]


def _failing_reduce(raised):
    class FailingReduce(Exception):
        def __reduce__(self):
            raise raised("failing __reduce__")

    return FailingReduce


@pytest.mark.parametrize("raised", _FAILURES)
def test_error_info_from_exception_degrades_or_propagates(raised):
    exc_type = _failing_reduce(raised)

    if issubclass(raised, Exception):
        err_info = ErrorInfo.from_exception(exc_type("payload"))
        assert err_info.raw_error_source is None
        assert err_info.raw_error_data is None
        assert err_info.error_messages == ["FailingReduce: payload"]
        with pytest.raises(RemoteException, match="payload"):
            err_info.reraise()
    else:
        # a system error raised while probing must propagate, never be swallowed
        with pytest.raises(raised):
            ErrorInfo.from_exception(exc_type("payload"))


@pytest.mark.parametrize("raised", _FAILURES)
def test_error_info_to_json_degrades_or_propagates(raised):
    exc_type = _failing_reduce(raised)
    err_info = ErrorInfo(["payload"], [[]], ErrorSource.PYTHON, exc_type("payload"))

    if issubclass(raised, Exception):
        payload = err_info.to_json()
        assert payload["raw_error_source"] is None
        assert "raw_error_data" not in payload
    else:
        with pytest.raises(raised):
            err_info.to_json()


def test_error_info_transports_ordinary_raw_error():
    # the branch handling ordinary exceptions is the one that pickles the
    # payload: without it the client would only see a RemoteException
    try:
        raise ValueError("ERR_DATA")
    except ValueError as ex:
        err_info = ErrorInfo.from_exception(ex)

    payload = err_info.to_json()
    assert payload["raw_error_source"] == ErrorSource.PYTHON.value
    assert "raw_error_data" in payload

    restored = ErrorInfo.from_json(_json_round_trip(payload))
    assert isinstance(restored.raw_error_data, ValueError)
    with pytest.raises(ValueError):
        restored.reraise()


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


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_error_info_drops_control_flow_signals(exc_type):
    err_info = ErrorInfo.from_exception(exc_type())

    # signals are never recorded as raw error data, so the transported
    # error_info cannot resurrect them on the receiving side
    assert err_info.raw_error_source is None
    assert err_info.raw_error_data is None
    assert err_info.error_messages == [f"{exc_type.__name__}: "]
    with pytest.raises(RemoteException):
        err_info.reraise()

    # a record written before signals were filtered still carries the pickled
    # signal: serialization and deserialization must both drop it
    legacy = ErrorInfo(
        error_messages=[str(exc_type())],
        error_tracebacks=[],
        raw_error_source=ErrorSource.PYTHON,
        raw_error_data=exc_type(),
    )
    legacy_json = legacy.to_json()
    assert legacy_json["raw_error_source"] is None
    assert "raw_error_data" not in legacy_json

    # an in-memory record carrying the signal must not raise it either
    with pytest.raises(RemoteException):
        legacy.reraise()

    legacy_json["raw_error_source"] = ErrorSource.PYTHON.value
    legacy_json["raw_error_data"] = [
        base64.b64encode(buf).decode() for buf in pickle_buffers(exc_type())
    ]
    restored = ErrorInfo.from_json(_json_round_trip(legacy_json))
    assert restored.raw_error_source is None
    assert restored.raw_error_data is None
    with pytest.raises(RemoteException):
        restored.reraise()


def test_failure_info_roundtrip():
    # ErrorInfo transport must strip heavy raw bytes.
    arrow_data, table_meta = pandas_to_arrow(pd.DataFrame({"a": [1]}))
    fi = {
        "substep_id": "abc",
        "operator_name": "Apply",
        "table": "proj.tmp_tbl",
        "arrow_data": serialize_serializable(arrow_data),
        "table_meta": table_meta.to_json(),
    }

    # ExceptionSerializer roundtrip (container -> host)
    exc: Any = ValueError("boom")
    exc._failure_info = fi
    blob = serialize_serializable(exc)

    restored = deserialize_serializable(blob)
    assert getattr(restored, "_failure_info") == fi

    # Serialize normally; deserialize while unpickle is forbidden (e.g. FrameDriver).
    # Exception body becomes RemoteException but _failure_info must still attach.
    value = serialize_serializable(exc)
    with wrapped_pickle.switch_unpickle():
        restored_fd = deserialize_serializable(value)
    assert isinstance(restored_fd, RemoteException)
    assert getattr(restored_fd, "_failure_info") == fi

    # Both directions under forbidden unpickle (still preserves failure_info on wire).
    with wrapped_pickle.switch_unpickle():
        blob2 = serialize_serializable(exc)
        restored2 = deserialize_serializable(blob2)
    assert getattr(restored2, "_failure_info") == fi

    # ErrorInfo JSON roundtrip (host -> client)
    err_info = ErrorInfo.from_exception(exc)
    assert err_info.failure_info == {
        "table": "proj.tmp_tbl",
        "table_meta": table_meta.to_json(),
        "substep_id": "abc",
        "operator_name": "Apply",
    }

    json_restored = ErrorInfo.from_json(_json_round_trip(err_info.to_json()))
    assert json_restored.failure_info == {
        "table": "proj.tmp_tbl",
        "table_meta": table_meta.to_json(),
        "substep_id": "abc",
        "operator_name": "Apply",
    }

    with pytest.raises(ValueError) as exc_info:
        json_restored.reraise()
    reraised: Any = exc_info.value
    assert getattr(reraised, "_failure_info") == {
        "table": "proj.tmp_tbl",
        "table_meta": table_meta.to_json(),
        "substep_id": "abc",
        "operator_name": "Apply",
    }


@pytest.mark.parametrize("failed_cu_usage", [6.0, 0.0, None])
@pytest.mark.parametrize("cancelled_cu_usage", [8.0, None])
def test_dag_info_json_serialize(failed_cu_usage, cancelled_cu_usage):
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
        display_messages=[DisplayMessage(message="warning_message", level="warning")],
        subdag_infos={
            "subdag": SubDagInfo(
                subdag_id="subdag",
                status=ExecutionStatus.FAILED,
                failed_cu_usage=failed_cu_usage,
                cancelled_cu_usage=cancelled_cu_usage,
            )
        },
    )
    json_info = info.to_json()
    json_info["non_existing_field"] = "non_existing"
    deserial_info = DagInfo.from_json(_json_round_trip(json_info))
    assert deserial_info.subdag_infos["subdag"].failed_cu_usage == failed_cu_usage
    assert deserial_info.subdag_infos["subdag"].cancelled_cu_usage == cancelled_cu_usage
    binary_info = deserialize_serializable(serialize_serializable(info))
    assert binary_info.subdag_infos["subdag"].failed_cu_usage == failed_cu_usage
    assert binary_info.subdag_infos["subdag"].cancelled_cu_usage == cancelled_cu_usage
    assert ("failed_cu_usage" in json_info["subdag_infos"]["subdag"]) == (
        failed_cu_usage is not None
    )
    assert ("cancelled_cu_usage" in json_info["subdag_infos"]["subdag"]) == (
        cancelled_cu_usage is not None
    )
    assert deserial_info.session_id == info.session_id
    assert deserial_info.dag_id == info.dag_id
    assert deserial_info.status == info.status
    assert deserial_info.progress == info.progress
    assert (
        deserial_info.tileable_to_result_infos["tileable_key"].full_table_name
        == info.tileable_to_result_infos["tileable_key"].full_table_name
    )
    assert deserial_info.display_messages == info.display_messages
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
        display_messages=[DisplayMessage(message="warning_message", level="warning")],
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
    assert deserial_info.display_messages == info.display_messages


def test_displayed_info_json_serialize():
    info = DisplayMessage(message="warning_message", level="warning")
    deserial_info = DisplayMessage.from_json(_json_round_trip(info.to_json()))
    assert deserial_info == info
