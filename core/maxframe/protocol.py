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

import base64
import enum
import uuid
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import pandas as pd

from .core import OutputType, TileableGraph
from .core.graph.entity import SerializableGraph
from .lib import wrapped_pickle as pickle
from .lib.tblib import pickling_support
from .serialization import PickleContainer, RemoteException, pickle_buffers
from .serialization.serializables import (
    AnyField,
    BoolField,
    BytesField,
    DictField,
    EnumField,
    FieldTypes,
    Float64Field,
    ListField,
    ReferenceField,
    Serializable,
    SeriesField,
    StringField,
    TupleField,
)

pickling_support.install()

BodyType = TypeVar("BodyType", bound="Serializable")


class JsonSerializable(Serializable):
    @classmethod
    def from_json(cls, serialized: dict) -> "JsonSerializable":
        raise NotImplementedError

    def to_json(self) -> dict:
        raise NotImplementedError


class ProtocolBody(Generic[BodyType], Serializable):
    request_id: bytes = BytesField(
        "request_id", default_factory=lambda: uuid.uuid4().bytes
    )
    body: BodyType = AnyField("body", default=None)


class DagStatus(enum.Enum):
    PREPARING = 0
    RUNNING = 1
    SUCCEEDED = 2
    FAILED = 3
    CANCELLING = 4
    CANCELLED = 5


class DimensionIndex(Serializable):
    is_slice: bool = BoolField("is_slice", default=None)
    is_int_index: bool = BoolField("is_int_index", default=None)
    data: List = ListField("data", default=None)


class ResultType(enum.Enum):
    NULL = 0
    ODPS_TABLE = 1
    ODPS_VOLUME = 2


class DataSerializeType(enum.Enum):
    PICKLE = 0


class VolumeDataMeta(Serializable):
    output_type: OutputType = EnumField(
        "output_type", OutputType, FieldTypes.int8, default=None
    )
    serial_type: DataSerializeType = EnumField(
        "serial_type", DataSerializeType, FieldTypes.int8, default=None
    )
    shape: Tuple[int, ...] = TupleField("shape", FieldTypes.int64, default=None)
    nsplits: Tuple[Tuple[int, ...], ...] = TupleField(
        "nsplits", FieldTypes.tuple(FieldTypes.tuple(FieldTypes.int64)), default=None
    )


_result_type_to_info_cls: Dict[ResultType, Type["ResultInfo"]] = dict()


class ResultInfo(JsonSerializable):
    _result_type = ResultType.NULL

    result_type: Optional[ResultType] = EnumField(
        "result_type", ResultType, default=ResultType.NULL
    )
    slices: Optional[List[DimensionIndex]] = ListField(
        "slices", FieldTypes.reference, default=None
    )

    @classmethod
    def _cls_from_result_type(cls, result_type: ResultType):
        if not _result_type_to_info_cls:
            for dest_cls in globals().values():
                if isinstance(dest_cls, type) and issubclass(dest_cls, ResultInfo):
                    _result_type_to_info_cls[dest_cls._result_type] = dest_cls
        return _result_type_to_info_cls[result_type]

    @classmethod
    def _json_to_kwargs(cls, serialized: dict) -> dict:
        # todo retrieve slices from json once implemented
        kw = serialized.copy()
        kw["result_type"] = ResultType(kw["result_type"])
        return kw

    @classmethod
    def from_json(cls, serialized: dict) -> "ResultInfo":
        res_type = ResultType(serialized["result_type"])
        res_cls = cls._cls_from_result_type(res_type)
        return res_cls(**res_cls._json_to_kwargs(serialized))

    def to_json(self) -> dict:
        # todo convert slices to json once implemented
        return {"result_type": self.result_type.value if self.result_type else None}


ResultInfoType = TypeVar("ResultInfoType", bound=ResultInfo)


class ODPSTableResultInfo(ResultInfo):
    _result_type = ResultType.ODPS_TABLE

    full_table_name: str = StringField("full_table_name", default=None)
    partition_specs: Optional[List[str]] = ListField(
        "partition_specs", FieldTypes.string, default=None
    )

    def __init__(self, result_type: ResultType = None, **kw):
        result_type = result_type or ResultType.ODPS_TABLE
        super().__init__(result_type=result_type, **kw)

    def to_json(self) -> dict:
        ret = super().to_json()
        ret["full_table_name"] = self.full_table_name
        if self.partition_specs:
            ret["partition_specs"] = self.partition_specs
        return ret


class ODPSVolumeResultInfo(ResultInfo):
    _result_type = ResultType.ODPS_VOLUME

    volume_name: str = StringField("volume_name", default=None)
    volume_path: str = StringField("volume_path", default=None)

    def __init__(self, result_type: ResultType = None, **kw):
        result_type = result_type or ResultType.ODPS_VOLUME
        super().__init__(result_type=result_type, **kw)

    def to_json(self) -> dict:
        ret = super().to_json()
        ret["volume_name"] = self.volume_name
        ret["volume_path"] = self.volume_path
        return ret


class ErrorSource(enum.Enum):
    PYTHON = 0


class ErrorInfo(JsonSerializable):
    error_messages: Optional[List[str]] = ListField("error_messages", FieldTypes.string)
    error_tracebacks: Optional[List[List[str]]] = ListField(
        "error_tracebacks", FieldTypes.list
    )
    raw_error_source: ErrorSource = EnumField(
        "raw_error_source", ErrorSource, FieldTypes.int8
    )
    raw_error_data: Optional[Exception] = AnyField("raw_error_data")

    @classmethod
    def from_exception(cls, exc: Exception):
        remote_exc = RemoteException.from_exception(exc)
        messages, tracebacks = remote_exc.messages, remote_exc.tracebacks
        return cls(messages, tracebacks, ErrorSource.PYTHON, exc)

    def reraise(self):
        if self.raw_error_source == ErrorSource.PYTHON:
            raise self.raw_error_data
        raise RemoteException(self.error_messages, self.error_tracebacks, [])

    @classmethod
    def from_json(cls, serialized: dict) -> "ErrorInfo":
        kw = serialized.copy()
        kw["raw_error_source"] = ErrorSource(serialized["raw_error_source"])
        if kw.get("raw_error_data"):
            bufs = [base64.b64decode(s) for s in kw["raw_error_data"]]
            try:
                kw["raw_error_data"] = pickle.loads(bufs[0], buffers=bufs[1:])
            except:
                kw["raw_error_data"] = None
        return cls(**kw)

    def to_json(self) -> dict:
        ret = {
            "error_messages": self.error_messages,
            "error_tracebacks": self.error_tracebacks,
            "raw_error_source": self.raw_error_source.value,
        }
        err_data_bufs = None
        if isinstance(self.raw_error_data, (PickleContainer, RemoteException)):
            err_data_bufs = self.raw_error_data.get_buffers()
        elif isinstance(self.raw_error_data, BaseException):
            err_data_bufs = pickle_buffers(self.raw_error_data)
        if err_data_bufs:
            ret["raw_error_data"] = [
                base64.b64encode(s).decode() for s in err_data_bufs
            ]
        return ret


class DagInfo(JsonSerializable):
    session_id: str = StringField("session_id", default=None)
    dag_id: str = StringField("dag_id", default=None)
    status: DagStatus = EnumField("status", DagStatus, FieldTypes.int8, default=None)
    progress: float = Float64Field("progress", default=None)
    tileable_to_result_infos: Dict[str, ResultInfo] = DictField(
        "tileable_to_result_infos",
        FieldTypes.string,
        FieldTypes.reference,
        default_factory=dict,
    )
    error_info: Optional[ErrorInfo] = ReferenceField("error_info", default=None)

    @classmethod
    def from_json(cls, serialized: dict) -> "DagInfo":
        kw = serialized.copy()
        kw["status"] = DagStatus(kw["status"])
        if kw.get("tileable_to_result_infos"):
            kw["tileable_to_result_infos"] = {
                k: ResultInfo.from_json(s)
                for k, s in kw["tileable_to_result_infos"].items()
            }
        if kw.get("error_info"):
            kw["error_info"] = ErrorInfo.from_json(kw["error_info"])
        return DagInfo(**kw)

    def to_json(self) -> dict:
        ret = {
            "session_id": self.session_id,
            "dag_id": self.dag_id,
            "status": self.status.value,
            "progress": self.progress,
        }
        if self.tileable_to_result_infos:
            ret["tileable_to_result_infos"] = {
                k: v.to_json() for k, v in self.tileable_to_result_infos.items()
            }
        if self.error_info:
            ret["error_info"] = self.error_info.to_json()
        return ret


class CreateSessionRequest(Serializable):
    settings: Dict[str, Any] = DictField("settings", default=None)


class SessionInfo(JsonSerializable):
    session_id: str = StringField("session_id")
    settings: Dict[str, Any] = DictField(
        "settings", key_type=FieldTypes.string, default=None
    )
    start_timestamp: float = Float64Field("start_timestamp", default=None)
    idle_timestamp: float = Float64Field("idle_timestamp", default=None)
    dag_infos: Dict[str, Optional[DagInfo]] = DictField(
        "dag_infos",
        key_type=FieldTypes.string,
        value_type=FieldTypes.reference,
        default=None,
    )
    error_info: Optional[ErrorInfo] = ReferenceField("error_info", default=None)

    @classmethod
    def from_json(cls, serialized: dict) -> "SessionInfo":
        kw = serialized.copy()
        if kw.get("dag_infos"):
            kw["dag_infos"] = {
                k: DagInfo.from_json(v) for k, v in kw["dag_infos"].items()
            }
        if kw.get("error_info"):
            kw["error_info"] = ErrorInfo.from_json(kw["error_info"])
        return SessionInfo(**kw)

    def to_json(self) -> dict:
        ret = {
            "session_id": self.session_id,
            "settings": self.settings,
            "start_timestamp": self.start_timestamp,
            "idle_timestamp": self.idle_timestamp,
        }
        if self.dag_infos:
            ret["dag_infos"] = {k: v.to_json() for k, v in self.dag_infos.items()}
        if self.error_info:
            ret["error_info"] = self.error_info.to_json()
        return ret


class ExecuteDagRequest(Serializable):
    session_id: str = StringField("session_id")
    dag: TileableGraph = ReferenceField(
        "dag",
        on_serialize=SerializableGraph.from_graph,
        on_deserialize=lambda x: x.to_graph(),
        default=None,
    )
    managed_input_infos: Dict[str, ResultInfo] = DictField(
        "managed_input_infos",
        key_type=FieldTypes.string,
        value_type=FieldTypes.reference,
        default=None,
    )


class SubDagInfo(Serializable):
    subdag_id: str = StringField("subdag_id")
    status: DagStatus = EnumField("status", DagStatus, FieldTypes.int8, default=None)
    progress: float = Float64Field("progress", default=None)
    error_info: Optional[ErrorInfo] = ReferenceField(
        "error_info", reference_type=ErrorInfo, default=None
    )
    tileable_to_result_infos: Dict[str, ResultInfo] = DictField(
        "tileable_to_result_infos",
        FieldTypes.string,
        FieldTypes.reference,
        default_factory=dict,
    )


class ExecuteSubDagRequest(Serializable):
    dag: TileableGraph = ReferenceField(
        "dag",
        on_serialize=SerializableGraph.from_graph,
        on_deserialize=lambda x: x.to_graph(),
        default=None,
    )
    settings: Dict[str, Any] = DictField("settings", default=None)


class DecrefRequest(Serializable):
    keys: List[str] = ListField("keys", FieldTypes.string, default=None)


class DataFrameTableMeta(Serializable):
    __slots__ = "_pd_column_names", "_pd_index_level_names"

    table_name: Optional[str] = StringField("table_name", default=None)
    type: OutputType = EnumField("type", OutputType, FieldTypes.int8, default=None)
    table_column_names: List[str] = ListField(
        "table_column_names", FieldTypes.string, default=None
    )
    table_index_column_names: List[str] = ListField(
        "table_index_column_names", FieldTypes.string, default=None
    )
    pd_column_dtypes: pd.Series = SeriesField("pd_column_dtypes", default=None)
    pd_column_level_names: List[Any] = ListField("pd_column_level_names", default=None)
    pd_index_dtypes: pd.Series = SeriesField("pd_index_dtypes", default=None)

    @property
    def pd_column_names(self) -> List[Any]:
        try:
            return self._pd_column_names
        except AttributeError:
            self._pd_column_names = self.pd_column_dtypes.index.tolist()
            return self._pd_column_names

    @property
    def pd_index_level_names(self) -> List[Any]:
        try:
            return self._pd_index_level_names
        except AttributeError:
            self._pd_index_level_names = self.pd_index_dtypes.index.tolist()
            return self._pd_index_level_names

    def __eq__(self, other: "Serializable") -> bool:
        if not isinstance(other, type(self)):
            return False
        for k in self._FIELDS:
            v = getattr(self, k, None)
            is_same = v == getattr(other, k, None)
            if callable(getattr(is_same, "all", None)):
                is_same = is_same.all()
            if not is_same:
                return False
        return True
