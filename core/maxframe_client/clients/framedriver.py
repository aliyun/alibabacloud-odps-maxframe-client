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

from typing import Any, Dict, List

from tornado import httpclient

from maxframe.core import TileableGraph
from maxframe.protocol import (
    CreateSessionRequest,
    DagInfo,
    DecrefRequest,
    ExecuteDagRequest,
    ProtocolBody,
    ResultInfo,
    SessionInfo,
)
from maxframe.typing_ import TimeoutType
from maxframe.utils import (
    deserialize_serializable,
    format_timeout_params,
    serialize_serializable,
    wait_http_response,
)


class FrameDriverClient:
    def __init__(self, endpoint: str):
        self._endpoint = endpoint.rstrip("/")

    async def create_session(self, settings: Dict[str, Any] = None) -> SessionInfo:
        req_body: ProtocolBody[CreateSessionRequest] = ProtocolBody(
            body=CreateSessionRequest(settings=settings)
        )
        req_url = f"{self._endpoint}/api/sessions"
        resp = await httpclient.AsyncHTTPClient().fetch(
            req_url, method="POST", body=serialize_serializable(req_body)
        )
        return deserialize_serializable(resp.body).body

    async def get_session(self, session_id: str) -> SessionInfo:
        req_url = f"{self._endpoint}/api/sessions/{session_id}"
        resp = await httpclient.AsyncHTTPClient().fetch(req_url, method="GET")
        return deserialize_serializable(resp.body).body

    async def delete_session(self, session_id: str):
        req_url = f"{self._endpoint}/api/sessions/{session_id}"
        await httpclient.AsyncHTTPClient().fetch(req_url, method="DELETE")

    async def submit_dag(
        self,
        session_id: str,
        dag: TileableGraph,
        managed_input_infos: Dict[str, ResultInfo] = None,
    ) -> DagInfo:
        req_url = f"{self._endpoint}/api/sessions/{session_id}/dags"
        req_body = ExecuteDagRequest(session_id, dag, managed_input_infos)
        resp = await httpclient.AsyncHTTPClient().fetch(
            req_url,
            method="POST",
            body=serialize_serializable(ProtocolBody(body=req_body)),
        )
        return deserialize_serializable(resp.body).body

    async def get_dag_info(self, session_id: str, dag_id: str) -> DagInfo:
        req_url = f"{self._endpoint}/api/sessions/{session_id}/dags/{dag_id}"
        resp = await httpclient.AsyncHTTPClient().fetch(req_url, method="GET")
        return deserialize_serializable(resp.body).body

    async def wait_dag(self, session_id: str, dag_id: str, timeout: TimeoutType = None):
        query_part = format_timeout_params(timeout)
        req_url = (
            f"{self._endpoint}/api/sessions/{session_id}/dags/{dag_id}{query_part}"
        )
        try:
            resp = await wait_http_response(
                req_url, method="GET", request_timeout=timeout
            )
            info = deserialize_serializable(resp.body).body
        except TimeoutError:
            info = await self.get_dag_info(session_id, dag_id)
        return info

    async def cancel_dag(
        self, session_id: str, dag_id: str, timeout: TimeoutType = None
    ):
        query_part = format_timeout_params(timeout)
        req_url = (
            f"{self._endpoint}/api/sessions/{session_id}/dags/{dag_id}{query_part}"
        )
        try:
            resp = await wait_http_response(
                req_url, method="DELETE", request_timeout=timeout
            )
            info = deserialize_serializable(resp.body).body
        except TimeoutError:
            info = await self.get_dag_info(session_id, dag_id)
        return info

    async def decref(self, session_id: str, tileable_keys: List[str]) -> None:
        req_url = f"{self._endpoint}/api/sessions/{session_id}/lifecycle?decref"
        req_body = DecrefRequest(tileable_keys)
        await wait_http_response(
            req_url,
            method="POST",
            body=serialize_serializable(ProtocolBody(body=req_body)),
        )
