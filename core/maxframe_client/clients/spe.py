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

from typing import Any, Dict, Optional

from tornado import httpclient

from maxframe.core import TileableGraph
from maxframe.protocol import ExecuteSubDagRequest, ProtocolBody, SubDagInfo
from maxframe.typing_ import TimeoutType
from maxframe.utils import (
    deserialize_serializable,
    format_timeout_params,
    serialize_serializable,
    wait_http_response,
)


class SPEClient:
    def __init__(
        self,
        endpoint: str,
        session_id: Optional[str] = None,
        host: str = None,
    ):
        self._endpoint = endpoint.rstrip("/")
        self._session_id = session_id
        self._headers = {"Host": host}

    @staticmethod
    def _load_subdag_info(resp: httpclient.HTTPResponse) -> SubDagInfo:
        res: ProtocolBody[SubDagInfo] = deserialize_serializable(resp.body)
        return res.body

    async def submit_subdag(
        self, subdag: TileableGraph, settings: Dict[str, Any] = None
    ) -> SubDagInfo:
        req_url = f"{self._endpoint}/api/subdags"
        req_body: ProtocolBody[ExecuteSubDagRequest] = ProtocolBody(
            body=ExecuteSubDagRequest(dag=subdag, settings=settings),
        )

        if self._session_id is not None:
            req_url += f"?session_id={self._session_id}"

        resp = await httpclient.AsyncHTTPClient().fetch(
            req_url,
            method="POST",
            headers=self._headers,
            body=serialize_serializable(req_body),
        )
        return self._load_subdag_info(resp)

    async def get_subdag_info(self, subdag_id: str) -> SubDagInfo:
        req_url = f"{self._endpoint}/api/subdags/{subdag_id}?wait=0"
        resp = await httpclient.AsyncHTTPClient().fetch(
            req_url,
            method="GET",
            headers=self._headers,
        )
        return self._load_subdag_info(resp)

    async def wait_subdag(
        self, subdag_id: str, wait_timeout: TimeoutType = None
    ) -> SubDagInfo:
        req_url = f"{self._endpoint}/api/subdags/{subdag_id}"
        params = format_timeout_params(wait_timeout)
        try:
            resp = await wait_http_response(
                req_url + params,
                method="GET",
                headers=self._headers,
                request_timeout=wait_timeout,
            )
            return self._load_subdag_info(resp)
        except TimeoutError:
            return await self.get_subdag_info(subdag_id)

    async def cancel_subdag(
        self, subdag_id: str, wait_timeout: TimeoutType = None
    ) -> SubDagInfo:
        req_url = f"{self._endpoint}/api/subdags/{subdag_id}"
        params = format_timeout_params(wait_timeout)
        try:
            resp = await wait_http_response(
                req_url + params,
                method="DELETE",
                headers=self._headers,
                request_timeout=wait_timeout,
            )
            return self._load_subdag_info(resp)
        except TimeoutError:
            return await self.get_subdag_info(subdag_id)
