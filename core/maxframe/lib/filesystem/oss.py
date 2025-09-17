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

import enum
import re
from typing import Dict, Iterator, List, Tuple, Union
from urllib.parse import urlencode

from ...utils import implements, lazy_import
from ._oss_lib import common as oc
from ._oss_lib.glob import glob
from ._oss_lib.handle import OSSIOBase
from .base import FileSystem, path_type

oss2 = lazy_import("oss2", placeholder=True)
_ip_regex = re.compile(r"^([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})")

_oss_time_out = 10


class HostEnforceType(enum.Enum):
    no_enforce = 0
    force_internal = 1
    force_external = 2


class OSSFileSystem(FileSystem):
    def __init__(
        self,
        access_key_id: str = None,
        access_key_secret: str = None,
        security_token: str = None,
        host_enforce_type: Union[HostEnforceType, str] = HostEnforceType.no_enforce,
        **kw,
    ):
        self._access_key_id = access_key_id or kw.get("user")
        self._access_key_secret = access_key_secret or kw.get("password")
        self._security_token = security_token
        self._host_enforce_type = (
            host_enforce_type
            if isinstance(host_enforce_type, HostEnforceType)
            else getattr(HostEnforceType, host_enforce_type)
        )

    def _rewrite_path(self, path: str) -> str:
        return build_oss_path(
            path,
            access_key_id=self._access_key_id,
            access_key_secret=self._access_key_secret,
            security_token=self._security_token,
            host_enforce_type=self._host_enforce_type,
        )

    @implements(FileSystem.cat)
    def cat(self, path: path_type):
        raise NotImplementedError

    @implements(FileSystem.ls)
    def ls(self, path: path_type) -> List[path_type]:
        file_list = []
        file_entry = oc.OSSFileEntry(path)
        if not file_entry.is_dir():
            raise OSError("ls for file is not supported")
        else:
            parsed_path = oc.parse_osspath(path)
            oss_bucket = oc.get_oss_bucket(parsed_path)
            for obj in oss2.ObjectIteratorV2(oss_bucket, prefix=parsed_path.key):
                if obj.key.endswith("/"):
                    continue
                obj_path = rf"{parsed_path.bucket}/{obj.key}"
                file_list.append(
                    build_oss_path(
                        obj_path,
                        parsed_path.endpoint,
                        parsed_path.access_key_id,
                        parsed_path.access_key_secret,
                        parsed_path.security_token,
                    )
                )
        return file_list

    @implements(FileSystem.delete)
    def delete(self, path: path_type, recursive: bool = False):
        return oc.oss_delete(self._rewrite_path(path))

    @implements(FileSystem.rename)
    def rename(self, path: path_type, new_path: path_type):
        # in OSS, you need to move file by copy and delete
        path = self._rewrite_path(path)
        oc.oss_copy_file(path, self._rewrite_path(new_path))
        oc.oss_delete(path)

    @implements(FileSystem.stat)
    def stat(self, path: path_type) -> Dict:
        ofe = oc.OSSFileEntry(self._rewrite_path(path))
        return ofe.stat()

    @implements(FileSystem.mkdir)
    def mkdir(self, path: path_type, create_parents: bool = True):
        raise NotImplementedError

    @implements(FileSystem.isdir)
    def isdir(self, path: path_type) -> bool:
        file_entry = oc.OSSFileEntry(self._rewrite_path(path))
        return file_entry.is_dir()

    @implements(FileSystem.isfile)
    def isfile(self, path: path_type) -> bool:
        file_entry = oc.OSSFileEntry(self._rewrite_path(path))
        return file_entry.is_file()

    @implements(FileSystem._isfilestore)
    def _isfilestore(self) -> bool:
        raise NotImplementedError

    @implements(FileSystem.exists)
    def exists(self, path: path_type):
        return oc.oss_exists(self._rewrite_path(path))

    @implements(FileSystem.open)
    def open(self, path: path_type, mode: str = "rb") -> OSSIOBase:
        file_handle = OSSIOBase(self._rewrite_path(path), mode)
        return file_handle

    @implements(FileSystem.walk)
    def walk(self, path: path_type) -> Iterator[Tuple[str, List[str], List[str]]]:
        raise NotImplementedError

    @implements(FileSystem.glob)
    def glob(self, path: path_type, recursive: bool = False) -> List[path_type]:
        return glob(self._rewrite_path(path), recursive=recursive)


def _rewrite_internal_endpoint(
    endpoint: str, host_enforce_type: HostEnforceType = HostEnforceType.no_enforce
) -> str:
    if (
        not endpoint
        or host_enforce_type == HostEnforceType.no_enforce
        or _ip_regex.match(endpoint)
    ):
        return endpoint

    ep_first, ep_rest = endpoint.split(".", 1)
    host_with_internal = ep_first.endswith("-internal")
    if host_enforce_type == HostEnforceType.force_external and host_with_internal:
        return ep_first.replace("-internal", "") + "." + ep_rest
    elif host_enforce_type == HostEnforceType.force_internal and not host_with_internal:
        return ep_first + "-internal." + ep_rest
    else:
        return endpoint


def build_oss_path(
    path: path_type,
    endpoint: str = None,
    access_key_id: str = None,
    access_key_secret: str = None,
    security_token: str = None,
    host_enforce_type: HostEnforceType = HostEnforceType.no_enforce,
):
    """
    Returns a path with oss info.
    Used to register the access_key_id, access_key_secret and
    endpoint of OSS. The access_key_id and endpoint are put
    into the url with url-safe-base64 encoding.

    Parameters
    ----------
    path : path_type
        The original OSS url.

    endpoint : str
        The endpoint of OSS.

    access_key_id : str
        The access key id of OSS.

    access_key_secret : str
        The access key secret of OSS.

    security_token : str
        The security token of OSS.

    Returns
    -------
    path_type
        Path include the encoded access key id, end point and
        access key secret of oss.
    """
    if isinstance(path, (list, tuple)):
        path = path[0]
    parse_result = oc.parse_osspath(path, check_errors=False)
    access_key_id = parse_result.access_key_id or access_key_id
    access_key_secret = parse_result.access_key_secret or access_key_secret
    security_token = parse_result.security_token or security_token

    scheme = parse_result.scheme or "oss"
    endpoint = _rewrite_internal_endpoint(
        parse_result.endpoint or endpoint, host_enforce_type
    )

    if access_key_id and access_key_secret:
        creds = f"{access_key_id}:{access_key_secret}@"
    else:
        creds = ""

    new_path = f"{scheme}://{creds}{endpoint}/{parse_result.bucket}"
    if parse_result.key:
        new_path += f"/{parse_result.key}"
    if security_token:
        new_path += f"?{urlencode(dict(security_token=security_token))}"
    # reparse to check errors
    oc.parse_osspath(new_path)
    return new_path
