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
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple, Union

from ...utils import implements, lazy_import
from ._oss_lib import common as oc
from ._oss_lib.common import HostEnforceType
from ._oss_lib.glob import glob
from ._oss_lib.handle import OSSIOBase
from .base import FileSystem, path_type

oss2 = lazy_import("oss2", placeholder=True)

_oss_time_out = 10


class AuthMode(enum.Enum):
    AK_SK = "ak_sk"
    ROLE_ARN = "role_arn"
    STS_TOKEN = "sts_token"


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
        return oc.build_oss_path(
            path,
            access_key_id=self._access_key_id,
            access_key_secret=self._access_key_secret,
            security_token=self._security_token,
            host_enforce_type=self._host_enforce_type,
        )

    @property
    def protocol(self) -> str:
        return "oss"

    @implements(FileSystem.cat)
    def cat(self, path: path_type):
        raise NotImplementedError

    @implements(FileSystem.ls)
    def ls(self, path: path_type) -> List[path_type]:
        file_list = []
        path = self._rewrite_path(path)
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
                    oc.build_oss_path(
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

    @implements(FileSystem.cp)
    def cp(self, path: path_type, new_path: path_type):
        oc.oss_copy_file(self._rewrite_path(path), self._rewrite_path(new_path))

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
        if not self.isdir(path):
            return
        parsed_path = oc.parse_osspath(self._rewrite_path(path))
        path = parsed_path.key.rstrip("/") + "/"
        parsed_path = parsed_path._replace(key=path)

        all_subfiles = sorted(self.ls(str(parsed_path)))
        prefixes_to_contents = defaultdict(lambda: (set(), set()))
        for file_path in all_subfiles:
            parsed_sub_path = oc.parse_osspath(file_path)
            if path == parsed_sub_path.key:
                continue
            rel_path = parsed_sub_path.key[len(parsed_path.key) :].lstrip("/")
            if "/" not in rel_path:
                prefixes_to_contents[parsed_path.key][1].add(rel_path)
            else:
                rel_root, fn = rel_path.split("/", 1)
                cur_root = parsed_path.key + rel_root.strip("/") + "/"
                prefixes_to_contents[cur_root][1].add(fn)
                rel_prefix = ""
                for part in rel_root.split("/"):
                    cur_root = (parsed_path.key + rel_prefix.lstrip("/")).rstrip(
                        "/"
                    ) + "/"
                    prefixes_to_contents[cur_root][0].add(part)
                    rel_prefix += "/" + part
        for root, (dirs, files) in sorted(prefixes_to_contents.items()):
            yield str(parsed_path._replace(key=root)), sorted(dirs), sorted(files)

    @implements(FileSystem.glob)
    def glob(self, path: path_type, recursive: bool = False) -> List[path_type]:
        return glob(self._rewrite_path(path), recursive=recursive)

    @implements(FileSystem.init_multipart_upload)
    def init_multipart_upload(self, path: path_type) -> str:
        path = self._rewrite_path(path)
        parsed_path = oc.parse_osspath(path)
        oss_bucket = oc.get_oss_bucket(parsed_path)
        return oss_bucket.init_multipart_upload(parsed_path.key).upload_id

    @implements(FileSystem.open_part_writer)
    def open_part_writer(
        self, path: path_type, upload_id: str, part_num: int
    ) -> OSSIOBase:
        path = self._rewrite_path(path)
        return OSSIOBase(path, "wb", upload_id, part_num)

    @implements(FileSystem.complete_multipart_upload)
    def complete_multipart_upload(
        self, path: path_type, upload_id: str, parts: List[Any]
    ) -> None:
        path = self._rewrite_path(path)
        parsed_path = oc.parse_osspath(path)
        oss_bucket = oc.get_oss_bucket(parsed_path)
        parts = [oss2.models.PartInfo(**info) for info in parts]
        oss_bucket.complete_multipart_upload(parsed_path.key, upload_id, parts)

    @implements(FileSystem.abort_multipart_upload)
    def abort_multipart_upload(self, path: path_type, upload_id: str) -> None:
        path = self._rewrite_path(path)
        parsed_path = oc.parse_osspath(path)
        oss_bucket = oc.get_oss_bucket(parsed_path)
        try:
            oss_bucket.abort_multipart_upload(parsed_path.key, upload_id)
        except oss2.exceptions.NoSuchUpload:
            pass

    @property
    def supports_partial_overwrite(self) -> bool:
        return False

    @property
    def supports_multipart_upload(self) -> bool:
        return True
