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

import logging
import os
from typing import NamedTuple, Optional
from urllib.parse import parse_qs, urlparse

from ....utils import lazy_import
from ..base import path_type, stringify_path

oss2 = lazy_import("oss2", placeholder=True)

logger = logging.getLogger(__name__)

# OSS api time out
_oss_time_out = 10


class OSSFileEntry:
    def __init__(
        self, path, *, is_dir=None, is_file=None, stat=None, storage_options=None
    ):
        self._path = path
        self._name = os.path.basename(path)
        self._is_file = is_file
        self._is_dir = is_dir
        self._stat = stat
        self._storage_options = storage_options

    def is_dir(self):
        if self._path.endswith("/"):
            self._is_dir = True
        if self._is_dir is None:
            self._is_dir = oss_isdir(self._path)
        return self._is_dir

    def is_file(self):
        if self._is_file is None:
            if self.is_dir() or not oss_exists(self._path):
                self._is_file = False
            else:
                self._is_file = True
        return self._is_file

    def stat(self):
        if self._stat is None:
            self._stat = oss_stat(self._path)
        return self._stat

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path


class ParsedOSSPath(NamedTuple):
    endpoint: str
    bucket: str
    key: str
    access_key_id: Optional[str] = None
    access_key_secret: Optional[str] = None
    security_token: Optional[str] = None
    scheme: str = None


def parse_osspath(path: path_type, check_errors: bool = True) -> ParsedOSSPath:
    # Extract OSS configuration from the encoded URL.
    str_path = stringify_path(path)
    parse_result = urlparse(str_path)
    if check_errors and parse_result.scheme != "oss":
        raise ValueError(
            f"Except scheme oss, but got scheme: {parse_result.scheme}"
            f" in path: {str_path}"
        )
    access_key_id = parse_result.username
    access_key_secret = parse_result.password

    if not parse_result.query:
        sts_token = None
    else:
        sts_token = parse_qs(parse_result.query).get("security_token", [None])[0]

    if check_errors and not (access_key_id and access_key_secret):
        raise ValueError(r"No credentials provided")

    key = parse_result.path
    key = key[1:] if key.startswith("/") else key
    if "/" not in key:
        bucket, key = key, None
        if check_errors:
            raise ValueError("Need to use format bucket/key to separate bucket and key")
    else:
        bucket, key = key.split("/", 1)

    endpoint = parse_result.hostname
    if endpoint and parse_result.port:
        endpoint += f":{parse_result.port}"
    return ParsedOSSPath(
        endpoint,
        bucket,
        key,
        access_key_id,
        access_key_secret,
        sts_token,
        parse_result.scheme,
    )


def get_oss_bucket(parsed_path: ParsedOSSPath):
    if parsed_path.security_token is not None:
        auth = oss2.StsAuth(
            parsed_path.access_key_id,
            parsed_path.access_key_secret,
            parsed_path.security_token,
        )
    else:
        auth = oss2.Auth(parsed_path.access_key_id, parsed_path.access_key_secret)
    oss_bucket = oss2.Bucket(
        auth=auth,
        endpoint=parsed_path.endpoint,
        bucket_name=parsed_path.bucket,
        connect_timeout=_oss_time_out,
    )
    return oss_bucket


def oss_exists(path: path_type):
    parsed_path = parse_osspath(path)
    oss_bucket = get_oss_bucket(parsed_path)
    return oss_bucket.object_exists(parsed_path.key) or oss_isdir(path)


def oss_isdir(path: path_type):
    """
    OSS has no concept of directories, but we define
    a ossurl is dir, When there is at least one object
    at the ossurl that is the prefix(end with char "/"),
    it is considered as a directory.
    """
    dirname = stringify_path(path)
    if not dirname.endswith("/"):
        dirname = dirname + "/"
    logger.info("Checking isdir for path %s", dirname)
    parsed_path = parse_osspath(dirname)
    oss_bucket = get_oss_bucket(parsed_path)
    isdir = False
    for obj in oss2.ObjectIteratorV2(oss_bucket, prefix=parsed_path.key, max_keys=2):
        if obj.key == parsed_path.key:
            continue
        isdir = True
        break
    return isdir


def oss_delete(path: path_type):
    """
    Perform both key deletion and prefix deletion. Once no files
    deleted in both scenarios, we can make assertion that the file
    does not exist.
    """
    parsed_path = parse_osspath(path)
    oss_bucket = get_oss_bucket(parsed_path)

    try:
        oss_bucket.delete_object(parsed_path.key)
        return
    except oss2.exceptions.NoSuchKey:
        pass

    is_missing = True
    dir_key = parsed_path.key.rstrip("/") + "/"
    for obj in oss2.ObjectIteratorV2(oss_bucket, prefix=dir_key):
        try:
            oss_bucket.delete_object(obj.key)
            is_missing = False
        except oss2.exceptions.NoSuchKey:
            pass
    if is_missing:
        raise FileNotFoundError("No such file or directory: %s", path)


def oss_copy_file(src_path: path_type, dest_path: path_type):
    # todo implements copy of huge files
    parsed_src_path = parse_osspath(src_path)
    parsed_dest_path = parse_osspath(dest_path)
    try:
        if oss_isdir(src_path):
            raise NotImplementedError("Copying directories not implemented yet")
    except:
        # fixme currently we cannot handle error with iterating files with STS token
        logger.exception("Failed to judge if src is a directory")

    oss_bucket = get_oss_bucket(parsed_dest_path)
    oss_bucket.copy_object(
        parsed_src_path.bucket, parsed_src_path.key, parsed_dest_path.key
    )


def oss_stat(path: path_type):
    path = stringify_path(path)
    parsed_path = parse_osspath(path)
    oss_bucket = get_oss_bucket(parsed_path)
    if oss_isdir(path):
        stat = dict(name=path, size=0, modified_time=-1)
        stat["type"] = "directory"
    else:
        meta = oss_bucket.get_object_meta(parsed_path.key)
        stat = dict(
            name=path,
            size=int(meta.headers["Content-Length"]),
            modified_time=meta.headers["Last-Modified"],
        )
        stat["type"] = "file"
    return stat


def oss_scandir(dirname: path_type):
    dirname = stringify_path(dirname)
    if not dirname.endswith("/"):
        dirname = dirname + "/"
    parsed_path = parse_osspath(dirname)
    oss_bucket = get_oss_bucket(parsed_path)
    dirname_set = set()
    for obj in oss2.ObjectIteratorV2(oss_bucket, prefix=parsed_path.key):
        rel_path = obj.key[len(parsed_path.key) :]
        try:
            inside_dirname, inside_filename = rel_path.split("/", 1)
        except ValueError:
            inside_dirname = None
            inside_filename = rel_path
        if inside_dirname is not None:
            if inside_dirname in dirname_set:
                continue
            dirname_set.add(inside_dirname)
            yield OSSFileEntry(
                "/".join([dirname, inside_dirname]),
                is_dir=True,
                is_file=False,
                stat={
                    "name": "/".join([dirname, inside_dirname]),
                    "type": "directory",
                    "size": 0,
                    "modified_time": -1,
                },
            )
        else:
            yield OSSFileEntry(
                "/".join([dirname, inside_filename]),
                is_dir=False,
                is_file=True,
                stat={
                    "name": "/".join([dirname, inside_filename]),
                    "type": "file",
                    "size": obj.size,
                    "modified_time": obj.last_modified,
                },
            )
