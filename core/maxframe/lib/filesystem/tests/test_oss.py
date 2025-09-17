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

import os
import time
from io import BytesIO

import mock
import pytest

from .. import oss
from .._oss_lib import glob as og
from .._oss_lib.common import OSSFileEntry
from ..oss import HostEnforceType, _rewrite_internal_endpoint, build_oss_path


class OSSObjInfo:
    def __init__(self, name, content):
        self.key = name
        # Use the current time as "Last-Modified" in the test.
        self.last_modified = int(time.time())
        self.size = len(content.encode("utf8"))


class ObjectMeta:
    def __init__(self, key, obj_dict):
        self.headers = {}
        self.headers["Last-Modified"] = int(time.time())
        self.headers["Content-Length"] = len(obj_dict[key].encode("utf8"))


class MockObject:
    def __init__(self, obj_dict, key, byte_range):
        self._stream = BytesIO(obj_dict[key].encode("utf8"))
        self._byte_range = byte_range

    def read(self):
        self._stream.seek(self._byte_range[0])
        if self._byte_range[1] is None:
            return self._stream.read()
        else:
            size = self._byte_range[1] - self._byte_range[0] + 1
            return self._stream.read(size)


class SideEffectBucket:
    cached_ctx = {}

    def __init__(self, *_, **__):
        cur_test = os.environ["PYTEST_CURRENT_TEST"]
        if cur_test in self.cached_ctx:
            self.obj_dict = self.cached_ctx[cur_test]
        else:
            self.obj_dict = self.cached_ctx[cur_test] = {
                "file.csv": "id1,id2,id3\n1,2,3\n",
                "dir/": "",
                "dir/file1.csv": "2",
                "dir/file2.csv": "3",
                "dir/subdir/": "",
                "dir/subdir/file3.csv": "s4",
                "dir/subdir/file4.csv": "s5",
                "dir2/": "",
                "dir2/file6.csv": "6",
                "dir2/file7.csv": "7",
            }

    def get_object_meta(self, key):
        return ObjectMeta(key, self.obj_dict)

    def object_exists(self, key):
        return key in self.obj_dict.keys()

    def get_object(self, key, byte_range):
        return MockObject(self.obj_dict, key, byte_range)

    def copy_object(self, bucket, src_key, dst_key):
        self.obj_dict[dst_key] = self.obj_dict[src_key]

    def delete_object(self, key):
        from oss2.exceptions import NoSuchKey

        try:
            del self.obj_dict[key]
        except KeyError:
            raise NoSuchKey(404, {}, key, {})


class SideEffectObjIter:
    def __init__(self, *args, **kwargs):
        self.bucket = args[0]
        self.prefix = kwargs["prefix"]

    def __iter__(self):
        for name, content in self.bucket.obj_dict.items():
            if name.startswith(self.prefix):
                yield OSSObjInfo(name, content)


@mock.patch("oss2.Bucket", side_effect=SideEffectBucket)
@mock.patch("oss2.ObjectIteratorV2", side_effect=SideEffectObjIter)
def test_oss_filesystem(fake_obj_iter, fake_oss_bucket):
    access_key_id = "your_access_key_id"
    access_key_secret = "your_access_key_secret"
    end_point = "your_endpoint"

    file_path = f"oss://your_endpoint/bucket/file.csv"
    new_file_path = f"oss://your_endpoint/bucket/file1.csv"
    dir_path = f"oss://your_endpoint/bucket/dir/"
    dir_path_content_magic = f"oss://your_endpoint/bucket/dir*/"
    other_scheme_path = f"scheme://netloc/path"
    not_exist_file_path = f"oss://your_endpoint/bucket/not_exist.csv"

    fake_file_path = build_oss_path(
        file_path, end_point, access_key_id, access_key_secret
    )
    fake_new_file_path = build_oss_path(
        new_file_path, end_point, access_key_id, access_key_secret
    )
    fake_dir_path = build_oss_path(
        dir_path, end_point, access_key_id, access_key_secret
    )
    fake_dir_path_contains_magic = build_oss_path(
        dir_path_content_magic, end_point, access_key_id, access_key_secret
    )
    fake_not_exist_file_path = build_oss_path(
        not_exist_file_path, end_point, access_key_id, access_key_secret
    )
    fs = oss.OSSFileSystem()

    # Test OSSFileSystem.
    assert len(fs.ls(fake_dir_path)) == 4
    assert not fs.isfile(fake_dir_path)
    assert fs.isdir(fake_dir_path)
    assert not fs.isdir(fake_file_path)
    assert fs.isfile(fake_file_path)
    assert fs.exists(fake_file_path)
    assert not fs.exists(fake_not_exist_file_path)
    assert fs.stat(fake_file_path)["type"] == "file"
    assert fs.stat(fake_dir_path)["type"] == "directory"
    assert fs.glob(fake_dir_path) == [fake_dir_path]

    msg1 = f"Except scheme oss, but got scheme: scheme in path: {other_scheme_path}"
    with pytest.raises(ValueError, match=msg1):
        fs.exists(other_scheme_path)

    with pytest.raises(ValueError, match="No credentials provided"):
        fs.exists(file_path)

    with pytest.raises(OSError):
        fs.ls(fake_file_path)

    assert len(fs.glob(fake_file_path)) == 1
    assert len(fs.glob(fake_dir_path + "*", recursive=True)) == 4
    assert len(fs.glob(fake_dir_path_contains_magic)) == 2

    # Test the specific functions of glob.
    assert og.has_magic(b"*")
    assert og.escape(b"*") == b"[*]"
    assert og.escape("*") == "[*]"

    # test OSSIOBase
    with fs.open(fake_file_path) as f:
        assert f.readline() == b"id1,id2,id3\n"
        assert f.readline() == b"1,2,3\n"
        f.seek(-1, 2)
        assert f.readline() == b"\n"
        with pytest.raises(AttributeError):
            f.fileno()
        with pytest.raises(OSError):
            f.seek(-1)
        with pytest.raises(OSError):
            f.seek(-100, 1)
        with pytest.raises(ValueError):
            f.seek(1, 3)
        f.seek(0)
        assert f.read() == b"id1,id2,id3\n1,2,3\n"
        f.seek(0)
        assert f.readline(2) == b"id"
        f.seek(0)
        with pytest.raises(TypeError):
            f.readline("2")

    fe = OSSFileEntry(fake_file_path)
    assert fe.path == fake_file_path

    fs.rename(fake_file_path, fake_new_file_path)
    assert not fs.exists(fake_file_path)
    assert fs.exists(fake_new_file_path)

    with pytest.raises(FileNotFoundError):
        fs.delete(fake_not_exist_file_path)


def test_host_rewrite():
    assert "cn-shanghai.oss.service.com" == _rewrite_internal_endpoint(
        "cn-shanghai.oss.service.com", HostEnforceType.force_external
    )
    assert "cn-shanghai.oss.service.com" == _rewrite_internal_endpoint(
        "cn-shanghai-internal.oss.service.com", HostEnforceType.force_external
    )
    assert "cn-shanghai-internal.oss.service.com" == _rewrite_internal_endpoint(
        "cn-shanghai.oss.service.com", HostEnforceType.force_internal
    )
    assert "cn-shanghai-internal.oss.service.com" == _rewrite_internal_endpoint(
        "cn-shanghai-internal.oss.service.com", HostEnforceType.force_internal
    )
    assert "1.2.3.4" == _rewrite_internal_endpoint(
        "1.2.3.4", HostEnforceType.force_internal
    )
