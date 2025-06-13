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

import pytest
from odps import ODPS

from ....tests.utils import create_test_volume, get_test_unique_name, tn
from ..volumeio import ODPSVolumeReader, ODPSVolumeWriter


@pytest.fixture
def create_volume(request, oss_config):
    test_vol_name = tn(f"test_vol_name_{get_test_unique_name(5)}_" + request.param)
    odps_entry = ODPS.from_environments()

    @contextlib.contextmanager
    def create_parted_volume():
        try:
            odps_entry.delete_volume(test_vol_name)
        except:
            pass
        try:
            odps_entry.create_parted_volume(test_vol_name)
            yield
        finally:
            try:
                odps_entry.delete_volume(test_vol_name)
            except BaseException:
                pass

    if request.param == "parted":
        ctx = create_parted_volume()
    else:
        ctx = create_test_volume(test_vol_name, oss_config)

    with ctx:
        yield test_vol_name


@pytest.mark.parametrize("create_volume", ["external"], indirect=True)
def test_read_write_volume(create_volume):
    test_vol_dir = "test_vol_dir"

    odps_entry = ODPS.from_environments()

    writer = ODPSVolumeWriter(
        odps_entry, create_volume, test_vol_dir, replace_internal_host=True
    )

    writer = ODPSVolumeWriter(
        odps_entry, create_volume, test_vol_dir, replace_internal_host=True
    )
    writer.write_file("file1", b"content1")
    writer.write_file("file2", b"content2")

    reader = ODPSVolumeReader(
        odps_entry, create_volume, test_vol_dir, replace_internal_host=True
    )
    assert reader.read_file("file1") == b"content1"
    assert reader.read_file("file2") == b"content2"

    assert ["file1", "file2"] == sorted(reader.list_files())
