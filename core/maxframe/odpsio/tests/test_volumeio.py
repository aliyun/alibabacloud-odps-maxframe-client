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

import pytest
from odps import ODPS

from ...tests.utils import tn
from ..volumeio import ODPSVolumeReader, ODPSVolumeWriter


@pytest.fixture
def create_volume(request, oss_config):
    test_vol_name = tn("test_vol_name_" + request.param)
    odps_entry = ODPS.from_environments()

    try:
        odps_entry.delete_volume(test_vol_name)
    except:
        pass

    oss_test_dir_name = None
    if request.param == "parted":
        odps_entry.create_parted_volume(test_vol_name)
    else:
        oss_test_dir_name = tn("test_oss_directory")
        if oss_config is None:
            pytest.skip("Need oss and its config to run this test")
        (
            oss_access_id,
            oss_secret_access_key,
            oss_bucket_name,
            oss_endpoint,
        ) = oss_config.oss_config
        test_location = "oss://%s:%s@%s/%s/%s" % (
            oss_access_id,
            oss_secret_access_key,
            oss_endpoint,
            oss_bucket_name,
            oss_test_dir_name,
        )
        oss_config.oss_bucket.put_object(oss_test_dir_name + "/", b"")
        odps_entry.create_external_volume(test_vol_name, location=test_location)
    try:
        yield test_vol_name
    finally:
        try:
            odps_entry.delete_volume(test_vol_name)
        except BaseException:
            pass

        if oss_test_dir_name is not None:
            import oss2

            keys = [
                obj.key
                for obj in oss2.ObjectIterator(oss_config.oss_bucket, oss_test_dir_name)
            ]
            oss_config.oss_bucket.batch_delete_objects(keys)


@pytest.mark.parametrize("create_volume", ["parted", "external"], indirect=True)
def test_read_write_volume(create_volume):
    test_vol_dir = "test_vol_dir"

    odps_entry = ODPS.from_environments()

    writer = ODPSVolumeWriter(odps_entry, create_volume, test_vol_dir)
    write_session_id = writer.create_write_session()

    writer = ODPSVolumeWriter(odps_entry, create_volume, test_vol_dir)
    writer.write_file("file1", b"content1", write_session_id)
    writer.write_file("file2", b"content2", write_session_id)
    writer.commit(["file1", "file2"], write_session_id)

    reader = ODPSVolumeReader(odps_entry, create_volume, test_vol_dir)
    assert reader.read_file("file1") == b"content1"
    assert reader.read_file("file2") == b"content2"

    assert ["file1", "file2"] == sorted(reader.list_files())
