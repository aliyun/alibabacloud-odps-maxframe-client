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

import pytest
from odps import ODPS

from ....tests.utils import tn
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

        if "test" in oss_endpoint:
            # offline config
            test_location = "oss://%s:%s@%s/%s/%s" % (
                oss_access_id,
                oss_secret_access_key,
                oss_endpoint,
                oss_bucket_name,
                oss_test_dir_name,
            )
            rolearn = None
        else:
            # online config
            endpoint_parts = oss_endpoint.split(".", 1)
            if "-internal" not in endpoint_parts[0]:
                endpoint_parts[0] += "-internal"
            test_location = "oss://%s/%s/%s" % (
                ".".join(endpoint_parts),
                oss_bucket_name,
                oss_test_dir_name,
            )
            rolearn = oss_config.oss_rolearn

        oss_config.oss_bucket.put_object(oss_test_dir_name + "/", b"")
        odps_entry.create_external_volume(
            test_vol_name, location=test_location, rolearn=rolearn
        )
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
