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
import numpy as np
import pytest
from odps import ODPS

from ....core import OutputType
from ....core.operator import ObjectOperatorMixin, Operator
from ....tensor.datasource import ArrayDataSource
from ....tests.utils import tn
from ...odpsio import ODPSVolumeReader, ODPSVolumeWriter
from ..core import get_object_io_handler


class TestObjectOp(Operator, ObjectOperatorMixin):
    def __call__(self):
        self._output_types = [OutputType.object]
        return self.new_tileable([])


@pytest.fixture(scope="module")
def create_volume(request, oss_config):
    test_vol_name = tn("test_object_io_volume")
    odps_entry = ODPS.from_environments()

    try:
        odps_entry.delete_volume(test_vol_name, auto_remove_dir=True, recursive=True)
    except:
        pass

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
            odps_entry.delete_volume(
                test_vol_name, auto_remove_dir=True, recursive=True
            )
        except:
            pass


def test_simple_object_io(create_volume):
    obj = TestObjectOp()()
    data = "abcdefg"

    odps_entry = ODPS.from_environments()

    reader = ODPSVolumeReader(odps_entry, create_volume, obj.key)
    writer = ODPSVolumeWriter(odps_entry, create_volume, obj.key)

    handler = get_object_io_handler(obj)()
    handler.write_object(writer, obj, data)
    assert data == handler.read_object(reader, obj)


def test_tensor_object_io(create_volume):
    data = np.array([[4, 9, 2], [3, 5, 7], [8, 1, 6]])
    obj = ArrayDataSource(data, dtype=data.dtype)(data.shape)

    odps_entry = ODPS.from_environments()

    reader = ODPSVolumeReader(odps_entry, create_volume, obj.key)
    writer = ODPSVolumeWriter(odps_entry, create_volume, obj.key)

    handler = get_object_io_handler(obj)()
    handler.write_object(writer, obj, data)
    np.testing.assert_equal(data, handler.read_object(reader, obj))
