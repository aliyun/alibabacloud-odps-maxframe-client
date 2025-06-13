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

import numpy as np
import pytest
from odps import ODPS

from ....core import OutputType
from ....core.operator import ObjectOperatorMixin, Operator
from ....tensor.datasource import ArrayDataSource
from ....tests.utils import create_test_volume, get_test_unique_name, tn
from ...odpsio import ODPSVolumeReader, ODPSVolumeWriter
from ..core import get_object_io_handler


class TestObjectOp(Operator, ObjectOperatorMixin):
    def __call__(self):
        self._output_types = [OutputType.object]
        return self.new_tileable([])


@pytest.fixture(scope="module")
def create_volume(oss_config):
    with create_test_volume(
        tn("test_object_io_vol_" + get_test_unique_name(5)), oss_config
    ) as test_vol_name:
        yield test_vol_name


def test_simple_object_io(create_volume):
    obj = TestObjectOp()()
    data = "abcdefg"

    odps_entry = ODPS.from_environments()

    reader = ODPSVolumeReader(
        odps_entry, create_volume, obj.key, replace_internal_host=True
    )
    writer = ODPSVolumeWriter(
        odps_entry, create_volume, obj.key, replace_internal_host=True
    )

    handler = get_object_io_handler(obj)()
    handler.write_object(writer, obj, data)
    assert data == handler.read_object(reader, obj)


def test_tensor_object_io(create_volume):
    data = np.array([[4, 9, 2], [3, 5, 7], [8, 1, 6]])
    obj = ArrayDataSource(data, dtype=data.dtype)(data.shape)

    odps_entry = ODPS.from_environments()

    reader = ODPSVolumeReader(
        odps_entry, create_volume, obj.key, replace_internal_host=True
    )
    writer = ODPSVolumeWriter(
        odps_entry, create_volume, obj.key, replace_internal_host=True
    )

    # test write and read full object
    handler = get_object_io_handler(obj)()
    handler.write_object(writer, obj, data)
    np.testing.assert_equal(data, handler.read_object(reader, obj))

    # test read single chunk
    params = {"index": (0, 0)}
    np.testing.assert_equal(data, handler.read_object_body(reader, params))
