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

from ....utils import extract_class_name
from ..objects import Object, ObjectData


class TestSubObjectData(ObjectData):
    __test__ = False


class TestSubObject(Object):
    __test__ = False


def test_object_init():
    assert TestSubObjectData.get_entity_class() is TestSubObject

    obj_data = ObjectData(object_class=extract_class_name(TestSubObjectData))
    assert isinstance(obj_data, TestSubObjectData)
    obj = Object(obj_data)
    assert isinstance(obj, TestSubObject)

    obj_data = ObjectData(object_class=TestSubObjectData)
    assert isinstance(obj_data, TestSubObjectData)

    obj_data = ObjectData(object_class=extract_class_name(TestSubObject))
    assert isinstance(obj_data, TestSubObjectData)

    obj_data = ObjectData(object_class=TestSubObject)
    assert isinstance(obj_data, TestSubObjectData)
