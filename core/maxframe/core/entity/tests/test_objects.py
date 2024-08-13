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

from ..objects import Object, ObjectData


class TestSubObjectData(ObjectData):
    __test__ = False


class TestSubObject(Object):
    __test__ = False


def test_object_init():
    assert TestSubObjectData.get_entity_class() is TestSubObject

    obj = ObjectData(
        object_class=TestSubObjectData.__module__ + "#" + TestSubObjectData.__name__
    )
    assert isinstance(obj, TestSubObjectData)

    obj = ObjectData(object_class=TestSubObjectData)
    assert isinstance(obj, TestSubObjectData)

    obj = ObjectData(
        object_class=TestSubObject.__module__ + "#" + TestSubObject.__name__
    )
    assert isinstance(obj, TestSubObjectData)

    obj = ObjectData(object_class=TestSubObject)
    assert isinstance(obj, TestSubObjectData)
