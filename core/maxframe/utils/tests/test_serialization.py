# Copyright 1999-2026 Alibaba Group Holding Ltd.
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
import pandas as pd

from ...serialization import Serializer
from ..serialization import (
    deserialize_serializable,
    on_deserialize_shape,
    on_serialize_nsplits,
    on_serialize_numpy_type,
    on_serialize_shape,
    serialize_serializable,
)


def test_on_serialize_shape():
    # Test normal shape
    shape = (10, 20)
    result = on_serialize_shape(shape)
    assert result == (10, 20)

    # Test shape with NaN
    shape = (10, np.nan, 30)
    result = on_serialize_shape(shape)
    assert result == (10, -1, 30)

    # Test shape with numpy scalars
    shape = (np.int32(10), np.int64(20))
    result = on_serialize_shape(shape)
    assert result == (10, 20)

    # Test empty shape
    shape = ()
    result = on_serialize_shape(shape)
    assert result == ()

    # Test None shape
    shape = None
    result = on_serialize_shape(shape)
    assert result is None


def test_on_deserialize_shape():
    # Test normal shape
    shape = (10, 20)
    result = on_deserialize_shape(shape)
    assert result == (10, 20)

    # Test shape with -1 (should become NaN)
    shape = (10, -1, 30)
    result = on_deserialize_shape(shape)
    assert result[0] == 10
    assert pd.isna(result[1])
    assert result[2] == 30

    # Test empty shape
    shape = ()
    result = on_deserialize_shape(shape)
    assert result == ()

    # Test None shape
    shape = None
    result = on_deserialize_shape(shape)
    assert result is None


def test_on_serialize_numpy_type():
    # Test numpy scalar
    value = np.int32(42)
    result = on_serialize_numpy_type(value)
    assert result == 42

    # Test pandas NaT
    value = pd.NaT
    result = on_serialize_numpy_type(value)
    assert result is None

    # Test regular numpy dtype
    value = np.dtype("int32")
    result = on_serialize_numpy_type(value)
    assert result == np.dtype("int32")


def test_on_serialize_nsplits():
    # Test normal nsplits
    nsplits = ((10, 20), (5, 15, 25))
    result = on_serialize_nsplits(nsplits)
    assert result == ((10, 20), (5, 15, 25))

    # Test nsplits with NaN
    nsplits = ((10, np.nan), (5, 15, np.nan))
    result = on_serialize_nsplits(nsplits)
    assert result == ((10, None), (5, 15, None))

    # Test None nsplits
    nsplits = None
    result = on_serialize_nsplits(nsplits)
    assert result is None


def test_serialize_deserialize_with_numpy_scalars():
    """Test serialization and deserialization specifically for numpy scalars"""
    # Test various numpy scalar types
    test_obj = {
        "int8": np.int8(8),
        "int16": np.int16(16),
        "int32": np.int32(32),
        "int64": np.int64(64),
        "uint8": np.uint8(8),
        "uint16": np.uint16(16),
        "uint32": np.uint32(32),
        "uint64": np.uint64(64),
        "float32": np.float32(3.14),
        "float64": np.float64(2.71),
        "bool": np.bool_(True),
    }

    # Serialize the object
    serialized = serialize_serializable(test_obj)

    # Deserialize the object
    deserialized = deserialize_serializable(serialized)

    # Check that numpy scalars were converted to Python natives during serialization
    # and properly handled during deserialization
    for key, value in test_obj.items():
        original_python_value = value.item()  # Convert numpy scalar to Python native
        deserialized_value = deserialized[key]
        assert (
            deserialized_value == original_python_value
        ), f"Mismatch for {key}: expected {original_python_value}, got {deserialized_value}"


def test_serialize_deserialize_with_ext_data():
    """Test serialization and deserialization with with_ext_data=True"""

    class CustomObject:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, CustomObject) and self.value == other.value

    class CustomSerializer(Serializer):
        def serial(self, obj, context):
            return [obj], [], True

        def deserial(self, serialized, context, subs):
            return serialized[0]

    CustomSerializer.register(CustomObject)

    try:
        # Create an object containing custom objects that need external data handling
        custom_obj1 = CustomObject("test1")
        custom_obj2 = CustomObject("test2")

        test_obj = {"data": [custom_obj1, custom_obj2], "values": [1, 2, 3]}

        # Serialize with ext_data support
        serialized, ext_data = serialize_serializable(
            test_obj, with_ext_data=True, allowed_ext_types=CustomObject
        )
        assert len(ext_data) == 2

        # Deserialize providing the external data
        deserialized = deserialize_serializable(serialized, ext_data=ext_data)

        # Check that the original objects are preserved
        assert len(deserialized["data"]) == 2
        assert deserialized["data"][0] == custom_obj1
        assert deserialized["data"][1] == custom_obj2
        assert deserialized["values"] == [1, 2, 3]
    finally:
        CustomSerializer.unregister(CustomObject)


def test_serialize_deserialize_compressed():
    """Test serialization and deserialization with compression enabled"""
    test_obj = {"large_list": list(range(1000)), "text": "A" * 1000}

    # Serialize without compression
    serialized_uncompressed = serialize_serializable(test_obj, compress=False)

    # Serialize with compression
    serialized_compressed = serialize_serializable(test_obj, compress=True)

    # Compressed version should be smaller
    assert len(serialized_compressed) < len(serialized_uncompressed)

    # Decompress the compressed data for proper comparison
    import zlib

    decompressed_data = zlib.decompress(serialized_compressed)

    # Both should deserialize to the same object
    deserialized_uncompressed = deserialize_serializable(serialized_uncompressed)
    deserialized_compressed = deserialize_serializable(decompressed_data)

    assert deserialized_uncompressed == deserialized_compressed
    assert deserialized_uncompressed == test_obj
    assert deserialized_compressed == test_obj
