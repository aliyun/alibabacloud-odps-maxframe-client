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

from collections import namedtuple
from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from ....core import EntityData
from ....lib.wrapped_pickle import switch_unpickle
from ....utils import no_default
from ... import clear_type_cache, deserialize, serialize
from .. import (
    AnyField,
    BoolField,
    BytesField,
    Complex64Field,
    Complex128Field,
    DataFrameField,
    DataTypeField,
    Datetime64Field,
    DictField,
    FieldTypes,
    Float16Field,
    Float32Field,
    Float64Field,
    FunctionField,
    IdentityField,
    IndexField,
    Int8Field,
    Int16Field,
    Int32Field,
    Int64Field,
    IntervalArrayField,
    KeyField,
    ListField,
    NamedTupleField,
    NDArrayField,
    OneOfField,
    ReferenceField,
    Serializable,
    SeriesField,
    SliceField,
    StringField,
    Timedelta64Field,
    TupleField,
    TZInfoField,
    UInt8Field,
    UInt16Field,
    UInt32Field,
    UInt64Field,
)

my_namedtuple = namedtuple("my_namedtuple", "a, b")


@pytest.fixture
def set_is_ci(request):
    from .. import field

    old_is_ci = field._is_ci
    try:
        field._is_ci = request.param
        yield
    finally:
        field._is_ci = old_is_ci


class MyHasKey(EntityData):
    def __init__(self, key=None, **kw):
        super().__init__(_key=key, **kw)
        self._id = "1"

    def __eq__(self, other):
        return isinstance(other, MyHasKey) and other._key == self._key


class MySimpleSerializable(Serializable):
    _id = IdentityField("id")
    _int_val = Int64Field("int_val", default=1000)
    _list_val = ListField("list_val", default_factory=list)
    _ref_val = ReferenceField("ref_val", "MySimpleSerializable")


class MySubSerializable(MySimpleSerializable):
    _m_int_val = Int64Field("m_int_val", default=250)
    _m_str_val = StringField("m_str_val", default="SUB_STR")


class MySerializable(Serializable):
    _id = IdentityField("id")
    _any_val = AnyField("any_val")
    _bool_val = BoolField("bool_val")
    _int8_val = Int8Field("int8_val")
    _int16_val = Int16Field("int16_val")
    _int32_val = Int32Field("int32_val")
    _int64_val = Int64Field("int64_val")
    _uint8_val = UInt8Field("uint8_val")
    _uint16_val = UInt16Field("uint16_val")
    _uint32_val = UInt32Field("uint32_val")
    _uint64_val = UInt64Field("uint64_val")
    _float16_val = Float16Field("float16_val")
    _float32_val = Float32Field(
        "float32_val", on_serialize=lambda x: x + 1, on_deserialize=lambda x: x - 1
    )
    _float64_val = Float64Field("float64_val")
    _complex64_val = Complex64Field("complex64_val")
    _complex128_val = Complex128Field("complex128_val")
    _string_val = StringField("string_val")
    _bytes_val = BytesField("bytes_val")
    _key_val = KeyField("key_val")
    _ndarray_val = NDArrayField("ndarray_val")
    _datetime64_val = Datetime64Field("datetime64_val")
    _timedelta64_val = Timedelta64Field("timedelta64_val")
    _datatype_val = DataTypeField("datatype_val")
    _index_val = IndexField("index_val")
    _series_val = SeriesField("series_val")
    _dataframe_val = DataFrameField("dataframe_val")
    _interval_array_val = IntervalArrayField("interval_array_val")
    _slice_val = SliceField("slice_val")
    _function_val = FunctionField("function_val")
    _named_tuple_val = NamedTupleField("named_tuple_val")
    _tzinfo_val = TZInfoField("tzinfo_val")
    _list_val = ListField("list_val", FieldTypes.int64)
    _tuple_val = TupleField("tuple_val", FieldTypes.string)
    _dict_val = DictField("dict_val", FieldTypes.string, FieldTypes.bytes)
    _ref_val = ReferenceField("ref_val", "self")
    _ref_val2 = ReferenceField("ref_val2", MySimpleSerializable)
    _oneof_val = OneOfField(
        "ref_val",
        oneof1_val=f"{__name__}.MySerializable",
        oneof2_val=MySimpleSerializable,
    )
    _no_default_val = Float64Field("no_default_val", default=no_default)


@pytest.mark.parametrize("set_is_ci", [False, True], indirect=True)
@switch_unpickle(forbidden=False)
def test_serializable(set_is_ci):
    my_serializable = MySerializable(
        _id="1",
        _any_val="any_value",
        _bool_val=True,
        _int8_val=-8,
        _int16_val=np.int16(-16),
        _int32_val=-32,
        _int64_val=-64,
        _uint8_val=8,
        _uint16_val=16,
        _uint32_val=np.uint32(32),
        _uint64_val=64,
        _float16_val=1.0,
        _float32_val=np.float32(2.0),
        _float64_val=2.0,
        _complex64_val=np.complex64(1 + 2j),
        _complex128_val=1 + 2j,
        _string_val="string_value",
        _bytes_val=b"bytes_value",
        _key_val=MyHasKey("aaa"),
        _ndarray_val=np.random.rand(4, 3),
        _datetime64_val=pd.Timestamp(123),
        _timedelta64_val=pd.Timedelta(
            days=1, seconds=123, microseconds=345, nanoseconds=132
        ),
        _datatype_val=np.dtype(np.int32),
        _index_val=pd.Index([1, 2]),
        _series_val=pd.Series(["a", "b"]),
        _dataframe_val=pd.DataFrame({"a": [1, 2, 3]}),
        _interval_array_val=pd.arrays.IntervalArray([]),
        _slice_val=slice(1, 10, 2),
        _function_val=lambda x: x + 1,
        _named_tuple_val=my_namedtuple(a=1, b=2),
        _tzinfo_val=timezone.utc,
        _list_val=[1, 2],
        _tuple_val=("a", "b"),
        _dict_val={"a": b"bytes_value"},
        _ref_val=MySerializable(),
        _oneof_val=MySerializable(_id="2"),
        _no_default_val=no_default,
    )

    header, buffers = serialize(my_serializable)
    my_serializable2 = deserialize(header, buffers)
    _assert_serializable_eq(my_serializable, my_serializable2)


@pytest.mark.parametrize("set_is_ci", [False, True], indirect=True)
@switch_unpickle
def test_compatible_serializable(set_is_ci):
    global MySimpleSerializable, MySubSerializable

    clear_type_cache()
    old_base, old_sub = MySimpleSerializable, MySubSerializable

    try:
        my_sub_serializable = MySubSerializable(
            _id="id_val",
            _list_val=["abcd", "wxyz"],
            _ref_val=MyHasKey(),
            _m_int_val=3412,
            _m_str_val="dfgghj",
        )
        header, buffers = serialize(my_sub_serializable)

        class MySimpleSerializable(Serializable):
            _id = IdentityField("id")
            _int_val = Int64Field("int_val", default=1000)
            _list_val = ListField("list_val", default_factory=list)
            _ref_val = ReferenceField("ref_val", "MySimpleSerializable")
            _dict_val = DictField("dict_val")

        class MyMidSerializable(MySimpleSerializable):
            _i_bool_val = Int64Field("i_bool_val", default=True)

        class MySubSerializable(MyMidSerializable):
            _m_int_val = Int64Field("m_int_val", default=250)
            _m_str_val = StringField("m_str_val", default="SUB_STR")

        my_sub_serializable2 = deserialize(header, buffers)
        assert type(my_sub_serializable) is not type(my_sub_serializable2)
        _assert_serializable_eq(my_sub_serializable, my_sub_serializable2)

        header, buffers = serialize(my_sub_serializable2)
    finally:
        MySimpleSerializable, MySubSerializable = old_base, old_sub
        MyMidSerializable = None
        clear_type_cache()

    my_sub_serializable3 = deserialize(header, buffers)
    assert type(my_sub_serializable2) is not type(my_sub_serializable3)
    _assert_serializable_eq(my_sub_serializable2, my_sub_serializable3)


def _assert_serializable_eq(my_serializable, my_serializable2):
    for field_name, field in my_serializable._FIELDS.items():
        if not hasattr(my_serializable, field.name) or not hasattr(
            my_serializable2, field.name
        ):
            continue
        expect_value = getattr(my_serializable, field_name)
        if expect_value is no_default:
            assert not hasattr(my_serializable2, field.name)
            continue
        else:
            actual_value = getattr(my_serializable2, field_name)
        if isinstance(expect_value, np.ndarray):
            np.testing.assert_array_equal(expect_value, actual_value)
        elif isinstance(expect_value, pd.DataFrame):
            pd.testing.assert_frame_equal(expect_value, actual_value)
        elif isinstance(expect_value, pd.Series):
            pd.testing.assert_series_equal(expect_value, actual_value)
        elif isinstance(expect_value, pd.Index):
            pd.testing.assert_index_equal(expect_value, actual_value)
        elif isinstance(expect_value, pd.api.extensions.ExtensionArray):
            pd.testing.assert_extension_array_equal(expect_value, actual_value)
        elif isinstance(expect_value, (MySimpleSerializable, MySerializable)):
            _assert_serializable_eq(expect_value, actual_value)
        elif callable(expect_value):
            assert expect_value(1) == actual_value(1)
        else:
            assert expect_value == actual_value, f"Field {field_name}"


@pytest.mark.parametrize("set_is_ci", [True], indirect=True)
def test_fields_errors(set_is_ci):
    my_simple = MySimpleSerializable(_id="1", _ref_val=MySimpleSerializable(_id="2"))
    my_serializeble = MySerializable(_oneof_val=my_simple)

    with pytest.raises(TypeError) as exc_info:
        my_simple._int_val = "10"
    assert "_int_val" in str(exc_info.value)

    del my_simple._ref_val
    with pytest.raises(AttributeError):
        _ = my_simple._ref_val

    del my_simple._id
    with pytest.raises(AttributeError):
        _ = my_simple._id

    assert my_simple._int_val == 1000
    assert my_simple._list_val == []

    del my_serializeble._oneof_val
    with pytest.raises(AttributeError):
        _ = my_serializeble._oneof_val

    my_serializeble._ref_val2 = MySimpleSerializable(_id="3")
    del my_serializeble._ref_val2
    with pytest.raises(AttributeError):
        _ = my_serializeble._ref_val2

    with pytest.raises(TypeError):
        my_serializeble._ref_val = my_simple

    with pytest.raises(TypeError):
        my_serializeble._oneof_val = 1

    with pytest.raises(AttributeError):
        del my_serializeble._oneof_val
