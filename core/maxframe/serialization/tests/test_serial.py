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

import datetime
import re
import threading
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, NamedTuple

import numpy as np
import pandas as pd
import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None
try:
    import scipy.sparse as sps
except ImportError:
    sps = None
try:
    import pytz
except ImportError:
    pytz = None
try:
    import zoneinfo
except ImportError:
    zoneinfo = None

from ...lib.sparse import SparseMatrix
from ...lib.wrapped_pickle import switch_unpickle
from ...tests.utils import require_cudf, require_cupy
from ...utils import lazy_import
from .. import (
    PickleContainer,
    RemoteException,
    deserialize,
    serialize,
    serialize_with_spawn,
)
from ..core import ListSerializer, Placeholder

cupy = lazy_import("cupy")
cudf = lazy_import("cudf")


class CustomList(list):
    pass


class CustomNamedTuple(NamedTuple):
    name: str
    idx: int


@pytest.mark.parametrize(
    "val",
    [
        None,
        False,
        123,
        3.567,
        3.5 + 4.3j,
        b"abcd",
        "abcd",
        slice(3, 9, 2),
        ["uvw", ("mno", "sdaf"), 4, 6.7],
        CustomNamedTuple("abcd", 13451),
        datetime.datetime.now(),
        datetime.datetime.now().astimezone(datetime.timezone.utc),
        datetime.date.today(),
        datetime.timedelta(1000, 10, 100, 10),
        re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", re.M | re.DOTALL),
        np.dtype(int),
        pd.StringDtype(),
        pd.Timestamp("2023-10-12 11:22:54.134561231"),
        pd.Timestamp("2023-10-12 11:22:54.134561231", tzinfo=datetime.timezone.utc),
        pd.Timedelta(102.234154131),
        {"abc": 5.6, "def": [3.4], "gh": None, "ijk": {}},
        OrderedDict([("abcd", 5.6)]),
    ],
)
@switch_unpickle
def test_core(val):
    deserialized = deserialize(*serialize(val))
    assert type(val) == type(deserialized)
    assert val == deserialized


@switch_unpickle
def test_strings():
    str_obj = "abcd" * 1024
    obj = [str_obj, str_obj]
    header, bufs = serialize(obj)
    assert len(header) < len(str_obj) * 2
    bufs = [memoryview(buf) for buf in bufs]
    assert obj == deserialize(header, bufs)


@switch_unpickle
def test_placeholder_obj():
    assert Placeholder(1024) == Placeholder(1024)
    assert hash(Placeholder(1024)) == hash(Placeholder(1024))
    assert Placeholder(1024) != Placeholder(1023)
    assert hash(Placeholder(1024)) != hash(Placeholder(1023))
    assert Placeholder(1024) != 1024
    assert "1024" in repr(Placeholder(1024))


@switch_unpickle
def test_nested_list():
    val = [b"a" * 1200] * 10
    val[0] = val
    deserialized = deserialize(*serialize(val))
    assert deserialized[0] is deserialized
    assert val[1:] == deserialized[1:]


@switch_unpickle
def test_nested_dict():
    val = {i: "b" * 100 for i in range(10)}
    val[0] = val
    deserialized = deserialize(*serialize(val))
    assert deserialized[0] is deserialized


@pytest.mark.parametrize(
    "val",
    [
        pytz.timezone("Europe/Berlin") if pytz else None,
        zoneinfo.ZoneInfo("America/New_York") if zoneinfo else None,
        datetime.timezone(datetime.timedelta(hours=7)),
        datetime.timezone(datetime.timedelta(hours=6), name="UTC+6"),
        datetime.timezone.utc,
    ],
)
@switch_unpickle
def test_timezones(val):
    if val is None:
        pytest.skip("Skip due to lack of library")
    deserialized = deserialize(*serialize(val))
    assert type(val) == type(deserialized)
    assert val == deserialized


@pytest.mark.parametrize(
    "val",
    [
        np.array([1024])[0],
        np.array(np.random.rand(100, 100)),
        np.array(np.random.rand(100, 100).T),
        np.array(["a", "bcd", None]),
    ],
)
@switch_unpickle
def test_numpy(val):
    deserialized = deserialize(*serialize(val))
    assert type(val) == type(deserialized)
    np.testing.assert_equal(val, deserialized)
    if val.flags.f_contiguous:
        assert deserialized.flags.f_contiguous


@switch_unpickle
def test_pandas():
    val = pd.Series([1, 2, 3, 4], name="nm")
    pd.testing.assert_series_equal(val, deserialize(*serialize(val)))

    val = pd.DataFrame(
        {
            "float_col": np.random.rand(1000),
            "str_col": np.random.choice(list("abcd"), size=(1000,)),
            "int_col": np.random.randint(0, 100, size=(1000,)),
            "str_array_col": pd.array(np.random.choice(list("abcd"), size=(1000,))),
            "cat_col": pd.Categorical(np.random.choice(list("abcd"), size=(1000,))),
        }
    )
    if pa is not None and hasattr(pd, "ArrowDtype"):
        val["arrow_col"] = pd.Series(
            np.random.rand(1000), dtype=pd.ArrowDtype(pa.float64())
        )
    pd.testing.assert_frame_equal(val, deserialize(*serialize(val)))

    val = pd.MultiIndex.from_arrays(
        [(1, 5, 4, 9, 6), list("BADCE")], names=["C1", "C2"]
    )
    pd.testing.assert_index_equal(val, deserialize(*serialize(val)))

    val = pd.CategoricalIndex(np.random.choice(list("abcd"), size=(1000,)))
    pd.testing.assert_index_equal(val, deserialize(*serialize(val)))


@pytest.mark.skipif(pa is None, reason="need pyarrow to run the cases")
@switch_unpickle
def test_arrow():
    test_array = np.random.rand(1000)
    test_df = pd.DataFrame(
        {
            "a": np.random.rand(1000),
            "b": np.random.choice(list("abcd"), size=(1000,)),
            "c": np.random.randint(0, 100, size=(1000,)),
        }
    )
    test_vals = [
        pa.array(test_array),
        pa.chunked_array([pa.array(test_array), pa.array(test_array)]),
        pa.RecordBatch.from_pandas(test_df),
        pa.Table.from_pandas(test_df),
    ]
    for val in test_vals:
        deserialized = deserialize(*serialize(val))
        assert type(val) is type(deserialized)
        np.testing.assert_equal(val, deserialized)


@pytest.mark.parametrize(
    "np_val",
    [np.random.rand(100, 100), np.random.rand(100, 100).T],
)
@require_cupy
def test_cupy(np_val):
    val = cupy.array(np_val)
    deserialized = deserialize(*serialize(val))
    assert type(val) is type(deserialized)
    cupy.testing.assert_array_equal(val, deserialized)


@require_cudf
def test_cudf():
    raw_df = pd.DataFrame(
        {
            "a": np.random.rand(1000),
            "b": np.random.choice(list("abcd"), size=(1000,)),
            "c": np.random.randint(0, 100, size=(1000,)),
        }
    )
    test_df = cudf.DataFrame(raw_df)
    cudf.testing.assert_frame_equal(test_df, deserialize(*serialize(test_df)))

    raw_df.columns = pd.MultiIndex.from_tuples([("a", "a"), ("a", "b"), ("b", "c")])
    test_df = cudf.DataFrame(raw_df)
    cudf.testing.assert_frame_equal(test_df, deserialize(*serialize(test_df)))


@pytest.mark.skipif(sps is None, reason="need scipy to run the test")
def test_scipy_sparse():
    val = sps.random(100, 100, 0.1, format="csr")
    deserial = deserialize(*serialize(val))
    assert (val != deserial).nnz == 0


@pytest.mark.skipif(sps is None, reason="need scipy to run the test")
def test_maxframe_sparse():
    val = SparseMatrix(sps.random(100, 100, 0.1, format="csr"))
    deserial = deserialize(*serialize(val))
    assert (val.spmatrix != deserial.spmatrix).nnz == 0


def test_pickle_container():
    def func_to_pk():
        return 1234

    deserial = deserialize(*serialize(func_to_pk))
    assert func_to_pk() == deserial()

    with switch_unpickle(forbidden=True):
        deserial = deserialize(*serialize(func_to_pk))
        assert isinstance(deserial, PickleContainer)

    with switch_unpickle(forbidden=False):
        deserial_val = deserial.get()
        assert deserial_val() == func_to_pk()

        deserial2 = deserialize(*serialize(deserial))
        assert deserial2() == func_to_pk()


def test_exceptions():
    try:
        raise ValueError("val")
    except BaseException as ex:
        exc = ex

    deserial = deserialize(*serialize(exc))
    assert isinstance(deserial, ValueError)
    assert deserial.args[0] == exc.args[0]

    with switch_unpickle(forbidden=True):
        deserial = deserialize(*serialize(exc))
        assert isinstance(deserial, RemoteException)

    with switch_unpickle(forbidden=False):
        deserial_val = deserial.get()
        assert isinstance(deserial_val, ValueError)
        assert deserial_val.args[0] == exc.args[0]

        deserial2 = deserialize(*serialize(deserial))
        assert isinstance(deserial2, ValueError)
        assert deserial2.args[0] == exc.args[0]


class MockSerializerForErrors(ListSerializer):
    serializer_id = 25951
    raises = False

    def on_deserial_error(
        self,
        serialized: List,
        context: Dict,
        subs_serialized: List,
        error_index: int,
        exc: BaseException,
    ):
        assert error_index == 1
        assert subs_serialized[error_index]
        try:
            raise SystemError from exc
        except BaseException as ex:
            return ex

    def deserial(self, serialized: List, context: Dict, subs: List[Any]):
        if len(subs) == 2 and self.raises:
            raise TypeError
        return super().deserial(serialized, context, subs)


class UnpickleWithError:
    def __getstate__(self):
        return (None,)

    def __setstate__(self, state):
        raise ValueError


def test_deserial_errors():
    try:
        MockSerializerForErrors.raises = False
        MockSerializerForErrors.register(CustomList)
        ListSerializer.register(CustomList, name="test_name")

        # error of leaf object is raised
        obj = [1, [[3, UnpickleWithError()]]]
        with pytest.raises(ValueError):
            deserialize(*serialize(obj))

        # error of leaf object is rewritten in parent object
        obj = CustomList([[1], [[3, UnpickleWithError()]]])
        with pytest.raises(SystemError) as exc_info:
            deserialize(*serialize(obj))
        assert isinstance(exc_info.value.__cause__, ValueError)

        MockSerializerForErrors.raises = True

        # error of non-leaf object is raised
        obj = [CustomList([[1], [[2]]])]
        with pytest.raises(TypeError):
            deserialize(*serialize(obj))
        deserialize(*serialize(obj, {"serializer": "test_name"}))

        # error of non-leaf CustomList is rewritten in parent object
        obj = CustomList([[1], CustomList([[1], [[2]]]), [2]])
        with pytest.raises(SystemError) as exc_info:
            deserialize(*serialize(obj))
        assert isinstance(exc_info.value.__cause__, TypeError)
        deserialize(*serialize(obj, {"serializer": "test_name"}))
    finally:
        MockSerializerForErrors.unregister(CustomList)
        ListSerializer.unregister(CustomList, name="test_name")
        # Above unregister will remove the ListSerializer from deserializers,
        # so we need to register ListSerializer again to make the
        # deserializers correct.
        ListSerializer.register(list)


class MockSerializerForSpawn(ListSerializer):
    thread_calls = defaultdict(lambda: 0)

    def serial(self, obj: Any, context: Dict):
        self.thread_calls[threading.current_thread().ident] += 1
        return super().serial(obj, context)


@pytest.mark.asyncio
async def test_spawn_threshold():
    try:
        assert 0 == deserialize(*(await serialize_with_spawn(0)))

        MockSerializerForSpawn.register(CustomList)
        obj = [CustomList([i]) for i in range(200)]
        serialized = await serialize_with_spawn(obj, spawn_threshold=100)
        assert serialized[0][0]["_N"] == 201
        deserialized = deserialize(*serialized)
        for s, d in zip(obj, deserialized):
            assert s[0] == d[0]

        calls = MockSerializerForSpawn.thread_calls
        assert sum(calls.values()) == 200
        assert calls[threading.current_thread().ident] == 101
    finally:
        MockSerializerForSpawn.unregister(CustomList)
