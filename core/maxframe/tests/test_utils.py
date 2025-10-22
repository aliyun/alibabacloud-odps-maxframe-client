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

import asyncio
import copy
import functools
import os
import shutil
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from .. import utils
from ..lib.dtypes_extension import ArrowDtype
from ..serialization import PickleContainer
from ..utils import parse_size_to_megabytes, validate_and_adjust_resource_ratio


def test_string_conversion():
    s = None
    assert utils.to_binary(s) is None
    assert utils.to_str(s) is None
    assert utils.to_text(s) is None

    s = "abcdefg"
    assert isinstance(utils.to_binary(s), bytes)
    assert utils.to_binary(s) == b"abcdefg"
    assert isinstance(utils.to_str(s), str)
    assert utils.to_str(s) == "abcdefg"
    assert isinstance(utils.to_text(s), str)
    assert utils.to_text(s) == "abcdefg"

    ustr = type("ustr", (str,), {})
    assert isinstance(utils.to_str(ustr(s)), str)
    assert utils.to_str(ustr(s)) == "abcdefg"

    s = b"abcdefg"
    assert isinstance(utils.to_binary(s), bytes)
    assert utils.to_binary(s) == b"abcdefg"
    assert isinstance(utils.to_str(s), str)
    assert utils.to_str(s) == "abcdefg"
    assert isinstance(utils.to_text(s), str)
    assert utils.to_text(s) == "abcdefg"

    ubytes = type("ubytes", (bytes,), {})
    assert isinstance(utils.to_binary(ubytes(s)), bytes)
    assert utils.to_binary(ubytes(s)) == b"abcdefg"

    s = "abcdefg"
    assert isinstance(utils.to_binary(s), bytes)
    assert utils.to_binary(s) == b"abcdefg"
    assert isinstance(utils.to_str(s), str)
    assert utils.to_str(s) == "abcdefg"
    assert isinstance(utils.to_text(s), str)
    assert utils.to_text(s) == "abcdefg"

    uunicode = type("uunicode", (str,), {})
    assert isinstance(utils.to_text(uunicode(s)), str)
    assert utils.to_text(uunicode(s)) == "abcdefg"

    with pytest.raises(TypeError):
        utils.to_binary(utils)
    with pytest.raises(TypeError):
        utils.to_str(utils)
    with pytest.raises(TypeError):
        utils.to_text(utils)


def test_tokenize():
    import shutil
    import tempfile

    class TestEnum(Enum):
        VAL1 = "val1"

    tempdir = tempfile.mkdtemp("maxframe_test_utils_")
    try:
        filename = os.path.join(tempdir, "test_npa.dat")
        mmp_array = np.memmap(filename, dtype=float, mode="w+", shape=(3, 4))
        mmp_array[:] = np.random.random((3, 4)).astype(float)
        mmp_array.flush()
        del mmp_array

        mmp_array1 = np.memmap(filename, dtype=float, shape=(3, 4))
        mmp_array2 = np.memmap(filename, dtype=float, shape=(3, 4))

        try:
            v = [
                1,
                2.3,
                "456",
                "789",
                b"101112",
                2147483649,
                None,
                np.ndarray,
                [912, "uvw"],
                np.arange(0, 10),
                np.array(10),
                np.array([b"\x01\x32\xff"]),
                np.int64,
                TestEnum.VAL1,
            ]
            copy_v = copy.deepcopy(v)
            assert utils.tokenize(v + [mmp_array1], ext_data=1234) == utils.tokenize(
                copy_v + [mmp_array2], ext_data=1234
            )
        finally:
            del mmp_array1, mmp_array2
    finally:
        shutil.rmtree(tempdir)

    v = {"a", "xyz", "uvw"}
    assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    v = dict(x="abcd", y=98765)
    assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    v = dict(x=dict(a=1, b=[1, 2, 3]), y=12345)
    assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    # pandas relative
    if pd is not None:
        df = pd.DataFrame(
            [[utils.to_binary("测试"), utils.to_text("数据")]],
            index=["a"],
            columns=["中文", "data"],
        )
        v = [df, df.index, df.columns, df["data"], pd.Categorical(list("ABCD"))]
        assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    class NonTokenizableCls:
        def __getstate__(self):
            raise SystemError

    with pytest.raises(TypeError):
        utils.tokenize(NonTokenizableCls())

    class CustomizedTokenize(object):
        def __maxframe_tokenize__(self):
            return id(type(self)), id(NonTokenizableCls)

    assert utils.tokenize(CustomizedTokenize()) == utils.tokenize(CustomizedTokenize())

    v = lambda x: x + 1
    assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    def f(a, b):
        return np.add(a, b)

    assert utils.tokenize(f) == utils.tokenize(copy.deepcopy(f))

    partial_f = partial(f, 1, k=0)
    partial_f2 = partial(f, 1, k=1)
    assert utils.tokenize(partial_f) == utils.tokenize(copy.deepcopy(partial_f))
    assert utils.tokenize(partial_f) != utils.tokenize(partial_f2)

    container = PickleContainer([b"abcd", b"efgh"])
    assert utils.tokenize(container) == utils.tokenize(copy.deepcopy(container))


def test_lazy_import():
    old_sys_path = sys.path
    mock_mod = textwrap.dedent(
        """
        __version__ = '0.1.0b1'
        """.strip()
    )
    mock_mod2 = textwrap.dedent(
        """
        from maxframe.utils import lazy_import
        mock_mod = lazy_import("mock_mod")

        def get_version():
            return mock_mod.__version__
        """
    )

    temp_dir = tempfile.mkdtemp(prefix="maxframe-utils-test-")
    sys.path += [temp_dir]
    try:
        with open(os.path.join(temp_dir, "mock_mod.py"), "w") as outf:
            outf.write(mock_mod)
        with open(os.path.join(temp_dir, "mock_mod2.py"), "w") as outf:
            outf.write(mock_mod2)

        non_exist_mod = utils.lazy_import("non_exist_mod", locals=locals())
        assert non_exist_mod is None

        non_exist_mod1 = utils.lazy_import("non_exist_mod1", placeholder=True)
        with pytest.raises(AttributeError) as ex_data:
            non_exist_mod1.meth()
        assert "required" in str(ex_data.value)

        mod = utils.lazy_import(
            "mock_mod", globals=globals(), locals=locals(), rename="mod"
        )
        assert mod is not None
        assert mod.__version__ == "0.1.0b1"

        glob = globals().copy()
        mod = utils.lazy_import("mock_mod", globals=glob, locals=locals(), rename="mod")
        glob["mod"] = mod
        assert mod is not None
        assert mod.__version__ == "0.1.0b1"
        assert type(glob["mod"]).__name__ == "module"

        import mock_mod2 as mod2

        assert type(mod2.mock_mod).__name__ != "module"
        assert mod2.get_version() == "0.1.0b1"
        assert type(mod2.mock_mod).__name__ == "module"
    finally:
        shutil.rmtree(temp_dir)
        sys.path = old_sys_path
        sys.modules.pop("mock_mod", None)
        sys.modules.pop("mock_mod2", None)


def test_type_dispatcher():
    dispatcher = utils.TypeDispatcher()

    type1 = type("Type1", (), {})
    type2 = type("Type2", (type1,), {})
    type3 = type("Type3", (), {})
    type4 = type("Type4", (type2,), {})
    type5 = type("Type5", (type4,), {})

    dispatcher.register(object, lambda x: "Object")
    dispatcher.register(type1, lambda x: "Type1")
    dispatcher.register(type4, lambda x: "Type4")
    dispatcher.register("pandas.DataFrame", lambda x: "DataFrame")
    dispatcher.register(utils.NamedType("ray", type1), lambda x: "RayType1")

    assert "Type1" == dispatcher(type2())
    assert "DataFrame" == dispatcher(pd.DataFrame())
    assert "Object" == dispatcher(type3())

    tp = utils.NamedType("ray", type1)
    assert dispatcher.get_handler(tp)(tp) == "RayType1"
    tp = utils.NamedType("ray", type2)
    assert dispatcher.get_handler(tp)(tp) == "RayType1"
    tp = utils.NamedType("xxx", type2)
    assert dispatcher.get_handler(tp)(tp) == "Type1"
    assert "Type1" == dispatcher(type2())
    tp = utils.NamedType("ray", type5)
    assert dispatcher.get_handler(tp)(tp) == "Type4"

    dispatcher.unregister(object)
    with pytest.raises(KeyError):
        dispatcher(type3())


def test_timer():
    with utils.Timer() as timer:
        time.sleep(0.1)

    assert timer.duration >= 0.1


def test_module_placeholder():
    required_module = utils.ModulePlaceholder("required_module")

    with pytest.raises(AttributeError):
        required_module()
    with pytest.raises(AttributeError) as e:
        required_module.method()
    msg = e.value.args[0]
    assert msg == "required_module is required but not installed."


def test_estimate_pandas_size():
    df1 = pd.DataFrame(np.random.rand(50, 10))
    assert utils.estimate_pandas_size(df1) == sys.getsizeof(df1)

    df2 = pd.DataFrame(np.random.rand(1000, 10))
    assert utils.estimate_pandas_size(df2) == sys.getsizeof(df2)

    s1 = pd.Series(np.random.rand(1000))
    assert utils.estimate_pandas_size(s1) == sys.getsizeof(s1)

    if hasattr(pd, "ArrowDtype"):
        arrow_array = pa.array(np.random.choice(["abcd", "def", "gh"], size=(1000,)))
        array = pd.array(arrow_array, dtype=ArrowDtype(arrow_array.type))
        s2 = pd.Series(array)
        assert utils.estimate_pandas_size(s2) == sys.getsizeof(s2)

    s3 = pd.Series(np.random.choice(["abcd", "def", "gh"], size=(1000,)))
    assert (
        pytest.approx(utils.estimate_pandas_size(s3) / sys.getsizeof(s3), abs=0.5) == 1
    )

    idx1 = pd.MultiIndex.from_arrays(
        [np.arange(0, 1000), np.random.choice(["abcd", "def", "gh"], size=(1000,))]
    )
    assert utils.estimate_pandas_size(idx1) == sys.getsizeof(idx1)

    string_idx = pd.Index(np.random.choice(["a", "bb", "cc"], size=(1000,)))
    assert (
        pytest.approx(
            utils.estimate_pandas_size(string_idx) / sys.getsizeof(string_idx), abs=0.5
        )
        == 1
    )

    # dataframe with multi index
    idx2 = pd.MultiIndex.from_arrays(
        [np.arange(0, 1000), np.random.choice(["abcd", "def", "gh"], size=(1000,))]
    )
    df4 = pd.DataFrame(
        {
            "A": np.random.choice(["abcd", "def", "gh"], size=(1000,)),
            "B": np.random.rand(1000),
            "C": np.random.rand(1000),
        },
        index=idx2,
    )
    assert (
        pytest.approx(utils.estimate_pandas_size(df4) / sys.getsizeof(df4), abs=0.5)
        == 1
    )

    # series with multi index
    idx3 = pd.MultiIndex.from_arrays(
        [
            np.random.choice(["a1", "a2", "a3"], size=(1000,)),
            np.random.choice(["abcd", "def", "gh"], size=(1000,)),
        ]
    )
    s4 = pd.Series(np.arange(1000), index=idx3)

    assert utils.estimate_pandas_size(s4) == sys.getsizeof(s4)


@pytest.mark.parametrize("id_length", [0, 5, 32, 63])
def test_gen_random_id(id_length):
    rnd_id = utils.new_random_id(id_length)
    assert len(rnd_id) == id_length


def test_arrow_type_from_string():
    def _assert_arrow_type_convert(tp: pa.DataType) -> None:
        assert utils.arrow_type_from_str(str(tp)) == tp

    _assert_arrow_type_convert(pa.int64())
    _assert_arrow_type_convert(pa.binary(20))
    _assert_arrow_type_convert(pa.decimal128(10, 2))
    _assert_arrow_type_convert(pa.list_(pa.int64()))
    _assert_arrow_type_convert(pa.map_(pa.string(), pa.int64()))
    _assert_arrow_type_convert(pa.date32())
    _assert_arrow_type_convert(pa.date64())
    _assert_arrow_type_convert(
        pa.struct([("key", pa.string()), ("value", pa.list_(pa.int64()))])
    )
    _assert_arrow_type_convert(
        pa.struct([("key", pa.string(), False), ("value", pa.list_(pa.int64()))])
    )
    _assert_arrow_type_convert(
        pa.struct([("key", pa.string()), ("value", pa.list_(pa.int64()), False)])
    )


@pytest.mark.parametrize("use_async", [False, True])
async def test_call_with_retry(use_async):
    retry_idx_list = [0]

    def sync_func(delay=0):
        if delay:
            time.sleep(delay)
        if retry_idx_list[0] < 3:
            retry_idx_list[0] += 1
            raise ValueError

    async def async_func(delay=0):
        if delay:
            await asyncio.sleep(delay)
        if retry_idx_list[0] < 3:
            retry_idx_list[0] += 1
            raise ValueError

    func = async_func if use_async else sync_func

    async def wait_coro(res):
        if asyncio.iscoroutine(res):
            return await res
        return res

    # test cases for retry times
    with pytest.raises(ValueError):
        retry_idx_list[0] = 0
        await wait_coro(
            utils.call_with_retry(func, retry_times=1, exc_type=(TypeError, ValueError))
        )
    assert retry_idx_list[0] == 2

    retry_idx_list[0] = 0
    await wait_coro(
        utils.call_with_retry(func, retry_times=3, exc_type=(TypeError, ValueError))
    )
    assert retry_idx_list[0] == 3

    retry_idx_list[0] = 0
    exc_info = await wait_coro(
        utils.call_with_retry(
            func, retry_times=1, exc_type=(TypeError, ValueError), no_raise=True
        )
    )
    assert isinstance(exc_info[1], ValueError)
    assert retry_idx_list[0] == 2

    delay_func = functools.partial(func, delay=0.5)
    with pytest.raises(ValueError):
        retry_idx_list[0] = 0
        await wait_coro(
            utils.call_with_retry(delay_func, retry_times=None, retry_timeout=0.7)
        )
    assert retry_idx_list[0] == 2

    retry_idx_list[0] = 0
    await wait_coro(
        utils.call_with_retry(delay_func, retry_times=None, retry_timeout=2.2)
    )
    assert retry_idx_list[0] == 3

    retry_idx_list[0] = 0
    exc_info = await wait_coro(
        utils.call_with_retry(
            delay_func, retry_times=None, retry_timeout=0.7, no_raise=True
        )
    )
    assert isinstance(exc_info[1], ValueError)
    assert retry_idx_list[0] == 2


def test_debug_to_thread():
    class MixinTestCls(utils.ToThreadMixin):
        async def run_in_coro(self):
            await self.to_thread(time.sleep, 0.5)
            await self.to_thread(functools.partial(time.sleep), 0.5)

    def thread_body():
        loop = asyncio.new_event_loop()
        loop.set_debug(True)
        asyncio.set_event_loop(loop)
        loop.run_until_complete(MixinTestCls().run_in_coro())

    tpe = ThreadPoolExecutor(max_workers=1)
    tpe.submit(thread_body).result()
    tpe.shutdown()


@pytest.mark.parametrize(
    "val, expected, make_series",
    [
        (int, np.dtype(int), True),
        ([int], [np.dtype(int)], False),
        ([int], pd.Series([np.dtype(int)]), True),
        (np.dtype("float64"), np.dtype("float64"), True),
        ("category", "category", True),
        ("string", "string", True),
        (pd.Timestamp, np.dtype("datetime64[ns]"), True),
        (pd.Timedelta, np.dtype("timedelta64[ns]"), True),
        ({"col": float}, {"col": np.dtype(float)}, False),
        ({"col": float}, pd.Series({"col": np.dtype(float)}), True),
        (
            pd.Series([float], index=["col"]),
            pd.Series([np.dtype(float)], index=["col"]),
            True,
        ),
    ],
)
def test_make_dtypes(val, expected, make_series):
    result = utils.make_dtypes(val, make_series=make_series)
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(result, expected)
    else:
        assert result == expected


# Define conversion constants
BYTES_PER_KIB = 1024
BYTES_PER_MIB = 1024**2
BYTES_PER_GIB = 1024**3
BYTES_PER_TIB = 1024**4

BYTES_PER_KB = 1000
BYTES_PER_MB = 1000**2
BYTES_PER_GB = 1000**3
BYTES_PER_TB = 1000**4


# Test numeric values with default units
@pytest.mark.parametrize("value", [0, 1, 4, 5, 0.01, 0.1, 0.5, 1.5])
@pytest.mark.parametrize(
    "default_unit", ["KiB", "MiB", "GiB", "TiB", "KB", "MB", "GB", "TB"]
)
def test_numeric_inputs_with_default_units(value, default_unit):
    """Test numeric inputs with various default units"""
    # Define unit conversions to MiB for reusability
    unit_to_mib_factor = {
        "KiB": BYTES_PER_KIB / BYTES_PER_MIB,
        "MiB": 1,
        "GiB": BYTES_PER_GIB / BYTES_PER_MIB,
        "TiB": BYTES_PER_TIB / BYTES_PER_MIB,
        "KB": BYTES_PER_KB / BYTES_PER_MIB,
        "MB": BYTES_PER_MB / BYTES_PER_MIB,
        "GB": BYTES_PER_GB / BYTES_PER_MIB,
        "TB": BYTES_PER_TB / BYTES_PER_MIB,
    }
    expected = value * unit_to_mib_factor[default_unit]
    result = parse_size_to_megabytes(value, default_number_unit=default_unit)
    assert pytest.approx(result) == expected


@pytest.mark.parametrize(
    "input_string, expected",
    [
        # Basic binary units
        ("1B", 1 / BYTES_PER_MIB),
        ("1KiB", BYTES_PER_KIB / BYTES_PER_MIB),
        ("5miB", 5),
        ("2giB", 2 * BYTES_PER_GIB / BYTES_PER_MIB),
        ("0.1TiB", 0.1 * BYTES_PER_TIB / BYTES_PER_MIB),
        # Basic decimal units
        ("1KB", BYTES_PER_KB / BYTES_PER_MIB),
        ("10MB", 10 * BYTES_PER_MB / BYTES_PER_MIB),
        ("0.5GB", 0.5 * BYTES_PER_GB / BYTES_PER_MIB),
        ("0.01TB", 0.01 * BYTES_PER_TB / BYTES_PER_MIB),
        # Abbreviated forms
        ("1K", BYTES_PER_KB / BYTES_PER_MIB),
        ("10M", 10 * BYTES_PER_MB / BYTES_PER_MIB),
        ("0.5g", 0.5 * BYTES_PER_GB / BYTES_PER_MIB),
        ("0.01T", 0.01 * BYTES_PER_TB / BYTES_PER_MIB),
        # With spaces
        ("1 kiB", BYTES_PER_KIB / BYTES_PER_MIB),
        ("10 MB", 10 * BYTES_PER_MB / BYTES_PER_MIB),
        (" 0.5 GiB ", 0.5 * BYTES_PER_GIB / BYTES_PER_MIB),
        ("0.01  TB", 0.01 * BYTES_PER_TB / BYTES_PER_MIB),
    ],
)
def test_string_values_with_units(input_string, expected):
    """Test various string inputs with different units"""
    result = parse_size_to_megabytes(input_string, default_number_unit="GiB")
    assert pytest.approx(result) == expected


@pytest.mark.parametrize(
    "invalid_input, default_unit",
    [
        ("invalid", "GiB"),  # Non-numeric input
        ("1.2.3GiB", "GiB"),  # Invalid number format
        ("1ZiB", "GiB"),  # Invalid unit
        ("GiB", "GiB"),  # Missing number
        ("1G1B", "GiB"),  # Invalid format
        (5, None),  # Numeric input with default_number_unit as None
        ("5", None),  # String numeric input with default_number_unit as None
        ("5", "Gibb"),  # String numeric input with bad default_number_unit
    ],
)
def test_parse_size_to_mega_bytes_invalid_inputs(invalid_input, default_unit):
    """Test invalid inputs that should raise ValueError"""
    with pytest.raises(ValueError):  # Catch ValueError
        parse_size_to_megabytes(invalid_input, default_number_unit=default_unit)


@pytest.mark.parametrize(
    "udf_resources, max_memory_cpu_ratio, adjust, expected_resources, expected_adjusted, should_warn",
    [
        ({"other": "value"}, 4, False, {"other": "value"}, False, False),
        ({"memory": 8}, 4, False, {"memory": 8}, False, False),
        ({"cpu": 2}, 4, False, {"cpu": 2}, False, False),
        ({"cpu": 2, "memory": 2}, 4, True, {"cpu": 2, "memory": 2}, False, False),
        ({"cpu": 2, "memory": 8}, 4, False, {"cpu": 2, "memory": 8}, False, False),
        ({"cpu": 1, "memory": 8}, 4, False, {"cpu": 1, "memory": 8}, False, False),
        ({"cpu": 2, "memory": 8}, 4, False, {"cpu": 2, "memory": 8}, False, False),
        ({"cpu": 2, "memory": 8}, 4, False, {"cpu": 2, "memory": 8}, False, False),
        ({"cpu": 1, "memory": 8}, 4, True, {"cpu": 2, "memory": 8}, True, True),
        ({"cpu": 1, "memory": 18}, 7, True, {"cpu": 3, "memory": 18}, True, True),
        ({"cpu": 1, "memory": 7.5}, 4, True, {"cpu": 2, "memory": 7.5}, True, True),
    ],
)
def test_validate_and_adjust_resource_ratio(
    udf_resources,
    max_memory_cpu_ratio,
    adjust,
    expected_resources,
    expected_adjusted,
    should_warn,
    recwarn,
):
    result_resources, was_adjusted = validate_and_adjust_resource_ratio(
        udf_resources, max_memory_cpu_ratio, adjust
    )
    assert result_resources == expected_resources
    assert was_adjusted == expected_adjusted
    if should_warn:
        # check warning
        assert len(recwarn) == 1
