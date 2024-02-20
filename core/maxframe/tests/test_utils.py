# -*- coding: utf-8 -*-
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

import copy
import os
import shutil
import sys
import tempfile
import textwrap
import time
from enum import Enum
from functools import partial

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from .. import utils


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

    df3 = pd.DataFrame(
        {
            "A": np.random.choice(["abcd", "def", "gh"], size=(1000,)),
            "B": np.random.rand(1000),
            "C": np.random.rand(1000),
        }
    )
    assert utils.estimate_pandas_size(df3) != sys.getsizeof(df3)

    s1 = pd.Series(np.random.rand(1000))
    assert utils.estimate_pandas_size(s1) == sys.getsizeof(s1)

    from ..dataframe.arrays import ArrowStringArray

    array = ArrowStringArray(np.random.choice(["abcd", "def", "gh"], size=(1000,)))
    s2 = pd.Series(array)
    assert utils.estimate_pandas_size(s2) == sys.getsizeof(s2)

    s3 = pd.Series(np.random.choice(["abcd", "def", "gh"], size=(1000,)))
    assert utils.estimate_pandas_size(s3) != sys.getsizeof(s3)
    assert (
        pytest.approx(utils.estimate_pandas_size(s3) / sys.getsizeof(s3), abs=0.5) == 1
    )

    idx1 = pd.MultiIndex.from_arrays(
        [np.arange(0, 1000), np.random.choice(["abcd", "def", "gh"], size=(1000,))]
    )
    assert utils.estimate_pandas_size(idx1) == sys.getsizeof(idx1)

    string_idx = pd.Index(np.random.choice(["a", "bb", "cc"], size=(1000,)))
    assert utils.estimate_pandas_size(string_idx) != sys.getsizeof(string_idx)
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
    assert utils.estimate_pandas_size(df4) != sys.getsizeof(df4)
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
    _assert_arrow_type_convert(
        pa.struct([("key", pa.string()), ("value", pa.list_(pa.int64()))])
    )
