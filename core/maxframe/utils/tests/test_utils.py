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

import os
import shutil
import sys
import tempfile
import textwrap
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ... import utils
from ..datatypes import is_arrow_dtype_supported, wrap_arrow_dtype
from ..utils import parse_size_to_megabytes, validate_and_adjust_resource_ratio


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

    if is_arrow_dtype_supported():
        arrow_array = pa.array(np.random.choice(["abcd", "def", "gh"], size=(1000,)))
        array = pd.array(arrow_array, dtype=wrap_arrow_dtype(arrow_array.type))
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


def test_check_unexpected_kwargs():
    with pytest.raises(TypeError) as excinfo:
        utils.check_unexpected_kwargs({"a": 1, "b": 2, "c": 3})
    assert (
        excinfo.value.args[0]
        == "test_check_unexpected_kwargs() got an unexpected keyword argument 'a'"
    )


def test_key_log_wrapper():
    keys = [uuid.uuid4().hex for _ in range(10)]
    wrapped_log = str(utils.KeyLogWrapper(keys))
    sub_strs = [s.rstrip(".").strip() for s in wrapped_log[1:-1].split(",")]
    assert [k[:8] for k in keys] == sub_strs

    wrapped_log = str(utils.KeyLogWrapper(keys, limit=5))
    assert ", and 5 more" in wrapped_log
    sub_strs = [s.rstrip(".").strip() for s in wrapped_log[1:-1].split(",")[:-1]]
    assert [k[:8] for k in keys[:5]] == sub_strs
