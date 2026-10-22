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

import pytest

from maxframe import opcodes
from maxframe.liteframe import LiteFrame
from maxframe.liteframe.core import LITEFRAME_TYPE, FrameMetadata
from maxframe.liteframe.datastore.to_csv import to_csv
from maxframe.liteframe.datastore.to_json import to_json
from maxframe.liteframe.datastore.to_lance import to_lance
from maxframe.liteframe.datastore.to_odps import to_odps_table
from maxframe.liteframe.operators.core import LiteFrameDataFormat
from maxframe.liteframe.operators.datastore import (
    LiteFrameLakeDataStore,
    LiteFrameToLance,
    LiteFrameToODPSTable,
)

# Operator __call__ tests


def test_operator_call_creates_stub():
    """Test operator __call__ creates proper stub output"""
    lf = LiteFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    op = LiteFrameToODPSTable(dtypes=lf.dtypes, table_name="test_table")
    result = op(lf)

    assert result.shape == (0, 0)
    assert len(result.dtypes) == 0


# API function tests


def test_function_creates_operator():
    """Test API function creates operator and calls it"""
    lf = LiteFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = to_odps_table(lf, "test_table")

    assert isinstance(result, LITEFRAME_TYPE)
    assert result.op._op_type_ == opcodes.TO_ODPS_TABLE
    assert result.op.table_name == "test_table"
    assert result.op.mode == "overwrite"


# Validation tests


def test_to_odps_table_validates_partition_columns():
    """Test that API function validates partition columns exist in LiteFrame"""
    lf = LiteFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with pytest.raises(ValueError, match="Partition column.*is not the data column"):
        to_odps_table(lf, "test_table", partition_col="nonexistent_column")


def test_to_csv_function_validates_mode():
    """Test API function validates mode parameter"""
    lf = LiteFrame({"a": [1, 2, 3]})

    with pytest.raises(NotImplementedError, match="only support to_csv with mode 'w'"):
        to_csv(lf, "output-*.csv", mode="a")


def test_to_json_validates_orient_with_lines():
    """Test that to_json validates orient must be 'records' when lines=True"""
    lf = LiteFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="orient must be 'records' when lines=True"):
        to_json(lf, "output-*.json", lines=True, orient="columns")


def test_to_json_fields():
    """Test to_json operator fields"""
    lf = LiteFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = to_json(lf, "output-*.jsonl", lines=True, compression="gzip")

    op = result.op
    assert isinstance(op, LiteFrameLakeDataStore)
    assert op.data_format == LiteFrameDataFormat.JSON
    assert op.path == "output-*.jsonl"
    assert op.write_kwargs["lines"] is True
    assert op.write_kwargs["orient"] == "records"
    assert op.compression == "gzip"


def test_to_lance_operator_call():
    """Test LiteFrameToLance.__call__ creates stub with shape (0, 0)"""
    lf = LiteFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    op = LiteFrameToLance(path="output.lance", mode="overwrite")
    result = op(lf)
    assert result.shape == (0, 0)
    assert len(result.dtypes) == 0


def test_to_lance_function_creates_operator():
    """Test to_lance API creates operator and calls it"""
    lf = LiteFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = to_lance(lf, "output.lance")

    assert isinstance(result, LITEFRAME_TYPE)
    assert result.op._op_type_ == opcodes.TO_LANCE
    assert result.op.path == "output.lance"
    assert result.op.mode == "overwrite"


def test_to_lance_validates_mode():
    """Test to_lance raises ValueError for invalid mode"""
    lf = LiteFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="mode must be"):
        to_lance(lf, "output.lance", mode="invalid")


def test_to_lance_merge_requires_rowaddr():
    """Test merge mode raises ValueError when _rowaddr is missing"""
    lf = LiteFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="_rowaddr"):
        to_lance(lf, "output.lance", mode="merge")


def test_to_lance_on_and_left_on_exclusive():
    """Test that on and left_on/right_on are mutually exclusive"""
    lf = LiteFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="Cannot specify both"):
        to_lance(lf, "output.lance", mode="merge", on="x", left_on="y")


def test_to_lance_on_resolves_to_left_right():
    """Test that on parameter is resolved into left_on and right_on"""
    lf = LiteFrame({"a": [1, 2, 3], "_rowaddr": [0, 1, 2]})

    lf._data.frame_metadata = FrameMetadata(hidden_columns=["_rowaddr"])

    result = to_lance(lf, "output.lance", mode="merge", on="my_key")
    assert result.op.left_on == "my_key"
    assert result.op.right_on == "my_key"


def test_to_lance_merge_defaults_rowaddr():
    """Test merge mode defaults left_on/right_on to _rowaddr"""
    lf = LiteFrame({"a": [1, 2, 3], "_rowaddr": [0, 1, 2]})

    lf._data.frame_metadata = FrameMetadata(hidden_columns=["_rowaddr"])

    result = to_lance(lf, "output.lance", mode="merge")
    assert result.op.left_on == "_rowaddr"
    assert result.op.right_on == "_rowaddr"


def test_to_lance_one_file_property():
    """Test one_file is always False for Lance"""
    op = LiteFrameToLance(path="output.lance", mode="overwrite")
    assert op.one_file is False
