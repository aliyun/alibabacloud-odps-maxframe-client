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
from maxframe.liteframe.core import LITEFRAME_TYPE
from maxframe.liteframe.datastore.to_odps import LiteFrameToODPSTable, to_odps_table

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
