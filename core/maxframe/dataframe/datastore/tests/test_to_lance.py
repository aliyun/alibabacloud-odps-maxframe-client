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

import pandas as pd
import pytest

from maxframe import opcodes
from maxframe.dataframe.datastore.to_lance import DataFrameToLance, to_lance
from maxframe.dataframe.initializer import DataFrame

try:
    import lance
except (ImportError, AttributeError):
    # lance not installed or not compatible
    lance = None


def test_to_lance_op_creation():
    """Test DataFrameToLance operator creation with OSS path and role_arn."""
    op = DataFrameToLance(
        path="oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset",
        mode="create",
        storage_options={"role_arn": "acs:ram::1234567890:role/maxframe-oss"},
    )
    assert op._op_type_ == opcodes.TO_LANCE
    assert op.path == "oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset"
    assert op.mode == "create"
    assert op.storage_options == {"role_arn": "acs:ram::1234567890:role/maxframe-oss"}


def test_to_lance_op_modes():
    """Test different write modes."""
    for mode in ["create", "append", "overwrite"]:
        op = DataFrameToLance(
            path="oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset",
            mode=mode,
        )
        assert op.mode == mode


def test_to_lance_invalid_mode():
    """Test that invalid mode raises error."""
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    df = DataFrame(pdf)
    with pytest.raises(ValueError, match="mode must be"):
        to_lance(
            df,
            "oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset",
            mode="invalid",
        )


def test_to_lance_with_lance_kwargs():
    """Test passing additional lance kwargs."""
    op = DataFrameToLance(
        path="oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset",
        mode="create",
        lance_kwargs={"max_rows_per_file": 1000000},
    )
    assert op.lance_kwargs == {"max_rows_per_file": 1000000}


def test_to_lance_function():
    """Test to_lance function creates correct operator."""
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = DataFrame(pdf)

    result = to_lance(
        df,
        "oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset",
        mode="create",
    )
    # Result should be a MaxFrame DataFrame (empty, as it's a write operation)
    assert hasattr(result, "execute")
    assert result.shape == (0, 0)


def test_to_lance_with_storage_options():
    """Test to_lance with storage options using role_arn."""
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    df = DataFrame(pdf)

    result = to_lance(
        df,
        "oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset",
        storage_options={"role_arn": "acs:ram::1234567890:role/maxframe-oss"},
    )
    assert hasattr(result, "execute")


def test_to_lance_write_stage_field():
    """Test write_stage field (used by DPE for two-stage write)."""
    op = DataFrameToLance(
        path="oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset",
        mode="create",
        write_stage="write_fragments",
    )
    assert op.write_stage == "write_fragments"

    op2 = DataFrameToLance(
        path="oss://oss-cn-beijing.aliyuncs.com/my-bucket/dataset",
        mode="append",
        write_stage="commit",
    )
    assert op2.write_stage == "commit"
