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

import pyarrow as pa
import pytest

from maxframe.liteframe import LiteFrame
from maxframe.liteframe.operators.drop_duplicates import (
    MASK_COL_NAME,
    LiteFrameDropDuplicates,
)
from maxframe.utils import wrap_arrow_dtype

_BOOL_DTYPE = wrap_arrow_dtype(pa.bool_())


@pytest.mark.parametrize(
    "which", ["drop_duplicates", "duplicated"], ids=["drop", "dup"]
)
def test_invalid_args_raise(which):
    lf = LiteFrame({"a": [1, 1, 2]})
    call = getattr(lf, which)
    with pytest.raises(ValueError, match="keep"):
        call(keep=0)  # 0 == False must not pass validation (identity check)
    with pytest.raises(ValueError, match="method"):
        call(method="random")
    with pytest.raises(KeyError):
        call(subset=["missing"])
    with pytest.raises(ValueError, match="empty list"):
        call(subset=[])


@pytest.mark.parametrize(
    "which,return_mask,return_value",
    [("drop_duplicates", False, True), ("duplicated", True, False)],
    ids=["value", "mask"],
)
def test_op_structure(which, return_mask, return_value):
    lf = LiteFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
    result = getattr(lf, which)()
    op = result.data.op
    assert isinstance(op, LiteFrameDropDuplicates)
    assert op.return_mask is return_mask
    assert op.return_value is return_value
    assert op.output_limit == 1
    assert op.keep == "first"
    if return_mask:
        assert result.dtypes[MASK_COL_NAME] == _BOOL_DTYPE


def test_dual_output_rejected():
    lf = LiteFrame({"a": [1, 1, 2]})
    with pytest.raises(ValueError, match="cannot both be True"):
        LiteFrameDropDuplicates(
            subset=["a"],
            keep="first",
            return_mask=True,
            return_value=True,
        )(lf)
