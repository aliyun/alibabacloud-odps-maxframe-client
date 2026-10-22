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

from maxframe.codegen.spe.core import SPECodeContext
from maxframe.codegen.spe.liteframe.indexing import (
    LiteFrameIlocGetItemAdapter,
    LiteFrameIlocSetItemAdapter,
)
from maxframe.liteframe.operators.indexing import (
    LiteFrameIlocGetItem,
    LiteFrameIlocSetItem,
)


def test_iloc_getitem_adapter(make_liteframe):
    """LiteFrameIlocGetItemAdapter should generate .iloc[] access."""
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    data = make_liteframe(df)
    op = LiteFrameIlocGetItem(indexes=[slice(0, 2), slice(None)])
    out = op(data)
    op = out.op
    adapter = LiteFrameIlocGetItemAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 1
    assert ".iloc[" in result[0]


def test_iloc_setitem_adapter(make_liteframe):
    """LiteFrameIlocSetItemAdapter should generate .copy() + .iloc[] = assignment."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    data = make_liteframe(df)
    op = LiteFrameIlocSetItem(indexes=[slice(0, 2), slice(None)], value=0)
    out = op(data)
    op = out.op
    adapter = LiteFrameIlocSetItemAdapter()
    context = SPECodeContext()
    result = adapter.generate_code(op, context)
    assert len(result) == 2
    assert ".copy()" in result[0]
    assert ".iloc[" in result[1]
