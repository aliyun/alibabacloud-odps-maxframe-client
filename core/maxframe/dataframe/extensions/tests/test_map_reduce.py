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

import functools
from collections import defaultdict

import pandas as pd
import pytest

from .... import dataframe as md
from ...groupby.apply_chunk import GroupByApplyChunk
from .. import DataFrameApplyChunk


@pytest.fixture
def df1():
    return md.DataFrame(
        {
            "name": ["name key", "name", "key", "name", "key name"],
            "id": [4, 2, 4, 3, 3],
            "fid": [5.3, 3.5, 4.2, 2.2, 4.1],
        }
    )


def test_map_reduce_with_map_only(df1):
    func = functools.partial(lambda x: x)

    with pytest.raises(ValueError):
        df1.mf.map_reduce(func, reducer_dtypes={"col": "string"})

    mapped = df1.mf.map_reduce(func)
    assert isinstance(mapped.op, DataFrameApplyChunk)
    assert mapped.op.func is func

    map_combined = df1.mf.map_reduce(
        func, combiner=func, mapper_dtypes=df1.dtypes, mapper_index=df1.index
    )
    assert isinstance(map_combined.op, DataFrameApplyChunk)
    assert map_combined.op.func.__name__ == "CombinedMapper"


def test_mapper_with_combiner(df1):
    class BaseFunc:
        def __init__(self):
            self._word_to_count = defaultdict(lambda: 0)

        def _collect_df(self):
            word_to_count = self._word_to_count.copy()
            self._word_to_count.clear()
            return pd.DataFrame(
                [list(tp) for tp in word_to_count.items()],
                columns=["word", "count"],
            )

        def close(self):
            print(f"Close {type(self)}")

    class MapperCls(BaseFunc):
        def __call__(self, batch, end=False):
            for words in batch["name"]:
                for w in words.split():
                    self._word_to_count[w] += 1
            if end:
                return self._collect_df()

    class CombinerCls(BaseFunc):
        def __call__(self, batch, end=False):
            for _, row in batch.iterrows():
                self._word_to_count[row["word"]] = row["count"]
            if end:
                return self._collect_df()

    map_combined = df1.mf.map_reduce(
        MapperCls,
        combiner=CombinerCls,
        group_cols="word",
        mapper_dtypes={"word": "str", "count": "int"},
        mapper_index=df1.index,
    )

    raw = df1.op.data
    combiner = map_combined.op.func()
    ret1 = combiner(raw.iloc[:3], end=True)
    ret2 = combiner(raw.iloc[3:], end=True)
    close_ret = combiner.close()
    expected1 = pd.DataFrame([["key", 2], ["name", 2]], columns=["word", "count"])
    expected2 = pd.DataFrame([["key", 1], ["name", 2]], columns=["word", "count"])
    assert close_ret is None
    pd.testing.assert_frame_equal(ret1.reset_index(drop=True), expected1)
    pd.testing.assert_frame_equal(ret2.reset_index(drop=True), expected2)


def test_map_reduce_with_reduce_only(df1):
    func = functools.partial(lambda x: x)

    with pytest.raises(ValueError):
        df1.mf.map_reduce(reducer=func, mapper_dtypes={"col": "string"})

    reduced = df1.mf.map_reduce(reducer=func, group_cols="name")
    assert isinstance(reduced.op, GroupByApplyChunk)
    assert reduced.op.func is func
    assert reduced.op.groupby_params["by"] == ["name"]


def test_map_reduce_with_both_funcs(df1):
    map_func = functools.partial(lambda x: x)

    class ReducerCls:
        def __call__(self, batch):
            return batch

    reduced = df1.mf.map_reduce(
        mapper=map_func,
        reducer=ReducerCls,
        group_cols="name",
        reducer_dtypes=df1.dtypes,
        reducer_index=df1.index,
    )
    assert isinstance(reduced.op, GroupByApplyChunk)
    assert reduced.op.func is ReducerCls
    assert reduced.op.groupby_params["by"] == ["name"]
    assert isinstance(reduced.inputs[0].op, DataFrameApplyChunk)
    assert reduced.inputs[0].op.func is map_func
