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

import pytest

from ..... import dataframe as md
from ...core import SPECodeContext
from ..groupby import (
    DataFrameGroupByAggAdapter,
    DataFrameGroupByOpAdapter,
    GroupByApplyAdapter,
    GroupByApplyChunkAdapter,
    GroupByExpandingAggAdapter,
    GroupByFillOperatorAdapter,
    GroupByHeadAdapter,
    GroupByIndexAdapter,
    GroupByRollingAggAdapter,
    GroupBySampleAdapter,
    GroupByTransformAdapter,
    _need_enforce_group_keys,
)


@pytest.fixture
def df1():
    return md.DataFrame(
        [
            [1, 10, 100, 1000],
            [2, 20, 200, 2000],
            [3, 30, 300, 3000],
            [7, 70, 700, 7000],
        ],
        index=md.Index([1, 2, 3, 7], name="test_idx"),
        columns=list("ABCD"),
    )


def test_dataframe_groupby(df1):
    context = SPECodeContext()
    adapter = DataFrameGroupByOpAdapter()

    res = df1.groupby("A")
    results = adapter.generate_code(res.op, context)
    expected_results = [
        "var_1 = var_0.groupby(by=['A'], as_index=True, sort=True, group_keys=True)"
    ]
    assert results == expected_results


def test_dataframe_groupby_agg(df1):
    context = SPECodeContext()
    adapter = DataFrameGroupByAggAdapter()

    res = df1.groupby("A").agg(["sum", "max", "median"])
    results = adapter.generate_code(res.op, context)
    expected_results = [
        "var_1 = var_0.groupby(by=['A'], as_index=True, sort=True, "
        "group_keys=True).agg(['sum', 'max', 'median'])"
    ]
    assert results == expected_results


@pytest.mark.parametrize(
    "func, func_kwargs, expected_results",
    [
        ("cumcount", {}, ["var_1 = {G}.cumcount(ascending=True)"]),
        ("cumcount", {"ascending": True}, ["var_1 = {G}.cumcount(ascending=True)"]),
        ("cumcount", {"ascending": False}, ["var_1 = {G}.cumcount(ascending=False)"]),
        ("cumsum", {}, ["var_1 = {G}.cumsum()"]),
        ("cumprod", {}, ["var_1 = {G}.cumprod()"]),
        ("cummax", {}, ["var_1 = {G}.cummax()"]),
        ("cummin", {}, ["var_1 = {G}.cummin()"]),
    ],
)
def test_dataframe_groupby_cum(df1, func, func_kwargs, expected_results):
    context = SPECodeContext()
    adapter = GroupByExpandingAggAdapter()

    groupby_code = "var_0.groupby(by=['A'], as_index=True, sort=True, group_keys=True)"
    res = getattr(df1.groupby("A"), func)(**func_kwargs)
    results = adapter.generate_code(res.op, context)
    assert results == [s.replace("{G}", groupby_code) for s in expected_results]


def test_dataframe_groupby_expanding(df1):
    context = SPECodeContext()
    adapter = GroupByExpandingAggAdapter()
    res = df1.groupby("A").expanding(2).sum()
    results = adapter.generate_code(res.op, context)
    assert results == [
        "def _exp_fun_var_1(frame, **_):",
        "    func = 'sum' if func != \"prod\" else lambda x: x.prod()",
        "    out_frame = frame.expanding(min_periods=2).agg(func)",
        "    return out_frame",
        "var_1 = var_0.groupby(by=['A'], as_index=True, sort=True, group_keys=True"
        ").apply(_exp_fun_var_1, include_groups=False)",
    ]

    res = df1.groupby("A").expanding(2, shift=1).sum()
    results = adapter.generate_code(res.op, context)
    assert results == [
        "def _exp_fun_var_2(frame, **_):",
        "    func = 'sum' if func != \"prod\" else lambda x: x.prod()",
        "    frame = frame.shift(-1)",
        "    out_frame = frame.expanding(min_periods=2).agg(func)",
        "    return out_frame",
        "var_2 = var_0.groupby(by=['A'], as_index=True, sort=True, "
        "group_keys=True).apply(_exp_fun_var_2, include_groups=False)",
    ]

    res = df1.groupby("A").expanding(2, shift=1, reverse_range=True).sum()
    results = adapter.generate_code(res.op, context)
    assert results == [
        "def _exp_fun_var_3(frame, **_):",
        "    func = 'sum' if func != \"prod\" else lambda x: x.prod()",
        "    frame = frame.shift(-1)",
        "    frame = frame.iloc[::-1]",
        "    out_frame = frame.expanding(min_periods=2).agg(func)",
        "    out_frame = out_frame.iloc[::-1]",
        "    return out_frame",
        "var_3 = var_0.groupby(by=['A'], as_index=True, sort=True, "
        "group_keys=True).apply(_exp_fun_var_3, include_groups=False)",
    ]


def test_dataframe_groupby_rolling(df1):
    context = SPECodeContext()
    adapter = GroupByRollingAggAdapter()
    res = df1.groupby("A").rolling(3).sum()
    results = adapter.generate_code(res.op, context)
    assert results == [
        "def _roll_fun_var_1(frame, **_):",
        "    func = 'sum' if func != \"prod\" else lambda x: x.prod()",
        "    frame = frame.shift(0)",
        "    return frame.rolling(window=3, min_periods=None, center=False, "
        "win_type=None, axis=0, on=None, closed=None).agg(func)",
        "var_1 = var_0.groupby(by=['A'], as_index=True, sort=True, "
        "group_keys=True).apply(_roll_fun_var_1, include_groups=False)",
    ]

    res = df1.groupby("A").rolling(3, shift=1).sum()
    results = adapter.generate_code(res.op, context)
    assert results == [
        "def _roll_fun_var_2(frame, **_):",
        "    func = 'sum' if func != \"prod\" else lambda x: x.prod()",
        "    frame = frame.shift(-1)",
        "    return frame.rolling(window=3, min_periods=None, center=False, "
        "win_type=None, axis=0, on=None, closed=None).agg(func)",
        "var_2 = var_0.groupby(by=['A'], as_index=True, sort=True, "
        "group_keys=True).apply(_roll_fun_var_2, include_groups=False)",
    ]


def test_dataframe_groupby_fill(df1):
    context = SPECodeContext()
    adapter = GroupByFillOperatorAdapter()

    res = df1.groupby("A").fillna(10)
    results = adapter.generate_code(res.op, context)
    expected_results = ["var_1 = var_0.fillna(10)"]
    assert results == expected_results

    res = df1.groupby("A").ffill()
    results = adapter.generate_code(res.op, context)
    expected_results = ["var_2 = var_0.ffill(axis=0)"]
    assert results == expected_results


def test_dataframe_groupby_head(df1):
    context = SPECodeContext()
    adapter = GroupByHeadAdapter()

    res = df1.groupby("A").head(10)
    results = adapter.generate_code(res.op, context)
    expected_results = [
        "var_1 = var_0.groupby(by=['A'], sort=True, group_keys=True).head(10)"
    ]
    assert results == expected_results


def test_dataframe_groupby_index(df1):
    context = SPECodeContext()
    adapter = GroupByIndexAdapter()

    res = df1.groupby("A")[["B", "C"]]
    results = adapter.generate_code(res.op, context)
    expected_results = ["var_1 = var_0[['B', 'C']]"]
    assert results == expected_results


def test_dataframe_groupby_apply(df1):
    context = SPECodeContext()
    adapter = GroupByApplyAdapter()

    res = df1.groupby("A").apply(lambda x: x + 1)
    results = adapter.generate_code(res.op, context)
    f = context.get_udfs()[0]
    expected_results = [f"var_1 = var_0.apply({f.name})"]
    assert results == expected_results


def test_dataframe_groupby_apply_chunk(df1):
    adapter = GroupByApplyChunkAdapter()

    res = df1.groupby("A").mf.apply_chunk(lambda x: x + 1)
    context = SPECodeContext()
    results = adapter.generate_code(res.op, context)
    f = context.get_udfs()[0]
    if _need_enforce_group_keys:
        expected_results = [
            "var_1 = var_0.groupby("
            "by=['A'], as_index=True, sort=True, group_keys=True)"
            f"[['B', 'C', 'D', 'A']].apply({f.name})",
            "var_1 = var_1.set_index(['A', var_1.index])",
        ]
    else:
        expected_results = [
            "var_1 = var_0.groupby("
            "by=['A'], as_index=True, sort=True, group_keys=True)"
            f"[['B', 'C', 'D']].apply({f.name})",
        ]
    assert results == expected_results

    res = df1.groupby("A").mf.apply_chunk(lambda x: x + 1, batch_rows=5)
    context = SPECodeContext()
    results = adapter.generate_code(res.op, context)
    f = context.get_udfs()[0]
    if _need_enforce_group_keys:
        expected_results = [
            "def batch_apply_var_0(frame, *args, **kw):",
            f"    batches = [{f.name}(frame.iloc[i "
            ": i + 5], *args, **kw)     for i in range(0, len(frame), 5)]",
            "    if len(batches) > 1:",
            "        return pd.concat(batches)",
            "    else:",
            "        return batches[0]",
            "var_1 = var_0.groupby(by=['A'], as_index=True, sort=True, "
            "group_keys=True)[['B', 'C', 'D', 'A']].apply(batch_apply_var_0)",
            "var_1 = var_1.set_index(['A', var_1.index])",
        ]
    else:
        expected_results = [
            "def batch_apply_var_0(frame, *args, **kw):",
            f"    batches = [{f.name}(frame.iloc[i "
            ": i + 5], *args, **kw)     for i in range(0, len(frame), 5)]",
            "    if len(batches) > 1:",
            "        return pd.concat(batches)",
            "    else:",
            "        return batches[0]",
            "var_1 = var_0.groupby(by=['A'], as_index=True, sort=True, "
            "group_keys=True)[['B', 'C', 'D']].apply(batch_apply_var_0)",
        ]
    assert results == expected_results


def test_dataframe_groupby_transform(df1):
    context = SPECodeContext()
    adapter = GroupByTransformAdapter()

    res = df1.groupby("A").transform(lambda x: x + 1)
    results = adapter.generate_code(res.op, context)
    f = context.get_udfs()[0]
    expected_results = [f"var_1 = var_0.transform({f.name})"]
    assert results == expected_results


def test_dataframe_groupby_sample(df1):
    context = SPECodeContext()
    adapter = GroupBySampleAdapter()

    res = df1.groupby("A").sample(frac=0.5)
    results = adapter.generate_code(res.op, context)
    expected_results = [
        "var_1 = var_0.groupby(by=['A'], sort=True, group_keys=True)"
        ".agg(frac=0.5, replace=False)"
    ]
    assert results == expected_results
