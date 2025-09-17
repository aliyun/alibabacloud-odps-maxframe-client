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

import pandas as pd
import pytest

from ..... import dataframe as md
from ...core import SPECodeContext
from ..merge import (
    DataFrameConcatAdapter,
    DataFrameMergeAdapter,
    DataFrameUpdateAdapter,
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


@pytest.fixture
def df2():
    return md.DataFrame(
        [
            [1, 10, 101, 1001],
            [2, 20, 202, 2002],
            [3, 30, 303, 3003],
            [8, 80, 808, 8008],
        ],
        index=pd.MultiIndex.from_arrays(
            [[1, 2, 3, 8], [1, 2, 3, 8], [1, 2, 3, 8]],
            names=("idx_1", "idx_2", "idx_4"),
        ),
        columns=list("ABCE"),
    )


@pytest.fixture
def df3():
    return md.DataFrame(
        [
            [1, 10, 101],
            [9, 90, 909],
            [2, 20, 202],
        ],
        index=pd.MultiIndex.from_arrays(
            [[1, 9, 2], [1, 9, 2], [1, 9, 2]], names=("idx_3", "idx_4", "idx_5")
        ),
        columns=list("ZCE"),
    )


@pytest.fixture
def df4():
    return md.DataFrame(
        [
            [1, 10, 100, 1000],
            [2, 20, 200, 2000],
            [3, 30, 300, 3000],
            [4, 400, 4000, 40000],
        ],
        index=[1, 2, 3, 4],
        columns=list("ABCD"),
    )


@pytest.fixture
def df5():
    return md.DataFrame(
        [
            [1, 10, 100, 1000],
            [2, 20, 200, 2000],
            [3, 30, 300, 3000],
            [4, 4000, 40000, 400000],
        ],
        index=[1, 2, 3, 4],
        columns=list("ABCD"),
    )


@pytest.fixture
def df6():
    # Same index with df2
    return md.DataFrame(
        [
            [1, 10, 101, 1001],
            [2, 20, 202, 2002],
            [3, 30, 303, 3003],
            [8, 810, 8108, 81008],
        ],
        index=pd.MultiIndex.from_arrays(
            [[1, 2, 3, 8], [1, 2, 3, 8], [1, 2, 3, 8]],
            names=("idx_1", "idx_2", "idx_4"),
        ),
        columns=list("ABCF"),
    )


@pytest.fixture
def s1():
    return md.Series([5, 2, 3, 1], index=[5, 2, 3, 1], name="A")


@pytest.fixture
def s2():
    return md.Series([1, 2, 3, 4], index=[1, 2, 3, 4], name="B")


@pytest.fixture
def s3():
    return md.Series(
        [1, 2, 3, 6],
        index=pd.MultiIndex.from_arrays(
            [[1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6]],
            names=("idx_1", "idx_2", "idx_3"),
        ),
        name="A",
    )


def test_series_merge_series_using_index(s1, s2):
    merged = md.merge(s1, s2, left_index=True, right_index=True)
    context = SPECodeContext()
    adapter = DataFrameMergeAdapter()
    results = adapter.generate_code(merged.op, context)
    assert "pd" in context.imports["pandas"]
    expected_results = [
        "var_2 = pd.merge(var_0, var_1, how='inner', left_index=True, right_index=True,"
        " sort=False, suffixes=('_x', '_y'), indicator=False, validate=None, copy=True)"
    ]
    assert results == expected_results


def test_series_merge_series_on_index_and_columns(s1, s3):
    merged = md.merge(
        s1,
        s3,
        suffixes=("_left", "_right"),
        how="left",
        left_on="A",
        right_on="idx_2",
        indicator=True,
    )
    context = SPECodeContext()
    adapter = DataFrameMergeAdapter()
    results = adapter.generate_code(merged.op, context)
    expected_results = [
        "var_2 = pd.merge(var_0, var_1, how='left', left_on='A', right_on='idx_2',"
        " sort=False, suffixes=('_left', '_right'), indicator=True, validate=None,"
        " copy=True)"
    ]
    assert results == expected_results


def test_series_merge_series_on_columns(s1, s3):
    merged = md.merge(s1, s3, how="right", on="A", sort=True, copy=False)
    context = SPECodeContext()
    adapter = DataFrameMergeAdapter()
    results = adapter.generate_code(merged.op, context)
    expected_results = [
        "var_2 = pd.merge(var_0, var_1, how='right', on='A',"
        " sort=True, suffixes=('_x', '_y'), indicator=False, validate=None, copy=False)"
    ]
    assert results == expected_results


def test_series_merge_dataframe_using_index(s1, df1):
    merged = md.merge(
        s1, df1, how="outer", left_index=True, right_index=True, validate="m:m"
    )
    context = SPECodeContext()
    adapter = DataFrameMergeAdapter()
    results = adapter.generate_code(merged.op, context)
    expected_results = [
        "var_2 = pd.merge(var_0, var_1, how='outer', left_index=True, right_index=True,"
        " sort=False, suffixes=('_x', '_y'), indicator=False, validate='m:m', copy=True)"
    ]
    assert results == expected_results


def test_series_merge_dataframe_on_index_and_columns(s1, df2):
    merged = md.merge(
        s1,
        df2,
        suffixes=("_left", "_right"),
        how="outer",
        left_on="A",
        right_on="idx_2",
        sort=True,
    )
    context = SPECodeContext()
    adapter = DataFrameMergeAdapter()
    results = adapter.generate_code(merged.op, context)
    expected_results = [
        "var_2 = pd.merge(var_0, var_1, how='outer', left_on='A', right_on='idx_2',"
        " sort=True, suffixes=('_left', '_right'), indicator=False, validate=None,"
        " copy=True)"
    ]
    assert results == expected_results


def test_series_merge_dataframe_on_columns(s1, df2):
    merged = md.merge(s1, df2, how="cross")
    context = SPECodeContext()
    adapter = DataFrameMergeAdapter()
    results = adapter.generate_code(merged.op, context)
    expected_results = [
        "var_2 = pd.merge(var_0, var_1, how='cross',"
        " sort=False, suffixes=('_x', '_y'), indicator=False, validate=None, copy=True)"
    ]
    assert results == expected_results


def test_dataframe_merge_dataframe_using_index(df1, df2):
    merged = md.merge(df1, df2, how="inner", left_index=True, right_on="idx_4")
    context = SPECodeContext()
    adapter = DataFrameMergeAdapter()
    results = adapter.generate_code(merged.op, context)
    expected_results = [
        "var_2 = pd.merge(var_0, var_1, how='inner', left_index=True, right_on='idx_4',"
        " sort=False, suffixes=('_x', '_y'), indicator=False, validate=None, copy=True)"
    ]
    assert results == expected_results


def test_dataframe_merge_dataframe_on_index_and_columns(df1, df2):
    merged = md.merge(
        df1,
        df2,
        suffixes=("_left", "_right"),
        how="outer",
        left_on=["A", "B"],
        right_on=["idx_1", "idx_4"],
        sort=True,
    )
    context = SPECodeContext()
    adapter = DataFrameMergeAdapter()
    results = adapter.generate_code(merged.op, context)
    expected_results = [
        "var_2 = pd.merge(var_0, var_1, how='outer', left_on=['A', 'B'],"
        " right_on=['idx_1', 'idx_4'], sort=True, suffixes=('_left', '_right'),"
        " indicator=False, validate=None, copy=True)"
    ]
    assert results == expected_results


def test_dataframe_merge_dataframe_on_columns(df1, df2):
    merged = md.merge(df1, df2, how="outer", on=["A", "B"], sort=True)
    context = SPECodeContext()
    adapter = DataFrameMergeAdapter()
    results = adapter.generate_code(merged.op, context)
    expected_results = [
        "var_2 = pd.merge(var_0, var_1, how='outer', on=['A', 'B'],"
        " sort=True, suffixes=('_x', '_y'), indicator=False, validate=None, copy=True)"
    ]
    assert results == expected_results


def test_join(df1, df2):
    joined = df1.join(df2, how="right", on=["A", "B", "C"], lsuffix="_l", rsuffix="_r")
    context = SPECodeContext()
    adapter = DataFrameMergeAdapter()
    results = adapter.generate_code(joined.op, context)
    expected_results = [
        "var_2 = pd.merge(var_0, var_1, how='right', left_on=['A', 'B', 'C'],"
        " right_index=True, sort=False, suffixes=('_l', '_r'), indicator=False,"
        " validate=None, copy=True)"
    ]
    assert results == expected_results


def test_concat_single(df2):
    adapter = DataFrameConcatAdapter()
    df = md.concat([df2], axis="index")
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)

    expected_results = [
        "var_0 = pd.concat(axis=0, join='outer', ignore_index=False, keys=None,"
        " levels=None, names=None, verify_integrity=False, sort=False, objs=[var_1],"
        " copy=True)"
    ]
    assert results == expected_results

    expected_results = [
        "var_0 = pd.concat(axis=1, join='outer', ignore_index=False, keys=None,"
        " levels=None, names=None, verify_integrity=False, sort=False, objs=[var_1],"
        " copy=True)"
    ]
    df = md.concat([df2], axis="columns")
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    assert results == expected_results


def test_concat_indexes_from_series(s1, s2):
    adapter = DataFrameConcatAdapter()
    df = md.concat([s1, s2], names=["l1", "l2"], axis="index", ignore_index=True)
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_0 = pd.concat(axis=0, join='outer', ignore_index=True, keys=None,"
        " levels=None, names=['l1', 'l2'], verify_integrity=False, sort=False,"
        " objs=[var_1, var_2], copy=True)"
    ]
    assert results == expected_results


def test_concat_indexes_from_series_and_dataframes(s1, s2, df1):
    adapter = DataFrameConcatAdapter()
    df = md.concat([s1, s2, df1], axis=0, copy=False)
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_0 = pd.concat(axis=0, join='outer', ignore_index=False, keys=None,"
        " levels=None, names=None, verify_integrity=False, sort=False,"
        " objs=[var_1, var_2, var_3], copy=False)"
    ]
    assert results == expected_results


def test_concat_indexes_from_inner_join_dataframes(df2, df3):
    adapter = DataFrameConcatAdapter()
    df = md.concat([df2, df3], join="inner")
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_0 = pd.concat(axis=0, join='inner', ignore_index=False, keys=None,"
        " levels=None, names=None, verify_integrity=False, sort=False,"
        " objs=[var_1, var_2], copy=True)"
    ]
    assert results == expected_results


def test_concat_indexes_from_outer_join_dataframes(df2, df3):
    adapter = DataFrameConcatAdapter()
    df = md.concat([df2, df3], join="outer")
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_0 = pd.concat(axis=0, join='outer', ignore_index=False, keys=None,"
        " levels=None, names=None, verify_integrity=False, sort=False,"
        " objs=[var_1, var_2], copy=True)"
    ]
    assert results == expected_results


def test_concat_columns_from_series(s1, s2):
    adapter = DataFrameConcatAdapter()
    df = md.concat([s1, s2], axis="columns", ignore_index=True)
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_0 = pd.concat(axis=1, join='outer', ignore_index=True, keys=None,"
        " levels=None, names=None, verify_integrity=False, sort=False, "
        "objs=[var_1, var_2], copy=True)"
    ]
    assert results == expected_results


def test_concat_columns_from_series_and_dataframes(s2, df4, df5):
    adapter = DataFrameConcatAdapter()
    df = md.concat([s2, df4, df5], axis=1)
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_0 = pd.concat(axis=1, join='outer', ignore_index=False, keys=None,"
        " levels=None, names=None, verify_integrity=False, sort=False,"
        " objs=[var_1, var_2, var_3], copy=True)"
    ]
    assert results == expected_results


def test_concat_columns_from_dataframes(df2, df6):
    adapter = DataFrameConcatAdapter()
    df = md.concat([df2, df6], axis=1)
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = [
        "var_0 = pd.concat(axis=1, join='outer', ignore_index=False, keys=None,"
        " levels=None, names=None, verify_integrity=False, sort=False,"
        " objs=[var_1, var_2], copy=True)"
    ]
    assert results == expected_results


def test_update_dataframe(df5, df6):
    adapter = DataFrameUpdateAdapter()
    df5.update(df6)
    context = SPECodeContext()
    results = adapter.generate_code(df5.op, context)
    expected_results = [
        "var_1 = var_0.copy()",
        "var_1.update(var_2, join='left', overwrite=True, filter_func=None,"
        " errors='ignore')",
    ]
    assert results == expected_results


def test_combine_first_dataframe(df5, df6):
    adapter = DataFrameUpdateAdapter()
    df = df5.combine_first(df6)
    context = SPECodeContext()
    results = adapter.generate_code(df.op, context)
    expected_results = ["var_1 = var_0.combine_first(var_2)"]
    assert results == expected_results
