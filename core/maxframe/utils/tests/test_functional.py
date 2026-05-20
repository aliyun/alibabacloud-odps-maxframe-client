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

import warnings
from functools import partial

import maxframe.dataframe as md
from maxframe.utils.functional import (
    check_closure_for_entities,
    deprecate_positional_args,
)


def test_deprecate_positional_args():
    """Test function with many parameters."""

    @deprecate_positional_args
    def func(a, b=1, c=2, d=3, e=4):
        return a, b, c, d, e

    # Test that calling with many positional args generates warning for all after first
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # b, c, d, e passed positionally
        result = func(10, 20, 30, 40, 50)
        assert len(w) == 0  # No warnings should be raised
        assert result == (10, 20, 30, 40, 50)

    @deprecate_positional_args
    def func(a):
        return a

    # Test that calling with single positional arg works without warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = func(10)
        assert len(w) == 0  # No warnings should be raised
        assert result == 10

    @deprecate_positional_args
    def func(a=1, *, b=2, c=3):
        return a, b, c

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = func(10, 20, 30)  # b and c passed positionally
        assert len(w) == 1  # One warning for b and c
        assert "'b'" in str(w[0].message) and "'c'" in str(w[0].message)
        assert result == (10, 20, 30)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = func(10)  # no args passed positionally
        assert len(w) == 0  # No warning for b and c
        assert result == (10, 2, 3)


def test_check_closure_entities():
    """Test closure variable detection."""
    # Test entities in closure trigger warning
    df = md.DataFrame({"a": [1, 2, 3]})
    series = md.Series([1, 2, 3])

    def func_with_df(x):
        return x + df["a"].sum()

    def func_with_series(x):
        return x + series.sum()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_closure_for_entities(func_with_df, "apply")
        assert len(w) == 1
        assert "MaxFrame entities" in str(w[0].message)
        assert "apply" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_closure_for_entities(func_with_series, "apply")
        assert len(w) == 1
        assert "MaxFrame entities" in str(w[0].message)

    # Test no entities produces no warning
    def func_no_entity(x):
        return x + 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_closure_for_entities(func_no_entity, "apply")
        assert len(w) == 0


def test_check_global_entities():
    """Test global variable detection via bytecode."""
    global_df = md.DataFrame({"a": [1, 2, 3]})
    _unused_df = md.DataFrame({"b": [4, 5, 6]})  # noqa: F841

    def func_uses_global(x):
        return x + global_df["a"].sum()

    def func_ignores_global(x):
        # _unused_df exists in globals but not referenced
        return x + 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_closure_for_entities(func_uses_global, "apply")
        assert len(w) == 1
        assert "MaxFrame entities" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_closure_for_entities(func_ignores_global, "apply")
        assert len(w) == 0  # No warning for unreferenced global


def test_check_partial_entities():
    """Test partial function detection."""
    df = md.DataFrame({"a": [1, 2, 3]})

    def my_func(x, df_arg, df_kwarg=None):
        return x

    # Test entity in partial args
    partial_func = partial(my_func, df_arg=df)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_closure_for_entities(partial_func, "apply")
        assert len(w) == 1

    # Test entity in partial kwargs
    partial_func2 = partial(my_func, df_arg=1, df_kwarg=df)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_closure_for_entities(partial_func2, "apply_chunk")
        assert len(w) == 1


def test_check_closure_integration():
    """Test integration with apply operations."""
    _df = md.DataFrame({"a": [1, 2, 3]})  # noqa: F841
    captured_df = md.DataFrame({"b": [4, 5, 6]})

    def func_with_closure(x):
        return x + captured_df["b"].sum()

    # Test that the warning is raised when checking the function
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_closure_for_entities(func_with_closure, "apply")
        assert len(w) == 1
        assert "apply" in str(w[0].message)
