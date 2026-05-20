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

import re
from datetime import date, datetime

import pyarrow.dataset as ds
import pytest

from maxframe.utils.filters import (
    _convert_single_filter,
    convert_filters_to_arrow_expression,
    convert_filters_to_sql,
    validate_filters,
)


def _clean_expr_str(expr) -> str:
    """Helper to remove type annotations like '0:int64'"""
    if expr is None:
        return None
    # Remove type annotations like ':int64', ':double', etc.
    # Pattern matches colon followed by type name
    return re.sub(r":[a-zA-Z0-9_]+", "", str(expr)).replace("is in", "is_in")


# Tests for filter conversion utilities
@pytest.mark.parametrize(
    "filter_input,expected",
    [
        ("col1 = 'value'", "col1 = 'value'"),
        ("col1 > 10 AND col2 = 'test'", "col1 > 10 AND col2 = 'test'"),
        ("age BETWEEN 18 AND 65", "age BETWEEN 18 AND 65"),
    ],
)
def test_string_filter(filter_input, expected):
    """Test string filter is passed through"""

    assert convert_filters_to_sql(filter_input) == expected


def test_simple_list_filter():
    """Test simple list filter with single condition"""

    filters = [[("col1", "==", "value")]]
    assert convert_filters_to_sql(filters) == "`col1` = 'value'"


def test_and_filter():
    """Test multiple conditions ANDed together"""

    filters = [[("col1", "==", "value"), ("col2", ">", 10)]]
    result = convert_filters_to_sql(filters)
    assert "`col1` = 'value'" in result
    assert "`col2` > 10" in result
    assert "AND" in result


def test_or_filter():
    """Test multiple groups ORed together"""

    filters = [[("col1", "==", "value")], [("col2", ">", 10)]]
    result = convert_filters_to_sql(filters)
    assert "`col1` = 'value'" in result
    assert "`col2` > 10" in result
    assert "OR" in result


@pytest.mark.parametrize(
    "operator,expected_op,value",
    [
        ("==", "=", "value"),
        ("!=", "!=", "value"),
        ("<", "<", 10),
        (">", ">", 10),
        ("<=", "<=", 10),
        (">=", ">=", 10),
        ("in", "IN", [1, 2, 3]),
        ("not in", "NOT IN", [1, 2, 3]),
    ],
)
def test_all_operators(operator, expected_op, value):
    """Test all supported operators"""

    filters = [[("col1", operator, value)]]
    result = convert_filters_to_sql(filters)
    assert expected_op in result
    assert "`col1`" in result


@pytest.mark.parametrize(
    "value,expected_contains",
    [
        ("value", ("'", "`col`")),
        (10, ("`col` = 10",)),
        (True, ("TRUE",)),
        (False, ("FALSE",)),
        (None, ("NULL",)),
        ([1, 2, 3], ("`col` IN", "(1, 2, 3)")),
    ],
)
def test_value_types(value, expected_contains):
    """Test different value types"""

    # Determine appropriate operator for the value type
    if isinstance(value, (list, tuple)):
        op = "in"
    else:
        op = "=="

    result = convert_filters_to_sql([[("col", op, value)]])

    # All results should contain the column reference
    assert "`col`" in result

    # Check for expected substrings
    for expected in expected_contains:
        assert expected in result


def test_string_escaping():
    """Test single quote escaping in strings"""

    filters = [[("col", "==", "it's a test")]]
    result = convert_filters_to_sql(filters)
    assert "it''s a test" in result


def test_filter_errors():
    """Test all filter error cases in one function"""

    # Test unsupported operator
    filters = [[("col", "like", "value")]]
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        convert_filters_to_sql(filters, errors="raise")

    # Test malformed filter tuples
    with pytest.raises(ValueError, match="Each filter must be a 3-tuple"):
        convert_filters_to_sql([[("col", "==")]], errors="raise")  # Missing value

    with pytest.raises(ValueError, match="Each filter must be a 3-tuple"):
        convert_filters_to_sql(
            [[("col", "==", "val", "extra")]], errors="raise"
        )  # Too many elements

    # Test IN operator requires list/tuple value
    with pytest.raises(ValueError, match="Operator 'in' requires list or tuple value"):
        convert_filters_to_sql([[("col", "in", "value")]], errors="raise")

    # Test unsupported value types
    with pytest.raises(TypeError, match="Unsupported value type dict"):
        convert_filters_to_sql([[("col", "==", {"key": "value"})]], errors="raise")


def test_column_name_with_backtick():
    """Test column names with backticks are properly escaped"""

    filters = [[("col`name", "==", "value")]]
    result = convert_filters_to_sql(filters)
    assert "`col``name`" in result


@pytest.mark.parametrize(
    "filter_input",
    [
        None,
        [],
    ],
)
def test_none_and_empty_filter(filter_input):
    """Test that None and empty filter returns None"""

    assert convert_filters_to_sql(filter_input) is None


@pytest.mark.parametrize(
    "column_name",
    [
        "column with spaces",
        "col-name",
        "col.name",
        "select",  # SQL reserved keyword
    ],
)
def test_column_name_quoting(column_name):
    """Test column names with special characters are properly quoted"""

    filters = [[(column_name, ">", 10)]]
    result = convert_filters_to_sql(filters)
    assert f"`{column_name}`" in result


# ==================== Arrow Filter Tests ====================


@pytest.mark.parametrize(
    "op,value,expected_str",
    [
        ("=", 0, "(x == 0)"),
        ("==", 0, "(x == 0)"),
        ("!=", 0, "(x != 0)"),
        ("<", 5, "(x < 5)"),
        ("<=", 5, "(x <= 5)"),
        (">", 5, "(x > 5)"),
        (">=", 5, "(x >= 5)"),
    ],
)
def test_arrow_comparison_operators(op, value, expected_str):
    """Test comparison operator conversion"""
    field = ds.field("x")
    expr = _convert_single_filter(field, op, value)
    assert _clean_expr_str(expr) == expected_str


@pytest.mark.parametrize(
    "op,value",
    [
        ("in", [1, 2, 3]),
        ("not in", [1, 2, 3]),
    ],
)
def test_arrow_in_operators(op, value):
    """Test IN operator conversion"""
    field = ds.field("x")
    expr = _convert_single_filter(field, op, value)
    assert "is_in" in _clean_expr_str(expr)


@pytest.mark.parametrize(
    "filters",
    [
        None,
        [],
        [("x", "=", 0)],
        [[("x", "=", 0), ("y", ">", 5)]],
    ],
)
def test_valid_arrow_filters(filters):
    """Test arrow filter validation with valid inputs"""
    assert validate_filters(filters) is True


@pytest.mark.parametrize(
    "invalid_filters,error_type,error_pattern",
    [
        ("invalid", TypeError, "filters must be a list"),
        ([("not", "a", "list")], TypeError, "Each filter group must be a list"),
        ([("x", "=")], ValueError, "tuple of .*field, operator, value"),
        ([(123, "=", 0)], TypeError, "Filter field name must be a string"),
        ([("x", "invalid", 0)], ValueError, "Invalid operator 'invalid'"),
    ],
)
def test_invalid_arrow_filters(invalid_filters, error_type, error_pattern):
    """Test arrow filter validation with invalid inputs"""
    with pytest.raises(error_type, match=error_pattern):
        validate_filters(invalid_filters)


@pytest.mark.parametrize(
    "filters",
    [
        None,
        [],
    ],
)
def test_empty_arrow_filters(filters):
    """Test empty arrow filters conversion"""
    assert convert_filters_to_arrow_expression(filters) is None


def test_simple_equality_arrow_filter():
    """Test simple equality arrow filter conversion"""
    filters = [("x", "=", 0)]
    expr = convert_filters_to_arrow_expression(filters)
    assert _clean_expr_str(expr) == "(x == 0)"


def test_simple_list_format_implicit_and_arrow():
    """Test simple list format with implicit AND for arrow"""
    filters = [("x", "=", 0), ("y", ">", 5)]
    expr = convert_filters_to_arrow_expression(filters)
    cleaned_str = _clean_expr_str(expr)
    assert "and" in cleaned_str.lower()
    assert "(x == 0)" in cleaned_str
    assert "(y > 5)" in cleaned_str


def test_cnf_format_single_and_group_arrow():
    """Test CNF format with single AND group for arrow"""
    filters = [[("x", "=", 0), ("y", ">", 5)]]
    expr = convert_filters_to_arrow_expression(filters)
    cleaned_str = _clean_expr_str(expr)
    assert "and" in cleaned_str.lower()
    assert "(x == 0)" in cleaned_str
    assert "(y > 5)" in cleaned_str


def test_cnf_format_multiple_or_groups_arrow():
    """Test CNF format with multiple OR groups for arrow"""
    filters = [[("x", "=", 0)], [("y", ">", 5)]]
    expr = convert_filters_to_arrow_expression(filters)
    cleaned_str = _clean_expr_str(expr)
    assert "or" in cleaned_str.lower()
    assert "(x == 0)" in cleaned_str
    assert "(y > 5)" in cleaned_str


def test_complex_cnf_arrow_filter():
    """Test complex CNF arrow filter with both `AND` and `OR`"""
    filters = [[("x", "=", 0), ("y", ">", 5)], [("z", "=", "test")]]
    expr = convert_filters_to_arrow_expression(filters)
    cleaned_str = _clean_expr_str(expr)
    assert "or" in cleaned_str.lower()
    assert "and" in cleaned_str.lower()


def test_all_operators_in_arrow_filter():
    """Test all operators in a single arrow filter"""
    filters = [
        [("x", "=", 0)],
        [("y", "!=", 1)],
        [("z", "<", 10)],
        [("a", "<=", 10)],
        [("b", ">", 0)],
        [("c", ">=", 0)],
        [("d", "in", [1, 2])],
        [("e", "not in", [3, 4])],
    ]
    expr = convert_filters_to_arrow_expression(filters)
    assert expr is not None


# ==================== Flavor Tests ====================


@pytest.mark.parametrize(
    "value,operator,expected_result",
    [
        (
            datetime(2024, 3, 15, 10, 30, 45),
            ">",
            "`created_at` > CAST('2024-03-15T10:30:45' AS TIMESTAMP)",
        ),
        (date(2024, 3, 15), "==", "`birth_date` = CAST('2024-03-15' AS DATE)"),
    ],
)
def test_odps_flavor_with_datetime_and_date(value, operator, expected_result):
    """Test ODPS flavor with datetime and date values"""
    filters = [
        [
            (
                "created_at" if isinstance(value, datetime) else "birth_date",
                operator,
                value,
            )
        ]
    ]
    result = convert_filters_to_sql(filters, flavor="ODPS")
    assert result == expected_result


def test_odps_datetime_in_list():
    """Test ODPS flavor with datetime in IN clause"""
    dt1 = datetime(2024, 1, 1, 0, 0, 0)
    dt2 = datetime(2024, 12, 31, 23, 59, 59)
    filters = [[("created_at", "in", [dt1, dt2])]]
    result = convert_filters_to_sql(filters, flavor="ODPS")
    assert "CAST('2024-01-01T00:00:00' AS TIMESTAMP)" in result
    assert "CAST('2024-12-31T23:59:59' AS TIMESTAMP)" in result
    assert "IN" in result


@pytest.mark.parametrize(
    "value,flavor",
    [
        (date(2024, 3, 15), None),
        (date(2024, 3, 15), "OTHER"),
        (datetime(2024, 3, 15, 10, 30, 45), None),
    ],
)
def test_non_odps_flavor_with_date_and_datetime(value, flavor):
    """Test that non-ODPS flavor raises TypeError for date/datetime values"""
    filters = [[("col", "==", value)]]
    with pytest.raises(TypeError, match="Date/datetime type not supported"):
        convert_filters_to_sql(filters, flavor=flavor, errors="raise")


@pytest.mark.parametrize(
    "value,operator,expected_result",
    [
        ("value", "==", "`col` = 'value'"),
        (10, ">", "`col` > 10"),
        (True, "==", "`col` = TRUE"),
    ],
)
def test_flavor_none_with_supported_types(value, operator, expected_result):
    """Test that flavor=None works with supported types"""
    filters = [[("col", operator, value)]]
    assert convert_filters_to_sql(filters, flavor=None) == expected_result
