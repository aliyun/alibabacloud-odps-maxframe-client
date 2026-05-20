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

from datetime import date, datetime
from typing import Any, List, Optional, Tuple, Union


def convert_filters_to_sql(
    filters: Union[str, List[List[Tuple[str, str, Any]]]],
    flavor: Optional[str] = None,
    errors: str = "ignore",
) -> Optional[str]:
    """
    Convert filters to SQL WHERE clause string.

    This utility function is shared between MCSQL and DPE engines.

    Parameters
    ----------
    filters : Union[str, List[List[Tuple[str, str, Any]]]]
        Filter expression. Can be:
        - str: SQL WHERE clause (returned as-is)
        - List[List[Tuple[str, str, Any]]]: List format from read_parquet
    flavor : Optional[str]
        SQL flavor for date/datetime handling. When "ODPS", uses ODPS-specific
        date comparison syntax. Otherwise, raises for unsupported date types.
    errors : str
        Error handling mode. "ignore" (default) skips predicates that cannot be
        converted. "raise" raises exceptions for conversion errors.

    Returns
    -------
    Optional[str]
        SQL WHERE clause string, or None if filters is None or empty

    Raises
    ------
    ValueError
        If an unsupported operator is provided in list format (when errors="raise")
    ValueError
        If filter tuple is malformed (when errors="raise")
    ValueError
        If operator-value combination is invalid (e.g., IN with non-list value)
        (when errors="raise")
    TypeError
        If value type is not supported (when errors="raise")

    Examples
    --------
    >>> convert_filters_to_sql("col1 = 'value'")
    "col1 = 'value'"

    >>> convert_filters_to_sql([[('col1', '==', 'value'), ('col2', '>', 10)]])
    "(`col1` = 'value' AND `col2` > 10)"

    >>> convert_filters_to_sql([[('col1', '==', 'value')], [('col2', '>', 10)]])
    "(`col1` = 'value') OR (`col2` > 10)"
    """
    if filters is None:
        return None

    if isinstance(filters, str):
        # String format: use directly
        return filters

    # List format: convert to SQL
    return _convert_list_filters_to_sql(filters, flavor, errors)


def _convert_list_filters_to_sql(
    filters: List[List[Tuple[str, str, Any]]],
    flavor: Optional[str] = None,
    errors: str = "ignore",
) -> Optional[str]:
    """
    Convert list-of-lists filter format to SQL WHERE clause.

    Format:
    - Inner lists are ANDed together
    - Outer lists are ORed together
    - Each tuple: (column_name, operator, value)

    Parameters
    ----------
    filters : List[List[Tuple[str, str, Any]]]
        Nested list of filter tuples
    flavor : Optional[str]
        SQL flavor for date/datetime handling
    errors : str
        Error handling mode. "ignore" skips predicates that cannot be converted.
        "raise" raises exceptions for conversion errors.

    Returns
    -------
    Optional[str]
        SQL WHERE clause string, or None if all groups are empty

    Raises
    ------
    ValueError
        If filter tuple is malformed (when errors="raise")
    ValueError
        If operator-value combination is invalid (when errors="raise")
    TypeError
        If value type is not supported (when errors="raise")
    """
    if not filters:
        return None

    or_parts = []
    for and_group in filters:
        # Skip empty inner lists
        if not and_group:
            continue

        and_parts = []
        for filter_tuple in and_group:
            try:
                # Validate filter tuple structure
                if not isinstance(filter_tuple, tuple) or len(filter_tuple) != 3:
                    raise ValueError(
                        f"Each filter must be a 3-tuple (column, operator, value), "
                        f"got {filter_tuple!r}"
                    )
                col, op, val = filter_tuple

                # Validate operator-value combinations
                op_lower = op.lower().strip()
                if op_lower in ("in", "not in") and not isinstance(val, (list, tuple)):
                    raise ValueError(
                        f"Operator '{op}' requires list or tuple value, got {type(val).__name__}"
                    )

                # Convert operator
                sql_op = _convert_operator(op)
                # Convert value to SQL literal
                sql_val = _convert_value_to_sql_literal(val, flavor)
                # Wrap column name with backticks and escape existing backticks
                col_quoted = f"`{col.replace('`', '``')}`"
                # Build condition
                and_parts.append(f"{col_quoted} {sql_op} {sql_val}")
            except (ValueError, IndexError, TypeError):
                # Re-raise if errors mode is "raise"
                if errors == "raise":
                    raise
                # Otherwise skip filters that cannot be converted
                continue

        # Skip empty AND groups (all predicates were filtered out)
        if not and_parts:
            continue

        # AND the conditions in this group
        if len(and_parts) == 1:
            or_parts.append(and_parts[0])
        else:
            or_parts.append(f"({') AND ('.join(and_parts)})")

    # Return None if all groups were empty
    if not or_parts:
        return None

    # OR the groups
    if len(or_parts) == 1:
        return or_parts[0]
    else:
        return f"({' OR '.join(or_parts)})"


def _convert_operator(op: str) -> str:
    """
    Convert filter operator to SQL operator.

    Parameters
    ----------
    op : str
        Filter operator (e.g., '==', '!=', 'in')

    Returns
    -------
    str
        SQL operator (e.g., '=', '!=', 'IN')

    Raises
    ------
    ValueError
        If operator is not supported
    """
    op_map = {
        "==": "=",
        "=": "=",  # also accept single =
        "!=": "!=",
        "<>": "!=",  # alternative not equal
        "<": "<",
        ">": ">",
        "<=": "<=",
        ">=": ">=",
        "in": "IN",
        "not in": "NOT IN",
    }

    op_lower = op.lower().strip()
    if op_lower not in op_map:
        raise ValueError(
            f"Unsupported filter operator: {op}. "
            f"Supported operators: {list(op_map.keys())}"
        )

    return op_map[op_lower]


def _convert_value_to_sql_literal(val, flavor: Optional[str] = None) -> str:
    """
    Convert Python value to SQL literal string.

    Parameters
    ----------
    val
        Python value (str, int, float, bool, None, list, tuple, date, datetime)
    flavor : Optional[str]
        SQL flavor for date/datetime handling

    Returns
    -------
    str
        SQL literal string

    Raises
    ------
    TypeError
        If value type is not supported
    """
    if val is None:
        return "NULL"
    elif isinstance(val, str):
        # Escape single quotes in strings
        escaped = val.replace("'", "''")
        return f"'{escaped}'"
    elif isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    elif isinstance(val, (int, float)):
        return str(val)
    elif isinstance(val, (date, datetime)):
        # Handle date/datetime types
        if flavor == "ODPS":
            # ODPS-specific date comparison syntax
            if isinstance(val, datetime):
                # Convert datetime to ODPS timestamp format
                return f"CAST('{val.isoformat()}' AS TIMESTAMP)"
            else:
                # Convert date to ODPS date format
                return f"CAST('{val.isoformat()}' AS DATE)"
        else:
            # For other flavors, raise to skip the predicate
            raise TypeError(
                f"Date/datetime type not supported for SQL flavor '{flavor}'."
            )
    elif isinstance(val, (list, tuple)):
        # Handle IN clause values
        items = [_convert_value_to_sql_literal(item, flavor) for item in val]
        return f"({', '.join(items)})"
    else:
        raise TypeError(
            f"Unsupported value type {type(val).__name__} for filter. "
            "Supported types: str, int, float, bool, None, list, tuple,"
            " date, datetime"
        )


# Valid operators for arrow filter conditions
_VALID_ARROW_OPERATORS = set("= == != < <= > >= in".split()) | {"not in"}

# Python keywords that might appear as field names
_PYTHON_KEYWORDS = set(
    "not and or in is as if else elif for while with try except finally "
    "raise return yield import from class def lambda pass break continue "
    "assert del global nonlocal async await".split()
)


def validate_filters(filters: Union[List, None]) -> bool:
    """Validate arrow filter structure and content.

    Parameters
    ----------
    filters : list or None
        Filters to validate. Can be:
        - None: No filters
        - Empty list: No filters
        - List of tuples: [('field', 'op', value), ...] → implicit AND
        - CNF format: [[('field', 'op', value), ...], ...] → OR of ANDs

    Returns
    -------
    bool
        True if filters are valid

    Raises
    ------
    TypeError
        If filters structure is invalid
    ValueError
        If filter conditions are malformed
    """
    if filters is None:
        return True

    if not isinstance(filters, list):
        raise TypeError("filters must be a list")

    if not filters:
        return True

    # Auto-detect format and normalize to AND groups
    and_groups = _detect_filter_format(filters)

    # Validate each AND group
    for and_group in and_groups:
        if not isinstance(and_group, list):
            raise TypeError("Each filter group must be a list")

        if not and_group:
            raise ValueError("Filter group cannot be empty")

        for condition in and_group:
            if not isinstance(condition, (list, tuple)) or len(condition) != 3:
                raise ValueError(
                    "Each filter condition must be a tuple of (field, operator, value)"
                )

            field_name, op, value = condition

            if not isinstance(field_name, str):
                raise TypeError("Filter field name must be a string")

            if op not in _VALID_ARROW_OPERATORS:
                raise ValueError(
                    f"Invalid operator '{op}'. Must be one of {_VALID_ARROW_OPERATORS}"
                )

    return True


def _detect_filter_format(filters: List) -> List:
    """Auto-detect filter format and normalize to AND groups.

    Handles two formats:
    - Simple format: [('field', 'op', value), ...] → implicit AND
    - CNF format: [[('field', 'op', value), ...], ...] → OR of ANDs

    Parameters
    ----------
    filters : list
        Filters to detect format from

    Returns
    -------
    list
        List of AND groups (each group is a list of conditions)
    """
    # CNF format: first element is a list (not a tuple)
    if isinstance(filters[0], list):
        return filters

    # Simple format: first element is a tuple
    if isinstance(filters[0], tuple):
        # Check if this is actually a malformed CNF format where field name is a keyword
        # This helps provide better error messages
        if len(filters[0]) == 3 and filters[0][1] not in _VALID_ARROW_OPERATORS:
            field_name = filters[0][0]
            # If field name is a keyword, treat as CNF format (will fail with TypeError)
            if isinstance(field_name, str) and field_name in _PYTHON_KEYWORDS:
                return filters
        return [filters]

    # Fallback: treat as CNF format (will fail validation if invalid)
    return filters


def _convert_single_filter(field: Any, op: str, value: Any) -> Any:
    """Convert single filter condition to PyArrow Expression.

    Parameters
    ----------
    field : pyarrow.compute.Expression
        Field reference expression
    op : str
        Operator: '=', '==', '!=', '<', '<=', '>', '>=', 'in', 'not in'
    value : Any
        Value to compare against

    Returns
    -------
    pyarrow.compute.Expression
        Expression for the single condition
    """
    op_map = {
        "=": lambda f, v: f == v,
        "==": lambda f, v: f == v,
        "!=": lambda f, v: f != v,
        "<": lambda f, v: f < v,
        "<=": lambda f, v: f <= v,
        ">": lambda f, v: f > v,
        ">=": lambda f, v: f >= v,
        "in": lambda f, v: f.isin(v),
        "not in": lambda f, v: ~f.isin(v),
    }

    return op_map[op](field, value)


def convert_filters_to_arrow_expression(filters: Union[List, None]) -> Any:
    """Convert pandas-style filters to PyArrow Expression.

    Supports both formats:
    - List of tuples: [('x', '=', 0), ('y', '>', 5)] → implicit AND
    - CNF format: [[('x', '=', 0), ('y', '>', 5)], [('z', '=', 'test')]] → OR of ANDs

    Parameters
    ----------
    filters : list or None
        Filters to convert. None or empty list returns None.

    Returns
    -------
    pyarrow.compute.Expression or None
        PyArrow Expression object, or None if no filters

    Examples
    --------
    >>> import pyarrow.dataset as ds
    >>> filters = [('x', '=', 0), ('y', '>', 5)]
    >>> expr = convert_filters_to_arrow_expression(filters)
    >>> str(expr)
    '((x == 0) and (y > 5))'

    >>> filters = [[('x', '=', 0)], [('y', '>', 5)]]
    >>> expr = convert_filters_to_arrow_expression(filters)
    >>> str(expr)
    '((x == 0) or (y > 5))'
    """
    import pyarrow.dataset as pds

    if not filters:
        return None

    # Validate filters first
    validate_filters(filters)

    # Auto-detect format and normalize to AND groups
    and_groups = _detect_filter_format(filters)

    # Convert each AND group to expression
    and_expressions = []
    for and_group in and_groups:
        and_expr = None
        for field_name, op, value in and_group:
            field = pds.field(field_name)
            field_expr = _convert_single_filter(field, op, value)
            if and_expr is None:
                and_expr = field_expr
            else:
                and_expr = and_expr & field_expr
        and_expressions.append(and_expr)

    # OR all AND groups together
    if len(and_expressions) == 1:
        return and_expressions[0]

    result = and_expressions[0]
    for expr in and_expressions[1:]:
        result = result | expr
    return result
