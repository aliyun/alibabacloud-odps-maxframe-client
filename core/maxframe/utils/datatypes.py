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

import datetime
import io
import tokenize as pytokenize
import warnings
from typing import Union

import numpy as np
import pandas as pd

from maxframe.errors import OutputColumnMismatchError, OutputDtypeMismatchError
from maxframe.lib.dtypes_extension import ArrowBlobType, ArrowDtype, ExternalBlobDtype

try:
    import pyarrow as pa
except ImportError:
    pa = None


def make_dtype(
    dtype: Union[np.dtype, pd.api.extensions.ExtensionDtype],
    return_pd_dtype: bool = True,
):
    if dtype is None:
        return None
    elif return_pd_dtype and (
        (isinstance(dtype, str) and dtype == "category")
        or pd.api.types.is_extension_array_dtype(dtype)
    ):
        # return string dtype directly as legacy python version
        #  does not support ExtensionDtype
        return dtype
    elif isinstance(dtype, str) and dtype.lower() in ("blob", "maxframe.blob"):
        # Handle blob type strings
        return ExternalBlobDtype()
    elif dtype is pd.Timestamp or dtype is datetime.datetime:
        return np.dtype("datetime64[ns]")
    elif dtype is pd.Timedelta or dtype is datetime.timedelta:
        return np.dtype("timedelta64[ns]")
    else:
        try:
            ret = pd.api.types.pandas_dtype(dtype)
            if not return_pd_dtype and pd.api.types.is_extension_array_dtype(dtype):
                ret = np.dtype(ret.type)
            return ret
        except TypeError:
            return np.dtype("O")


def make_dtypes(
    dtypes: Union[
        list, dict, str, np.dtype, pd.Series, pd.api.extensions.ExtensionDtype
    ],
    make_series: bool = True,
):
    if dtypes is None:
        return None
    elif isinstance(dtypes, np.dtype):
        return dtypes
    elif isinstance(dtypes, list):
        val = [make_dtype(dt) for dt in dtypes]
        return val if not make_series else pd.Series(val)
    elif isinstance(dtypes, dict):
        val = {k: make_dtype(v) for k, v in dtypes.items()}
        return val if not make_series else pd.Series(val)
    elif isinstance(dtypes, pd.Series):
        return dtypes.map(make_dtype)
    else:
        return make_dtype(dtypes)


_arrow_type_constructors = {}
if pa:
    _arrow_type_constructors = {
        "bool": pa.bool_,
        "list": lambda x: pa.list_(dict(x)["item"]),
        "map": lambda x: pa.map_(*x),
        "struct": pa.struct,
        "fixed_size_binary": pa.binary,
        "halffloat": pa.float16,
        "float": pa.float32,
        "double": pa.float64,
        "decimal": pa.decimal128,
        # repr() of date32 and date64 has `day` or `ms`
        #  which is not needed in constructors
        "date32": lambda *_: pa.date32(),
        "date64": lambda *_: pa.date64(),
    }
    _plain_arrow_types = """
    null
    int8 int16 int32 int64
    uint8 uint16 uint32 uint64
    float16 float32 float64
    decimal128 decimal256
    string utf8 binary large_string
    time32 time64 duration timestamp
    month_day_nano_interval
    """
    for _type_name in _plain_arrow_types.split():
        try:
            _arrow_type_constructors[_type_name] = getattr(pa, _type_name)
        except AttributeError:  # pragma: no cover
            pass
    _arrow_type_constructors[ArrowBlobType.type_str] = ArrowBlobType


def arrow_type_from_str(type_str: str) -> pa.DataType:
    """
    Convert arrow type representations (for inst., list<item: int64>)
    into arrow DataType instances
    """
    # enable consecutive brackets to be tokenized
    type_str = type_str.replace("<", "< ").replace(">", " >")
    token_iter = pytokenize.tokenize(io.BytesIO(type_str.encode()).readline)
    value_stack, op_stack = [], []

    def _pop_make_type(with_args: bool = False, combined: bool = True):
        """
        Pops tops of value stacks, creates a DataType instance and push back

        Parameters
        ----------
            with_args: bool
                if True, will contain next item (parameter list) in
                the value stack as parameters
            combined: bool
                if True, will use first element of the top of the value stack
                in DataType constructors
        """
        args = () if not with_args else (value_stack.pop(-1),)
        if not combined:
            args = args[0]
        type_name = value_stack.pop(-1)
        if isinstance(type_name, pa.DataType):
            value_stack.append(type_name)
        elif type_name in _arrow_type_constructors:
            value_stack.append(_arrow_type_constructors[type_name](*args))
        else:  # pragma: no cover
            value_stack.append(type_name)

    def _pop_make_struct_field():
        """parameterized sub-types need to be represented as tuples"""
        nonlocal value_stack

        op_stack.pop(-1)
        if isinstance(value_stack[-1], str) and value_stack[-1].lower() in (
            "null",
            "not null",
        ):
            values = value_stack[-3:]
            value_stack = value_stack[:-3]
            values[-1] = values[-1] == "null"
        else:
            values = value_stack[-2:]
            value_stack = value_stack[:-2]
        value_stack.append(tuple(values))

    try:
        for token in token_iter:
            if token.type == pytokenize.OP:
                if token.string in (".", ":"):
                    op_stack.append(token.string)
                elif token.string == ",":
                    # gather previous sub-types
                    if op_stack[-1] in ("<", ":"):
                        _pop_make_type()
                    if op_stack[-1] == ":":
                        _pop_make_struct_field()

                    # put generated item into the parameter list
                    val = value_stack.pop(-1)
                    value_stack[-1].append(val)
                elif token.string in ("<", "[", "("):
                    # pushes an empty parameter list for future use
                    value_stack.append([])
                    op_stack.append(token.string)
                elif token.string in (")", "]"):
                    # put generated item into the parameter list
                    val = value_stack.pop(-1)
                    value_stack[-1].append(val)
                    # make DataType (i.e., fixed_size_binary / decimal) given args
                    _pop_make_type(with_args=True, combined=False)
                    op_stack.pop(-1)
                elif token.string == ">":
                    _pop_make_type()
                    if op_stack[-1] == ":":
                        _pop_make_struct_field()

                    # put generated item into the parameter list
                    val = value_stack.pop(-1)
                    value_stack[-1].append(val)
                    # make DataType (i.e., list / map / struct) given args
                    _pop_make_type(with_args=True)
                    op_stack.pop(-1)
            elif token.type == pytokenize.NAME:
                if op_stack and op_stack[-1] == ".":
                    # handle extension names like "maxframe.blob"
                    op_stack.pop(-1)
                    if value_stack:
                        value_stack[-1] = f"{value_stack[-1]}.{token.string}"
                elif value_stack and value_stack[-1] == "not":
                    value_stack[-1] += " " + token.string
                else:
                    value_stack.append(token.string)
            elif token.type == pytokenize.NUMBER:
                value_stack.append(int(token.string))
            elif token.type == pytokenize.ENDMARKER:
                # make final type
                _pop_make_type()
    except Exception as ex:
        raise ValueError(
            f"Unexpected error occurred when parsing type {type_str}: {ex}"
        ) from None
    if len(value_stack) > 1 or len(value_stack) == 0:
        raise ValueError(f"Cannot parse type {type_str}")
    elif isinstance(value_stack[-1], str):
        try:
            return _arrow_type_constructors[value_stack[-1]]()
        except (KeyError, TypeError):
            raise ValueError(f"Cannot parse type {type_str}")
    return value_stack[-1]


def wrap_arrow_dtype(arrow_type):
    if arrow_type == pa.string():
        try:
            return pd.StringDtype("pyarrow")
        except ImportError:  # pragma: no cover
            # pyarrow might not support ArrowDtype here, fallback
            pass
    return ArrowDtype(arrow_type)


def is_arrow_dtype_supported():
    return not getattr(ArrowDtype, "_is_fake", False)


def is_string_dtype(arr_or_dtype) -> bool:
    if isinstance(arr_or_dtype, ArrowDtype):
        return pa.types.is_string(arr_or_dtype.pyarrow_dtype)

    return pd.api.types.is_string_dtype(arr_or_dtype)


def is_bool_dtype(arr_or_dtype) -> bool:
    if isinstance(arr_or_dtype, ArrowDtype):
        return pa.types.is_boolean(arr_or_dtype.pyarrow_dtype)

    return pd.api.types.is_bool_dtype(arr_or_dtype)


def is_datetime64_dtype(arr_or_dtype) -> bool:
    if isinstance(arr_or_dtype, ArrowDtype):
        return pa.types.is_timestamp(arr_or_dtype.pyarrow_dtype)

    return pd.api.types.is_datetime64_any_dtype(arr_or_dtype)


def check_dtype_compatibility(
    actual_dtype: np.dtype,
    expected_dtype: np.dtype,
    column_name: str = None,
) -> None:
    """
    Check if actual dtype can be safely cast to expected dtype.

    Parameters
    ----------
    actual_dtype : np.dtype
        Actual dtype from result
    expected_dtype : np.dtype
        Expected dtype
    column_name : str, optional
        Column name for error messages

    Raises
    ------
    OutputDtypeMismatchError
        When dtype cannot be cast to expected dtype
    """

    # Helper function to check if a dtype is a blob type
    def _is_blob_type(dtype):
        return isinstance(dtype, ExternalBlobDtype) or (
            isinstance(dtype, ArrowDtype)
            and isinstance(dtype.pyarrow_dtype, ArrowBlobType)
        )

    # Handle blob types (ExternalBlobDtype or ArrowBlobType)
    if _is_blob_type(actual_dtype) or _is_blob_type(expected_dtype):
        # Blob types are only compatible with other blob types
        if _is_blob_type(actual_dtype) and _is_blob_type(expected_dtype):
            return
        col_info = f" for column '{column_name}'" if column_name else ""
        raise OutputDtypeMismatchError(
            column=column_name or "unknown",
            actual_dtype=actual_dtype,
            expected_dtype=expected_dtype,
            can_cast=False,
            extra_msg=f"Cannot cast {actual_dtype} to {expected_dtype}{col_info}",
        )

    # Handle ArrowDtype
    if isinstance(actual_dtype, ArrowDtype) or isinstance(expected_dtype, ArrowDtype):
        # For Arrow dtypes, we check if they're the same type
        if actual_dtype == expected_dtype:
            return
        # Try to determine if cast is possible
        try:
            if isinstance(actual_dtype, ArrowDtype) and isinstance(
                expected_dtype, ArrowDtype
            ):
                # Both are Arrow types
                can_cast = pa.types.can_cast(
                    actual_dtype.pyarrow_dtype, expected_dtype.pyarrow_dtype
                )
            else:
                # Mixed Arrow and non-Arrow, be conservative
                can_cast = False

            if can_cast:
                return
            else:
                col_info = f" for column '{column_name}'" if column_name else ""
                error_msg = f"Cannot cast {actual_dtype} to {expected_dtype}{col_info}"
                raise OutputDtypeMismatchError(
                    column=column_name or "unknown",
                    actual_dtype=actual_dtype,
                    expected_dtype=expected_dtype,
                    can_cast=False,
                    extra_msg=error_msg,
                )
        except OutputDtypeMismatchError:
            raise
        except Exception:
            # If we can't determine, be conservative
            raise OutputDtypeMismatchError(
                column=column_name or "unknown",
                actual_dtype=actual_dtype,
                expected_dtype=expected_dtype,
                can_cast=False,
                extra_msg=f"Cannot determine castability for {actual_dtype}",
            ) from None

    # Handle regular numpy dtypes
    # Special check for float to integer conversion - this should not be allowed
    # as it causes silent truncation and data loss
    float_dtypes = {"float16", "float32", "float64", "float128"}
    integer_dtypes = {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    actual_dtype_str = str(actual_dtype)
    expected_dtype_str = str(expected_dtype)

    if actual_dtype_str in float_dtypes and expected_dtype_str in integer_dtypes:
        col_info = f" for column '{column_name}'" if column_name else ""
        raise OutputDtypeMismatchError(
            column=column_name or "unknown",
            actual_dtype=actual_dtype,
            expected_dtype=expected_dtype,
            can_cast=False,
            extra_msg=(
                f"Cannot cast float dtype {actual_dtype} to integer dtype "
                f"{expected_dtype}{col_info}. Float to integer conversion "
                "causes truncation and potential data loss. Please ensure "
                "your function returns the correct integer dtype."
            ),
        )

    try:
        # Check if we can safely cast
        if np.can_cast(actual_dtype, expected_dtype, casting="safe") or np.can_cast(
            actual_dtype, expected_dtype, casting="same_kind"
        ):
            # Can cast with potential precision loss
            return
        else:
            col_info = f" for column '{column_name}'" if column_name else ""
            error_msg = f"Cannot cast {actual_dtype} to {expected_dtype}{col_info}"
            raise OutputDtypeMismatchError(
                column=column_name or "unknown",
                actual_dtype=actual_dtype,
                expected_dtype=expected_dtype,
                can_cast=False,
                extra_msg=error_msg,
            )
    except OutputDtypeMismatchError:
        raise
    except (TypeError, ValueError):
        # If numpy can't determine castability, check if types are the same
        if actual_dtype == expected_dtype:
            return
        col_info = f" for column '{column_name}'" if column_name else ""
        raise OutputDtypeMismatchError(
            column=column_name or "unknown",
            actual_dtype=actual_dtype,
            expected_dtype=expected_dtype,
            can_cast=False,
            extra_msg=f"Cannot cast {actual_dtype} to {expected_dtype}{col_info}",
        ) from None


def validate_and_align_output(
    result: Union[pd.DataFrame, pd.Series],
    expected_dtypes: Union[pd.Series, np.dtype],
    check_output_dtypes: str = None,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Validate and align output from user function.

    Parameters
    ----------
    result : DataFrame or Series
        Result from user function
    expected_dtypes : Series or dtype
        Expected dtypes for output
    check_output_dtypes : str, default None
        Validation mode: 'ignore', 'warns', or 'raises'
        - 'ignore': No validation performed
        - 'warns': Validate and show warnings on mismatch (default when None)
        - 'raises': Validate and raise errors on mismatch

    Returns
    -------
    DataFrame or Series
        Aligned result with columns reordered and dtypes cast

    Raises
    ------
    OutputColumnMismatchError
        When check_output_dtypes='raises' and columns don't match
    OutputDtypeMismatchError
        When check_output_dtypes='raises' and dtype cannot be cast
    """
    # Normalize check_output_dtypes parameter
    # If None or False or 'ignore', skip all validation and casting
    if (
        check_output_dtypes is None
        or check_output_dtypes is False
        or check_output_dtypes == "ignore"
    ):
        return result

    # If expected_dtypes is None, skip validation
    if expected_dtypes is None:
        return result

    # Determine if we should raise or warn
    should_raise = check_output_dtypes == "raises"

    # Handle Series case
    if isinstance(result, pd.Series):
        if isinstance(expected_dtypes, pd.Series):
            # Expected dtypes is a Series (from DataFrame.dtypes), take the first value
            expected_dtype = (
                expected_dtypes.iloc[0] if len(expected_dtypes) > 0 else None
            )
        else:
            expected_dtype = expected_dtypes

        if expected_dtype is None:
            return result

        actual_dtype = result.dtype
        try:
            check_dtype_compatibility(actual_dtype, expected_dtype, result.name)

            # If check passed, try to cast if dtypes differ
            if actual_dtype != expected_dtype:
                try:
                    result = result.astype(expected_dtype)
                except (ValueError, TypeError):
                    error = OutputDtypeMismatchError(
                        result.name or "series",
                        actual_dtype,
                        expected_dtype,
                        can_cast=False,
                    )
                    if should_raise:
                        raise error
                    else:
                        warnings.warn(str(error), FutureWarning)
        except OutputDtypeMismatchError as e:
            if should_raise:
                raise
            else:
                warnings.warn(str(e), FutureWarning)

        return result

    # Handle DataFrame case
    if not isinstance(result, pd.DataFrame):
        return result

    # Get expected columns and dtypes
    if isinstance(expected_dtypes, pd.Series):
        expected_columns = expected_dtypes.index.tolist()
        expected_dtypes_dict = expected_dtypes.to_dict()
    else:
        # Single dtype for all columns
        expected_columns = result.columns.tolist()
        expected_dtypes_dict = {col: expected_dtypes for col in expected_columns}

    # Check for missing and extra columns
    result_columns = result.columns.tolist()
    missing_cols = [col for col in expected_columns if col not in result_columns]
    extra_cols = [col for col in result_columns if col not in expected_columns]

    if missing_cols or extra_cols:
        if should_raise:
            raise OutputColumnMismatchError(
                missing_cols=missing_cols, extra_cols=extra_cols
            )
        else:
            msg_parts = []
            if missing_cols:
                msg_parts.append(f"Missing columns: {missing_cols}")
            if extra_cols:
                msg_parts.append(f"Unexpected columns: {extra_cols}")
            warnings.warn("; ".join(msg_parts), FutureWarning)

    # Reorder columns to match expected order (only include columns that exist)
    aligned_columns = [col for col in expected_columns if col in result_columns]
    if aligned_columns:
        result = result[aligned_columns]

    # Check and cast dtypes for each column
    for col in aligned_columns:
        actual_dtype = result[col].dtype
        expected_dtype = expected_dtypes_dict.get(col)

        if expected_dtype is None:
            continue

        try:
            check_dtype_compatibility(actual_dtype, expected_dtype, col)

            # If check passed, try to cast if dtypes differ
            if actual_dtype != expected_dtype:
                try:
                    result[col] = result[col].astype(expected_dtype)
                except (ValueError, TypeError):
                    error = OutputDtypeMismatchError(
                        col, actual_dtype, expected_dtype, can_cast=False
                    )
                    if should_raise:
                        raise error
                    else:
                        warnings.warn(str(error), FutureWarning)
        except OutputDtypeMismatchError as e:
            if should_raise:
                raise
            else:
                warnings.warn(str(e), FutureWarning)

    return result
