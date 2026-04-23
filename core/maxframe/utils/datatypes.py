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
from typing import Union

import numpy as np
import pandas as pd

from ..lib.dtypes_extension import ArrowBlobType, ArrowDtype

try:
    import pyarrow as pa
except ImportError:
    pa = None


def make_dtype(dtype: Union[np.dtype, pd.api.extensions.ExtensionDtype]):
    if dtype is None:
        return None
    elif (
        isinstance(dtype, str) and dtype == "category"
    ) or pd.api.types.is_extension_array_dtype(dtype):
        # return string dtype directly as legacy python version
        #  does not support ExtensionDtype
        return dtype
    elif dtype is pd.Timestamp or dtype is datetime.datetime:
        return np.dtype("datetime64[ns]")
    elif dtype is pd.Timedelta or dtype is datetime.timedelta:
        return np.dtype("timedelta64[ns]")
    else:
        try:
            return pd.api.types.pandas_dtype(dtype)
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
    if len(value_stack) > 1:
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
