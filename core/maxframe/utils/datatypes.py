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
from maxframe.lib.dtypes_extension.blob import is_polars_blob_type

try:
    import pyarrow as pa
except ImportError:
    pa = None

try:
    import polars as pl
except ImportError:
    pl = None


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


def to_pandas_dtype(dt):
    """Resolve a dtype to a pandas dtype, accepting polars/pyarrow types.

    ``pd.api.types.pandas_dtype`` does not recognize polars dtype classes
    (``pl.Int64``) or instances (``pl.Decimal``, ``pl.List``); a 0-row
    polars Series round-trip yields the exact pandas equivalent, and
    pyarrow types map via ``pd.ArrowDtype``.
    """
    try:
        return pd.api.types.pandas_dtype(dt)
    except TypeError:
        pass
    if pl is not None:
        is_pl = isinstance(dt, pl.DataType) or (
            isinstance(dt, type) and issubclass(dt, pl.DataType)
        )
        if is_pl:
            # polars Decimal -> arrow -> pandas crashes under pyarrow < 2.0
            # (C ABI can't import the decimal array), and the value-dependent
            # success of the round-trip makes it unreliable.  A polars
            # Decimal column converts to pandas ``object`` regardless of
            # precision/scale, so return that directly.
            if dt is pl.Decimal or isinstance(dt, pl.Decimal):
                return np.dtype(object)
            return pl.Series("__c", [], dtype=dt).to_pandas().dtype
    if pa is not None and isinstance(dt, pa.DataType):
        return pd.ArrowDtype(dt)
    raise TypeError(f"cannot convert {dt!r} to a pandas dtype")


def _dtype_equal(a, b) -> bool:
    """Safely compare two dtypes for equality.

    Under old pyarrow (FakeArrowDtype, pandas<1.5) a numpy dtype on the left
    of ``==``/``!=`` tries to construct the ArrowDtype as a numpy dtype and
    raises ``TypeError: data type not understood``; the ArrowDtype-side
    ``__eq__`` handles the comparison safely, so fall back to it.
    """
    try:
        return a == b
    except TypeError:
        try:
            return b == a
        except TypeError:
            return False


def _make_dictionary_type(param_list):
    """Build a dictionary type from pyarrow's keyword form."""
    positional = []
    kwargs = {}
    for item in param_list:
        if isinstance(item, tuple) and len(item) == 3 and item[0] == "kw":
            kwargs[item[1]] = item[2]
        else:
            positional.append(item)
    if positional:
        indices = positional[0]
        values = positional[1] if len(positional) > 1 else kwargs.get("values")
    else:
        values = kwargs.get("values")
        indices = kwargs.get("indices")
    ordered = bool(kwargs.get("ordered", 0))
    return pa.dictionary(indices, values, ordered=ordered)


_arrow_type_constructors = {}
if pa:
    _arrow_type_constructors = {
        "bool": pa.bool_,
        "list": lambda x: pa.list_(dict(x)["item"]),
        "large_list": lambda x: pa.large_list(dict(x)["item"]),
        "map": lambda x: pa.map_(*x),
        "struct": pa.struct,
        "dictionary": _make_dictionary_type,
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

try:
    from maxframe.lib.dtypes_extension.vector import ArrowVectorType
except ImportError:
    ArrowVectorType = None

if ArrowVectorType is not None:
    # Register constructor for "maxframe.vector(element_type, dimension)"
    # The parser tokenizes "maxframe.vector" as a dotted name via the "." handling
    # When called with parenthesized args like maxframe.vector(float,128),
    # combined=False means args are unpacked with *args
    def _make_arrow_vector_type(*args):
        element_type = args[0]
        if not isinstance(element_type, pa.DataType):
            element_type = pa.type_for_alias(str(element_type))
        dimension = args[1] if isinstance(args[1], int) else int(args[1])
        return ArrowVectorType(element_type, dimension)

    _arrow_type_constructors["maxframe.vector"] = _make_arrow_vector_type


def arrow_type_from_str(type_str: str) -> pa.DataType:
    """
    Convert arrow type representations (for inst., list<item: int64>)
    into arrow DataType instances
    """
    # enable consecutive brackets to be tokenized
    type_str = type_str.replace("<", "< ").replace(">", " >")
    token_iter = pytokenize.tokenize(io.BytesIO(type_str.encode()).readline)
    value_stack, op_stack = [], []
    # pyarrow stringifies some types with ``key=value`` keyword syntax
    # (e.g. ``timestamp[us, tz=UTC]``, ``dictionary<values=..., ...>``). The
    # keyword name is stashed here when ``=`` is seen, and the completed value
    # is wrapped as a ("kw", name, value) entry, so keyword syntax is handled
    # at any nesting depth rather than only at the top level.
    # A stack of ``(name, depth)`` entries, one per in-flight ``key=value``
    # argument. ``depth`` is the owning bracket group's op-stack depth, so a
    # keyword whose value is itself a complex/nested type (e.g.
    # ``values=list<item: int64>``) is only wrapped when *its own* bracket
    # group completes -- never by an inner group's value-completion.
    pending_kw = []

    def _wrap_value(val):
        """Wrap *val* as a ("kw", name, value) entry if a keyword at the
        current bracket depth is pending; otherwise return it unchanged."""
        depth = len(op_stack)
        for i in range(len(pending_kw) - 1, -1, -1):
            if pending_kw[i][1] == depth:
                name, _ = pending_kw.pop(i)
                return ("kw", name, val)
        return val

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
        kwargs = {}
        if with_args:
            param_list = value_stack.pop(-1)
            if combined:
                # ``<>`` groups (list / struct / map / dictionary) receive
                # the whole parameter list; the dictionary constructor
                # extracts its own keyword args from it.
                args = [param_list]
            else:
                # ``[]``/``()`` groups spread positional args and separate
                # ``key=value`` kwargs (e.g. timestamp[unit, tz=...]).
                args = []
                for item in param_list:
                    if isinstance(item, tuple) and len(item) == 3 and item[0] == "kw":
                        kwargs[item[1]] = item[2]
                    else:
                        args.append(item)
        else:
            args = ()
        type_name = value_stack.pop(-1)
        if isinstance(type_name, pa.DataType):
            value_stack.append(type_name)
        elif type_name in _arrow_type_constructors:
            value_stack.append(_arrow_type_constructors[type_name](*args, **kwargs))
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
                if token.string in (".", ":", "/", "-"):
                    op_stack.append(token.string)
                elif token.string == "=":
                    # Begin a keyword argument for the current bracket group:
                    # the preceding NAME is the keyword; the value is wrapped
                    # when this group's next item completes (matched by depth).
                    pending_kw.append((value_stack.pop(-1), len(op_stack)))
                elif token.string == ",":
                    # gather previous sub-types
                    if op_stack[-1] in ("<", ":"):
                        _pop_make_type()
                    if op_stack[-1] == ":":
                        _pop_make_struct_field()

                    # put generated item into the parameter list
                    val = _wrap_value(value_stack.pop(-1))
                    value_stack[-1].append(val)
                elif token.string in ("<", "[", "("):
                    # pushes an empty parameter list for future use
                    value_stack.append([])
                    op_stack.append(token.string)
                elif token.string in (")", "]"):
                    # put generated item into the parameter list
                    val = _wrap_value(value_stack.pop(-1))
                    value_stack[-1].append(val)
                    # make DataType (i.e., fixed_size_binary / decimal) given args
                    _pop_make_type(with_args=True, combined=False)
                    op_stack.pop(-1)
                elif token.string == ">":
                    _pop_make_type()
                    if op_stack[-1] == ":":
                        _pop_make_struct_field()

                    # put generated item into the parameter list
                    val = _wrap_value(value_stack.pop(-1))
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
                elif op_stack and op_stack[-1] == "/":
                    # handle slash-separated names like IANA timezone
                    # identifiers ("America/New_York") inside keyword values
                    op_stack.pop(-1)
                    if value_stack:
                        value_stack[-1] = f"{value_stack[-1]}/{token.string}"
                elif op_stack and op_stack[-1] == "-":
                    # handle hyphen-separated segments inside IANA timezone
                    # identifiers ("America/Port-au-Prince") within keyword
                    # values. ``-`` never appears as an arithmetic operator in
                    # Arrow type reprs (no negative literals), so it is safe to
                    # treat it as a name joiner here, mirroring the ``/`` case.
                    op_stack.pop(-1)
                    if value_stack:
                        value_stack[-1] = f"{value_stack[-1]}-{token.string}"
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
        except (ImportError, TypeError):  # pragma: no cover
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


def is_blob_dtype(dtype) -> bool:
    """Check whether *dtype* is (or nests) a blob column type.

    Recognizes the pandas-side ``ExternalBlobDtype`` and the polars-side
    blob object-container extension type, and recurses into nested
    container dtypes — list/array element types, map value types, and
    struct field types — so blobs hidden inside containers are detected.
    Detection is duck-typed on the nested-type attributes (``inner`` /
    ``value_dtype`` / ``fields``) to stay compatible across polars
    versions (e.g. ``pl.Map`` was renamed to ``pl.Dict`` in 1.41).
    """
    if dtype is None:
        return False
    if isinstance(dtype, ExternalBlobDtype):
        return True
    try:
        if is_polars_blob_type(dtype):
            return True
    except (TypeError, AttributeError):
        # Non-polars / duck-typed dtype. SystemError from an incompatible
        # polars version must propagate, not silently disable detection.
        pass
    inner = getattr(dtype, "inner", None)
    if inner is not None:  # polars List / Array
        return is_blob_dtype(inner)
    value_dtype = getattr(dtype, "value_dtype", None)
    if value_dtype is not None:  # polars Map / Dict
        return is_blob_dtype(value_dtype)
    fields = getattr(dtype, "fields", None)
    if fields:  # polars Struct
        return any(is_blob_dtype(getattr(f, "dtype", None)) for f in fields)
    return False


def _is_string_dtype_kind(dtype):
    """True for pandas ``StringDtype``, Arrow string/large_string, and
    NumPy unicode (``<U``) dtypes.

    These are semantically the same "string" type and must be treated as
    compatible: the DPE executor round-trips results through polars, which
    materializes a ``StringDtype`` column as ``large_string[pyarrow]``, so a
    declared ``StringDtype``/``string[pyarrow]`` must validate against an
    actual ``large_string[pyarrow]`` (and vice versa) instead of falling back
    to the empty-numpy-string (``<U0``) equivalence.
    """
    if isinstance(dtype, ArrowDtype):
        pa_type = dtype.pyarrow_dtype
        return pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type)
    if isinstance(dtype, pd.StringDtype):
        return True
    if not isinstance(dtype, np.dtype):
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            return False
    return np.issubdtype(dtype, np.str_)


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

    # A pandas ``object`` column is the canonical real-world representation
    # of a plain-Python-string column (``pd.DataFrame({"cols": ["a"]})``
    # has dtype ``object``, not Arrow/String).  A UDF returning strings must
    # therefore validate against a *declared* string dtype — Arrow
    # ``string``/``large_string``, pandas ``StringDtype``, or NumPy unicode
    # (a bare ``str`` annotation normalizes to ``<U0``) — or every
    # string-returning UDF raises ``OutputDtypeMismatchError``.  Only this
    # direction is special-cased; a declared ``object`` schema stays subject
    # to the regular rules below.
    if pd.api.types.is_object_dtype(actual_dtype) and _is_string_dtype_kind(
        expected_dtype
    ):
        return

    # pandas ``str`` (StringDtype with na_value=nan, inferred by default in
    # pandas>=3) and ``string`` (StringDtype with na_value=<NA>) are the same
    # logical string dtype with different NA representations, as are the
    # numpy-unicode variants.  The Arrow branch below already treats all
    # string variants as compatible; mirror that rule here so pure-pandas
    # string pairs don't fall through to ``np.can_cast``, which cannot
    # represent StringDtype at all (``TypeError: data type not understood``).
    if _is_string_dtype_kind(actual_dtype) and _is_string_dtype_kind(expected_dtype):
        return

    # Handle ArrowDtype
    if isinstance(actual_dtype, ArrowDtype) or isinstance(expected_dtype, ArrowDtype):
        # For Arrow dtypes, we check if they're the same type.  Use the
        # safe helper: under old pyarrow (FakeArrowDtype, pandas<1.5) a
        # numpy dtype on the left of ``==`` raises ``TypeError: data type
        # not understood``; the ArrowDtype-side ``__eq__`` is safe.
        if _dtype_equal(actual_dtype, expected_dtype):
            return
        # All string variants (StringDtype / string[pyarrow] /
        # large_string[pyarrow]) are compatible; see _is_string_dtype_kind.
        if _is_string_dtype_kind(actual_dtype) and _is_string_dtype_kind(
            expected_dtype
        ):
            return
        # Decimal precision/scale has no faithful NumPy representation (both
        # sides degrade to ``object``), so the NumPy compatibility check below
        # would accept a lossy cast between different decimal parameters
        # (e.g. decimal128(10, 2) -> decimal128(10, 1) silently rounds values).
        # Only reject when BOTH sides are Decimal with unequal parameters — a
        # Decimal result declared ``object`` is lossless and must fall through
        # to the normal compatibility path.
        if pa is not None and all(
            isinstance(dt, ArrowDtype) and pa.types.is_decimal(dt.pyarrow_dtype)
            for dt in (actual_dtype, expected_dtype)
        ):
            col_info = f" for column '{column_name}'" if column_name else ""
            raise OutputDtypeMismatchError(
                column=column_name or "unknown",
                actual_dtype=actual_dtype,
                expected_dtype=expected_dtype,
                can_cast=False,
                extra_msg=(
                    f"Cannot cast {actual_dtype} to {expected_dtype}{col_info}: "
                    "decimal precision/scale differ and casting is lossy"
                ),
            )
        # Try to determine if cast is possible
        try:
            # Convert ArrowDtype side(s) to their numpy equivalents and fall
            # through to the regular ``np.can_cast`` logic below.  pyarrow has
            # no public ``can_cast`` helper, so this is the same path for both
            # the both-Arrow and mixed-Arrow/non-Arrow cases — keeping all
            # representations of the same logical dtype (``np.dtype("int64")``,
            # ``pl.Int64``, ``pa.int64()``, ``int64[pyarrow]``) equivalent.
            if isinstance(actual_dtype, ArrowDtype):
                actual_dtype = actual_dtype.numpy_dtype
            if isinstance(expected_dtype, ArrowDtype):
                expected_dtype = expected_dtype.numpy_dtype
        except Exception:
            raise OutputDtypeMismatchError(
                column=column_name or "unknown",
                actual_dtype=actual_dtype,
                expected_dtype=expected_dtype,
                can_cast=False,
                extra_msg=f"Cannot determine numpy equivalent for {actual_dtype}",
            ) from None
        # Fall through to regular numpy dtype handling below

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
        # If numpy can't determine castability, check if types are the same.
        # Use the safe helper: under old pandas/pyarrow a numpy dtype on the
        # left of ``==`` (e.g. ``np.dtype("int64") == StringDtype()``) raises
        # ``TypeError: data type not understood`` inside this handler.
        if _dtype_equal(actual_dtype, expected_dtype):
            return
        col_info = f" for column '{column_name}'" if column_name else ""
        raise OutputDtypeMismatchError(
            column=column_name or "unknown",
            actual_dtype=actual_dtype,
            expected_dtype=expected_dtype,
            can_cast=False,
            extra_msg=f"Cannot cast {actual_dtype} to {expected_dtype}{col_info}",
        ) from None


def _validate_object_string_values(result, expected_dtype, column_name=None):
    """Object->declared-string leniency must not bless non-string values.

    ``check_dtype_compatibility`` is dtype-only, so it cannot tell a genuine
    plain-Python-string object column apart from one holding arbitrary
    objects (dicts, ints, mixed values).  The dtype-level early return
    accepts the latter, and the subsequent ``astype`` then silently
    stringifies it even under ``check_output_dtypes='raises'``.  Called by
    ``validate_and_align_output`` immediately after the dtype check passes;
    raises the same ``OutputDtypeMismatchError`` so the caller's
    raise/warn policy applies.
    """
    if not (
        pd.api.types.is_object_dtype(result.dtype)
        and _is_string_dtype_kind(expected_dtype)
    ):
        return
    if (result.notna() & result.map(lambda v: not isinstance(v, str))).any():
        col_info = f" for column '{column_name}'" if column_name else ""
        raise OutputDtypeMismatchError(
            column=column_name or "unknown",
            actual_dtype=result.dtype,
            expected_dtype=expected_dtype,
            can_cast=False,
            extra_msg=(
                f"Object dtype column accepted against string dtype{col_info} "
                "must hold only string values; found non-string items. "
                "Refusing to silently stringify."
            ),
        )


def validate_and_align_output(
    result: Union[pd.DataFrame, pd.Series],
    expected_dtypes: Union[pd.Series, np.dtype],
    check_output_dtypes: str = None,
    normalize_str_columns: bool = False,
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
    normalize_str_columns : bool, default False
        If True, normalize column labels to strings on both the result and
        expected dtypes before comparison.  LiteFrame always stringifies
        column names, but mock type inference may produce integer keys.

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
    # Canonicalize column labels to strings before any dtype-policy early
    # exit.  LiteFrame column names are always strings (boundary contract);
    # a scalar/positional UDF result may normalize to a pandas frame with an
    # integer label (0).  Without this, the "ignore" policy early-returns a
    # frame whose integer label survives to the user, and the "raises"/"warns"
    # path reports a spurious mismatch (integer 0 vs string "0").  Run here so
    # every policy — including "ignore" — returns string-labeled frames.
    if (
        normalize_str_columns
        and isinstance(result, pd.DataFrame)
        and expected_dtypes is not None
    ):
        # Compute the stringified labels first, without touching the frame,
        # so the collision guard below runs before any copy.  String
        # normalization can collapse distinct labels (e.g. 0 and "0") onto
        # the same "0" string, producing duplicate column names -- a
        # duplicate label makes the schema ambiguous (result[col].dtype would
        # return a DataFrame and raise AttributeError instead of reporting a
        # column-contract error).
        if isinstance(expected_dtypes, pd.Series):
            expected_columns_raw = expected_dtypes.index.tolist()
            expected_dtypes_dict_raw = {
                str(k): v for k, v in expected_dtypes.to_dict().items()
            }
        else:
            expected_columns_raw = result.columns.tolist()
            expected_dtypes_dict_raw = {
                str(c): expected_dtypes for c in expected_columns_raw
            }
        result_str_cols = [str(c) for c in result.columns]
        expected_columns = [str(c) for c in expected_columns_raw]
        expected_dtypes_dict = expected_dtypes_dict_raw

        result_dupes = {c for c in result_str_cols if result_str_cols.count(c) > 1}
        expected_dupes = {c for c in expected_columns if expected_columns.count(c) > 1}
        dupes = result_dupes | expected_dupes
        if dupes:
            collision_msg = (
                f"Duplicate column labels after string normalization: "
                f"{sorted(dupes)}. Distinct labels (e.g. int 0 and str '0') "
                "collapsed onto the same name."
            )
            # A post-normalization collision makes the schema structurally
            # ambiguous -- the frame cannot be aligned or used safely
            # downstream, so we raise in both modes rather than warn+return
            # an unusable frame.
            raise OutputColumnMismatchError(msg=collision_msg)

        # Fast path: when the result labels are already strings, skip the
        # copy entirely -- the common LiteFrame runtime case where the UDF
        # returns a large frame with already-string labels must not pay for a
        # buffer copy on every validated chunk.  Only when a label actually
        # changes do we copy, and then shallowly (the data buffers are shared;
        # only the column axis metadata is replaced).
        if result_str_cols != result.columns.tolist():
            result = result.copy(deep=False)
            result.columns = result_str_cols

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

    # Polars DataFrame/LazyFrame: validate natively rather than bridging
    # through pandas (a ``pl.Categorical`` column's Arrow dictionary index is
    # rewritten from ``uint32`` to ``int64`` by the round-trip.
    # Falls back to the pandas path for non-arrow columns.
    if pl is not None and isinstance(result, (pl.DataFrame, pl.LazyFrame)):
        return validate_and_align_polars_output(
            result,
            expected_dtypes,
            check_output_dtypes=check_output_dtypes,
            normalize_str_columns=normalize_str_columns,
        )

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
            _validate_object_string_values(result, expected_dtype, result.name)

            # If check passed, try to cast if dtypes differ
            if not _dtype_equal(actual_dtype, expected_dtype):
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

    # Get expected columns and dtypes, normalizing polars/pyarrow declared
    # dtypes to their pandas equivalent so the pandas comparison below is
    # apples-to-apples (callers may pass raw ``pl.Int64`` / ``ArrowDtype``).
    # When normalize_str_columns ran above, expected_columns /
    # expected_dtypes_dict already hold the stringified labels; recompute only
    # when normalization was not requested.
    if not normalize_str_columns:
        if isinstance(expected_dtypes, pd.Series):
            expected_columns = expected_dtypes.index.tolist()
            expected_dtypes_dict = {
                col: to_pandas_dtype(dt)
                for col, dt in expected_dtypes.to_dict().items()
            }
        else:
            # Single dtype for all columns
            expected_columns = result.columns.tolist()
            expected_dtypes_dict = {
                col: to_pandas_dtype(expected_dtypes) for col in expected_columns
            }
    else:
        # normalize_str_columns already computed expected_columns /
        # expected_dtypes_dict above (with raw expected_dtypes values);
        # normalize the dtype values to pandas equivalents now.
        expected_dtypes_dict = {
            col: to_pandas_dtype(dt) for col, dt in expected_dtypes_dict.items()
        }
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
            _validate_object_string_values(result[col], expected_dtype, col)

            # If check passed, try to cast if dtypes differ
            if not _dtype_equal(actual_dtype, expected_dtype):
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


def _declared_dtype_to_arrow_type(declared):
    """Return the pyarrow type for a declared dtype, or ``None``.

    ``None`` signals a non-arrow declared dtype (plain numpy/pandas); the
    caller falls back to the pandas validator for that column.  Polars dtype
    classes/instances (``pl.Int64``, ``pl.Categorical``) round-trip through a
    0-row Series -> pyarrow so the native polars validator can compare
    against the result's arrow schema (a ``pl.Categorical`` column's
    dictionary arrow type is preserved only on the arrow view, not through
    ``to_pandas``).
    """
    if ArrowDtype is not None and isinstance(declared, ArrowDtype):
        return declared.pyarrow_dtype
    if pa is not None and isinstance(declared, pa.DataType):
        return declared
    if pl is not None and (
        isinstance(declared, pl.DataType)
        or (isinstance(declared, type) and issubclass(declared, pl.DataType))
    ):
        try:
            return pl.Series("__c", [], dtype=declared).to_arrow().type
        except Exception:
            return None
    return None


def _arrow_type_to_polars_dtype(arrow_type):
    """Map an arrow type to a polars dtype for native casting.

    For dictionary (categorical) arrow types the pyarrow->polars converter is
    not implemented, so map to ``pl.Categorical`` — the polars equivalent a
    declared ``pl.Categorical`` round-trips through.  Other arrow types cast
    via a 0-row polars Series built from a 0-row arrow array.
    """
    if pl is None or pa is None:
        return None
    if pa.types.is_dictionary(arrow_type):
        return pl.Categorical
    try:
        arr = pa.array([], type=arrow_type)
        s = pl.from_arrow(arr)
        return s.dtype if hasattr(s, "dtype") else None
    except Exception:
        return None


def _pandas_dtype_to_polars(pandas_dt):
    """Map a pandas/numpy dtype to a polars dtype for native casting.

    Mirrors ``to_pandas_dtype`` in reverse: a 0-row pandas Series built from
    ``pandas_dt`` round-trips through polars to the equivalent dtype, so an
    ``int64`` result declared ``float64`` casts natively in polars.  Returns
    ``None`` when the dtype has no polars equivalent.
    """
    if pl is None:
        return None
    try:
        return pl.from_pandas(pd.Series([], dtype=pandas_dt, name="__c")).dtype
    except Exception:
        return None


def validate_and_align_polars_output(
    result,
    expected_dtypes,
    check_output_dtypes="raises",
    normalize_str_columns=True,
):
    """Validate and align a polars DataFrame/LazyFrame against declared dtypes.

    Polars-native counterpart of :func:`validate_and_align_output` for the
    categorical (dictionary) dtype path the pandas bridge cannot preserve: a
    polars ``Categorical`` column round-trips through ``to_pandas`` /
    ``pl.from_pandas`` with its Arrow dictionary index rewritten from
    ``uint32`` to ``int64``, so the pandas validator raised
    ``OutputDtypeMismatchError`` even when the column matched the declared
    categorical schema, and the empty-Series round-trip raised
    ``ArrowNotImplementedError`` for the dictionary converter.

    Extracts the result schema via ``DataFrame.to_arrow()`` (lossless for
    categorical), checks column presence/order and per-column dtype
    compatibility, and applies permitted casts natively in polars.  A
    declared dtype may be a pyarrow type, an ``ArrowDtype`` (incl.
    categorical dictionary), a polars dtype, or a plain numpy/pandas dtype
    (e.g. ``np.dtype("int64")`` or a scalar dtype applied to every column).
    Representation is irrelevant: an integer is an integer regardless of
    whether it arrives as ``np.dtype("int64")``, ``pl.Int64``, or
    ``int64[pyarrow]``.  Non-arrow declared dtypes fall back to a per-column
    numpy/pandas compatibility check (not skipped), so a String result
    declared ``int64`` raises ``OutputDtypeMismatchError``.  Categorical
    (dictionary) declared dtypes are validated natively; categorical support
    is otherwise out of scope here.
    """
    if pl is None or pa is None:
        return result
    if not isinstance(result, (pl.DataFrame, pl.LazyFrame)):
        return result
    should_raise = check_output_dtypes == "raises"
    warn = check_output_dtypes == "warns"
    if not (should_raise or warn):
        return result

    was_lazy = isinstance(result, pl.LazyFrame)
    if was_lazy:
        result = result.collect()

    orig_cols = list(result.columns)
    actual_cols = [str(c) for c in orig_cols]
    if normalize_str_columns and actual_cols != orig_cols:
        # rename in place to str form
        result = result.rename({a: str(a) for a in orig_cols if str(a) != a})
        actual_cols = [str(c) for c in result.columns]

    # Normalize a scalar declared dtype to a Series indexed by result columns
    # (the pandas validator accepts a single dtype as "every column"); the
    # polars path needs per-column indexing, so expand it first (a scalar
    # ``np.dtype("int64")`` raised ``AttributeError`` on ``.index``).
    if not isinstance(expected_dtypes, pd.Series):
        declared_dtype = expected_dtypes
        expected_dtypes = pd.Series({c: declared_dtype for c in actual_cols})
    elif normalize_str_columns:
        # Mirror the pandas path: stringify the index so a Series inferred
        # with integer keys (e.g. mock inference producing ``pd.DataFrame({0:
        # ...})``) aligns with the runtime string columns (``"0"``).  Without
        # this, ``expected_dtypes["0"]`` raises ``KeyError`` because the
        # Series index is integer ``0``.
        str_keys = [str(c) for c in expected_dtypes.index]
        dupes = {k for k in str_keys if str_keys.count(k) > 1}
        if dupes:
            # Mirror the pandas path's collision guard: distinct labels (int
            # ``0`` and str ``"0"``) collapsing onto one name make the schema
            # ambiguous, so raise in both modes rather than silently dropping
            # one declaration (the dict rebuild would keep only the last).
            collision_msg = (
                f"Duplicate column labels after string normalization: {sorted(dupes)}. "
                "Distinct labels (e.g. int 0 and str '0') collapsed onto the same name."
            )
            raise OutputColumnMismatchError(msg=collision_msg)
        expected_dtypes = pd.Series({str(c): v for c, v in expected_dtypes.items()})
    expected_cols = [str(c) for c in expected_dtypes.index]

    missing = [c for c in expected_cols if c not in actual_cols]
    extra = [c for c in actual_cols if c not in expected_cols]
    if missing or extra:
        err = OutputColumnMismatchError(missing_cols=missing, extra_cols=extra)
        if should_raise:
            raise err
        warnings.warn(str(err), FutureWarning)
        # On warn, mirror the pandas path: select the intersection of
        # expected columns present in the result, reorder to the expected
        # order, and continue per-column validation rather than returning the
        # unchanged frame (warns with columns ["extra", "b", "a"] and
        # declaration ["a", "b"] must return ["a", "b"]).
        present = [c for c in expected_cols if c in actual_cols]
        if not present:
            return result.lazy() if was_lazy else result
        result = result.select(present)
        expected_cols = present

    result = result.select(expected_cols)
    result_arrow = {f.name: f.type for f in result.to_arrow().schema}
    result_pl_dtypes = {c: result.schema[c] for c in expected_cols}
    cast_map = {}
    for col in expected_cols:
        declared = expected_dtypes[col]
        actual_pa = result_arrow[col]
        expected_pa = _declared_dtype_to_arrow_type(declared)
        if expected_pa is None:
            # Non-arrow declared dtype (plain numpy/pandas, or a polars
            # dtype with no arrow round-trip).  Do NOT skip: normalize both
            # sides to pandas/numpy and check compatibility, so a String
            # result declared ``int64`` raises instead of passing silently.
            # Cast natively to the declared dtype's
            # polars equivalent when compatible.
            try:
                actual_pd = to_pandas_dtype(result_pl_dtypes[col])
                expected_pd = to_pandas_dtype(declared)
            except TypeError:
                # Unresolvable declared dtype: nothing to validate against.
                continue
            try:
                check_dtype_compatibility(actual_pd, expected_pd, col)
            except OutputDtypeMismatchError as e:
                if should_raise:
                    raise
                warnings.warn(str(e), FutureWarning)
                continue
            if actual_pd != expected_pd:
                target_pl = _pandas_dtype_to_polars(expected_pd)
                if target_pl is not None:
                    cast_map[col] = target_pl
            continue
        if actual_pa == expected_pa:
            continue
        # Arrow-declared dtype (incl. categorical dictionary): reuse the
        # pandas validator's dtype-compatibility check via the ArrowDtype
        # branch.  Mirrors the pandas path: on rejection, re-raise without
        # probing (a successful probe would silently truncate float->int).
        try:
            check_dtype_compatibility(
                ArrowDtype(actual_pa), ArrowDtype(expected_pa), col
            )
        except OutputDtypeMismatchError as e:
            if should_raise:
                raise
            warnings.warn(str(e), FutureWarning)
            continue
        # Compatible: queue a native polars cast to the declared dtype.
        target_pl = _arrow_type_to_polars_dtype(expected_pa)
        if target_pl is not None:
            cast_map[col] = target_pl
    if cast_map:
        # Cast column by column so a value-dependent native cast failure
        # (e.g. Int64 value 200 -> Int8 overflow raises polars'
        # InvalidOperationError) is attributed to its column and translated
        # according to ``check_output_dtypes`` instead of leaking the polars
        # exception: ``raises`` produces OutputDtypeMismatchError, ``warns``
        # emits the warning and keeps the original column dtype.
        for col, target in cast_map.items():
            try:
                result = result.with_columns(pl.col(col).cast(target))
            except Exception as e:
                err = OutputDtypeMismatchError(
                    col,
                    result_pl_dtypes[col],
                    target,
                    can_cast=False,
                    extra_msg=(
                        f"Native cast of column '{col}' from "
                        f"{result_pl_dtypes[col]} to {target} failed: {e}"
                    ),
                )
                if should_raise:
                    raise err
                warnings.warn(str(err), FutureWarning)
    return result.lazy() if was_lazy else result
