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

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

try:
    from maxframe.lib.dtypes_extension import ArrowDtype
except ImportError:  # pragma: no cover
    ArrowDtype = None


_VALID_ERRORS_VALUES = frozenset({"raise", "ignore", "coerce"})


def validate_errors(errors: str, allow_coerce: bool = True) -> None:
    """Validate the ``errors`` parameter accepted by functions like ``astype`` and ``drop``.

    Raises ``ValueError`` if *errors* is not one of the allowed values.
    When *allow_coerce* is False, ``'coerce'`` is rejected.
    """
    valid = _VALID_ERRORS_VALUES if allow_coerce else frozenset({"raise", "ignore"})
    if errors not in valid:
        raise ValueError(
            f"invalid value for errors: '{errors}'. " f"Must be one of {sorted(valid)}"
        )


def normalize_col_name(name: str) -> str:
    """Normalize a column name to lowercase for case-insensitive lookups."""
    return name.lower()


def normalize_dtypes_index(dtypes):
    """Lowercase the index of a dtypes Series."""
    if isinstance(dtypes, pd.Series):
        dtypes.index = dtypes.index.map(normalize_col_name)
    return dtypes


def _generate_value(dtype, fill_value):
    """Generate a mock value of the given dtype for building empty Series.

    Handles ArrowDtype by unwrapping to the underlying PyArrow type,
    then recursively generating a suitable value.
    """
    if ArrowDtype is not None and isinstance(dtype, ArrowDtype):
        return _generate_value(dtype.pyarrow_dtype, fill_value)

    if pa is not None and isinstance(dtype, pa.ListType):
        return [_generate_value(dtype.value_type, fill_value)]
    elif pa is not None and isinstance(dtype, pa.MapType):
        return [
            (
                _generate_value(dtype.key_type, fill_value),
                _generate_value(dtype.item_type, fill_value),
            )
        ]
    elif pa is not None and isinstance(dtype, pa.StructType):
        result = {}
        for i in range(dtype.num_fields):
            field = dtype[i]
            result[field.name] = _generate_value(field.type, fill_value)
        return result
    elif pa is not None and isinstance(dtype, pa.DataType):
        return pa.array([_generate_value(dtype.to_pandas_dtype(), fill_value)]).cast(
            dtype
        )[0]

    # special handle for datetime64 and timedelta64
    dispatch = {
        np.datetime64: pd.Timestamp,
        np.timedelta64: pd.Timedelta,
        pd.CategoricalDtype.type: lambda x: pd.CategoricalDtype([x]),
        # for object, we do not know the actual dtype,
        # just convert to str for common usage
        np.object_: lambda _: str(fill_value),
    }
    # otherwise, just use dtype.type itself to convert
    target_dtype = getattr(dtype, "type", dtype)
    convert = dispatch.get(target_dtype, target_dtype)
    ret = convert(fill_value)
    if target_dtype in (np.datetime64, np.timedelta64):
        target_unit = np.datetime_data(dtype)[0]
        if (
            target_unit
            and hasattr(ret, "unit")
            and ret.unit != target_unit
            and target_unit != "generic"
        ):
            ret = ret.as_unit(target_unit)
    return ret


def build_empty_series(dtype, index=None, name=None):
    """Build an empty pandas Series with the given dtype.

    Handles ArrowDtype and other ExtensionDtypes by generating a suitable
    mock value so that the Series can be created correctly.
    """
    length = len(index) if index is not None else 0
    return pd.Series(
        [_generate_value(dtype, 1) for _ in range(max(1, length))],
        dtype=dtype,
        index=index,
        name=name,
    )[:length]
