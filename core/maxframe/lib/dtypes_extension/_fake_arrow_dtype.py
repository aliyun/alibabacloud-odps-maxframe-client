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

import re
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas import DatetimeTZDtype, Timedelta, Timestamp
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_extension_dtype,
)
from pandas.api.types import is_integer, is_scalar

try:
    from pandas._libs.arrays import NDArrayBacked
except ImportError:
    NDArrayBacked = type("NDArrayBacked", (object,), {"is_fake": True})

from ...lib.version import parse as parse_version
from ..compat import cached_property

"""
This module is copied from pandas to use in framedriver as it can only run on python3.7,
which has the highest supported pandas version 1.3.5 without ArrowDtype defined.
Once the framedriver can run on python3.11, this class can be removed.

This module will be removed in released SDK.
"""

_dtype_search_re = re.compile(r"[\[\(].*[\]\)]")
# when pyarrow<4.0, it fixes type of generated pandas block as ExtensionBlock
#  which needs special handling
_pyarrow_fix_extension_block = parse_version(pa.__version__).major < 4
_pd_string_with_storage_option = parse_version(pd.__version__).release[:2] >= (1, 3)


class FakeCategoricalDtypeType(type):
    """
    the type of FakeCategoricalDtype, this metaclass determines subclass ability
    """

    pass


def _unpack_tuple_and_ellipses(item: tuple):
    """
    Possibly unpack arr[..., n] to arr[n]
    """
    if len(item) > 1:
        # Note: we are assuming this indexing is being done on a 1D arraylike
        if item[0] is Ellipsis:
            item = item[1:]
        elif item[-1] is Ellipsis:
            item = item[:-1]

    if len(item) > 1:
        raise IndexError("too many indices for array.")

    item = item[0]
    return item


def _validate_indices(indices: np.ndarray, n: int) -> None:
    if len(indices):
        min_idx = indices.min()
        if min_idx < -1:
            msg = f"'indices' contains values less than allowed ({min_idx} < -1)"
            raise ValueError(msg)

        max_idx = indices.max()
        if max_idx >= n:
            raise IndexError("indices are out-of-bounds")


def to_pyarrow_type(dtype):
    """
    Convert dtype to a pyarrow type instance.
    """
    if isinstance(dtype, FakeArrowDtype):
        return dtype.pyarrow_dtype
    elif isinstance(dtype, pa.DataType):
        return dtype
    elif isinstance(dtype, DatetimeTZDtype):
        return pa.timestamp(dtype.unit, dtype.tz)
    elif isinstance(dtype, pd.StringDtype):
        return pa.string()
    elif dtype:
        try:
            # Accepts python types too
            # Doesn't handle all numpy types
            return pa.from_numpy_dtype(dtype)
        except pa.ArrowNotImplementedError:
            pass
    return None


class FakeArrowExtensionArray(ExtensionArray, NDArrayBacked):
    """
    In framedriver, we use arrow array as a bridge between pandas and odps datatypes,
    so we only generate empty array here to simplify the code.
    """

    def __init__(self, values: Union[pa.Array, pa.ChunkedArray]) -> None:
        if isinstance(values, pa.Array):
            self._pa_array = pa.chunked_array([values])
        elif isinstance(values, pa.ChunkedArray):
            self._pa_array = values
        else:
            raise ValueError(
                f"Unsupported type '{type(values)}' for ArrowExtensionArray"
            )
        if getattr(NDArrayBacked, "is_fake", False):
            self._dtype = FakeArrowDtype(self._pa_array.type)
        else:
            NDArrayBacked.__init__(
                self, np.array([]), FakeArrowDtype(self._pa_array.type)
            )

    @classmethod
    def _from_sequence(
        cls, scalars, dtype=None, copy: bool = False
    ) -> "FakeArrowExtensionArray":
        """
        Construct a new ExtensionArray from a sequence of scalars.
        """
        pa_type = to_pyarrow_type(dtype)
        if isinstance(scalars, (pa.Array, pa.ChunkedArray)):
            pa_array = scalars
        else:
            if len(scalars) == 0:
                # special case where pyarrow raises on empty numpy arrays
                scalars = []
            pa_array = pa.array(scalars, type=pa_type)
        arr = cls(pa_array)
        return arr

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, dtype=None, copy: bool = False
    ) -> "FakeArrowExtensionArray":
        return cls._from_sequence(strings, dtype, copy)

    @property
    def dtype(self) -> ExtensionDtype:
        """
        Implementation of ExtensionArray.dtype.
        """
        return self._dtype

    @property
    def shape(self):
        return (self._pa_array.length(),)

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        return self._pa_array.nbytes

    def __len__(self) -> int:
        """
        Return an empty array length.
        """
        return len(self._pa_array)

    def _from_pyarrow_array(self, pa_array):
        return type(self)(pa_array)

    def __getitem__(self, item):
        # code from ArrowExtensionArray in pandas>=1.5
        if isinstance(item, np.ndarray):
            if not len(item):
                pa_dtype = self._dtype.pyarrow_dtype
                result = pa.chunked_array([], type=pa_dtype)
                return self._from_pyarrow_array(result)
            elif item.dtype.kind in "iu":
                return self.take(item)
            elif item.dtype.kind == "b":
                return self._from_pyarrow_array(self._pa_array.filter(item))
            else:
                raise IndexError(
                    "Only integers, slices and integer or "
                    "boolean arrays are valid indices."
                )
        elif isinstance(item, tuple):
            item = _unpack_tuple_and_ellipses(item)

        if item is Ellipsis:
            # TODO: should be handled by pyarrow?
            item = slice(None)

        if is_scalar(item) and not is_integer(item):
            # e.g. "foo" or 2.5
            # exception message copied from numpy
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )
            # We are not an array indexer, so maybe e.g. a slice or integer
            # indexer. We dispatch to pyarrow.
        if isinstance(item, slice):
            # Arrow bug https://github.com/apache/arrow/issues/38768
            if item.start == item.stop:
                pass
            elif (
                item.stop is not None
                and item.stop < -len(self)
                and item.step is not None
                and item.step < 0
            ):
                item = slice(item.start, None, item.step)
        value = self._pa_array[item]
        if isinstance(value, pa.ChunkedArray):
            return self._from_pyarrow_array(value)
        else:
            pa_type = self._pa_array.type
            scalar = value.as_py()
            if scalar is None:
                return self._dtype.na_value
            elif pa.types.is_timestamp(pa_type) and pa_type.unit != "ns":
                # GH 53326
                return Timestamp(scalar).as_unit(pa_type.unit)
            elif pa.types.is_duration(pa_type) and pa_type.unit != "ns":
                # GH 53326
                return Timedelta(scalar).as_unit(pa_type.unit)
            else:
                return scalar

    def __arrow_array__(self, type=None):
        """Convert myself to a pyarrow ChunkedArray."""
        return self._pa_array

    def copy(self) -> "FakeArrowExtensionArray":
        return self._from_pyarrow_array(self._pa_array)

    def isna(self) -> np.ndarray:
        # code from ArrowExtensionArray in pandas>=1.5
        null_count = self._pa_array.null_count
        if null_count == 0:
            return np.zeros(len(self), dtype=np.bool_)
        elif null_count == len(self):
            return np.ones(len(self), dtype=np.bool_)

        return self._pa_array.is_null().to_numpy()

    def take(self, indices, allow_fill=False, fill_value=None):
        # code from ArrowExtensionArray in pandas>=1.5
        indices_array = np.asanyarray(indices)

        if len(self._pa_array) == 0 and (indices_array >= 0).any():
            raise IndexError("cannot do a non-empty take")
        if indices_array.size > 0 and indices_array.max() >= len(self._pa_array):
            raise IndexError("out of bounds value in 'indices'.")

        if allow_fill:
            fill_mask = indices_array < 0
            if fill_mask.any():
                _validate_indices(indices_array, len(self._pa_array))
                # TODO(ARROW-9433): Treat negative indices as NULL
                indices_array = pa.array(indices_array, mask=fill_mask)
                result = self._pa_array.take(indices_array)
                if pd.isna(fill_value):
                    return self._from_pyarrow_array(result)
                # TODO: ArrowNotImplementedError: Function fill_null has no
                # kernel matching input types (array[string], scalar[string])
                result = self._from_pyarrow_array(result)
                result[fill_mask] = fill_value
                return result
                # return type(self)(pc.fill_null(result, pa.scalar(fill_value)))
            else:
                # Nothing to fill
                return self._from_pyarrow_array(self._pa_array.take(indices))
        else:  # allow_fill=False
            # TODO(ARROW-9432): Treat negative indices as indices from the right.
            if (indices_array < 0).any():
                # Don't modify in-place
                indices_array = np.copy(indices_array)
                indices_array[indices_array < 0] += len(self._pa_array)
            return self._from_pyarrow_array(self._pa_array.take(indices_array))

    @classmethod
    def _concat_same_type(cls, to_concat):
        # code from ArrowExtensionArray in pandas>=1.5
        chunks = [array for ea in to_concat for array in ea._pa_array.iterchunks()]
        if to_concat[0].dtype == "string":
            # StringDtype has no attribute pyarrow_dtype
            pa_dtype = pa.large_string()
        else:
            pa_dtype = to_concat[0].dtype.pyarrow_dtype
        arr = pa.chunked_array(chunks, type=pa_dtype)
        return to_concat[0]._from_pyarrow_array(arr)


@register_extension_dtype
class FakeArrowDtype(ExtensionDtype):
    def __new__(cls, pyarrow_dtype):
        # TODO: here we avoid returning FakeDatetimeTZArrowDtype to make
        #  the behavior of timestamp consistent with other types when
        #  ExtensionDtype is fixed on pyarrow. Remove `_pyarrow_fix_extension_block`
        #  condition once we can enforce pyarrow>=4.0
        if (
            not _pyarrow_fix_extension_block
            and cls is FakeArrowDtype
            and pa.types.is_timestamp(pyarrow_dtype)
            and pyarrow_dtype.unit == "ns"
        ):
            # Need special logic for DatetimeTZDtype
            return FakeDatetimeTZArrowDtype(pyarrow_dtype)
        elif _pd_string_with_storage_option and pyarrow_dtype == pa.string():
            # Use builtin StringDtype with arrow support to
            #  avoid compatibility issues
            return pd.StringDtype(storage="pyarrow")
        return object.__new__(cls)

    def __init__(self, pyarrow_dtype):
        super().__init__()
        self.pyarrow_dtype = pyarrow_dtype

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return FakeArrowExtensionArray

    @classmethod
    def construct_from_string(cls, string: str):
        """
        Construct this type from a string.

        Parameters
        ----------
        string : str
            string should follow the format f"{pyarrow_type}[pyarrow]"
            e.g. int64[pyarrow]
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if not string.endswith("[pyarrow]"):
            raise TypeError(f"'{string}' must end with '[pyarrow]'")
        if string == "string[pyarrow]":
            # Ensure Registry.find skips ArrowDtype to use StringDtype instead
            raise TypeError("string[pyarrow] should be constructed by StringDtype")

        base_type = string[:-9]  # get rid of "[pyarrow]"
        try:
            pa_dtype = pa.type_for_alias(base_type)
        except ValueError as err:
            has_parameters = _dtype_search_re.search(base_type)
            if has_parameters:
                # Fallback to try common temporal types
                try:
                    return cls._parse_temporal_dtype_string(base_type)
                except (NotImplementedError, ValueError):
                    # Fall through to raise with nice exception message below
                    pass

                raise NotImplementedError(
                    "Passing pyarrow type specific parameters "
                    f"({has_parameters.group()}) in the string is not supported. "
                    "Please construct an ArrowDtype object with a pyarrow_dtype "
                    "instance with specific parameters."
                ) from err
            raise TypeError(f"'{base_type}' is not a valid pyarrow data type.") from err
        return cls(pa_dtype)

    @property
    def _is_numeric(self) -> bool:
        """
        Whether columns with this dtype should be considered numeric.
        """
        # TODO: pa.types.is_boolean?
        return (
            pa.types.is_integer(self.pyarrow_dtype)
            or pa.types.is_floating(self.pyarrow_dtype)
            or pa.types.is_decimal(self.pyarrow_dtype)
        )

    @property
    def _is_boolean(self) -> bool:
        """
        Whether this dtype should be considered boolean.
        """
        return pa.types.is_boolean(self.pyarrow_dtype)

    def _get_common_dtype(self, dtypes):
        # We unwrap any masked dtypes, find the common dtype we would use
        #  for that, then re-mask the result.
        # Mirrors BaseMaskedDtype
        from pandas.core.dtypes.cast import find_common_type

        null_dtype = type(self)(pa.null())

        new_dtype = find_common_type(
            [
                dtype.numpy_dtype if isinstance(dtype, FakeArrowDtype) else dtype
                for dtype in dtypes
                if dtype != null_dtype
            ]
        )
        if not isinstance(new_dtype, np.dtype):
            return None
        try:
            pa_dtype = pa.from_numpy_dtype(new_dtype)
            return type(self)(pa_dtype)
        except NotImplementedError:
            return None

    @property
    def type(self):
        """
        Returns associated scalar type.
        """
        pa_type = self.pyarrow_dtype
        if pa.types.is_integer(pa_type):
            return int
        elif pa.types.is_floating(pa_type):
            return float
        elif pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
            return str
        elif (
            pa.types.is_binary(pa_type)
            or pa.types.is_fixed_size_binary(pa_type)
            or pa.types.is_large_binary(pa_type)
        ):
            return bytes
        elif pa.types.is_boolean(pa_type):
            return bool
        elif pa.types.is_duration(pa_type):
            if pa_type.unit == "ns":
                return Timedelta
            else:
                return timedelta
        elif pa.types.is_timestamp(pa_type):
            # TODO: here we avoid returning Timestamp when ExtensionDtype
            #  is fixed on pyarrow. Remove `_pyarrow_fix_extension_block`
            #  condition once we can enforce pyarrow>=4.0
            if not _pyarrow_fix_extension_block and pa_type.unit == "ns":
                return Timestamp
            else:
                return datetime
        elif pa.types.is_date(pa_type):
            return date
        elif pa.types.is_time(pa_type):
            return time
        elif pa.types.is_decimal(pa_type):
            return Decimal
        elif pa.types.is_dictionary(pa_type):
            # TODO: Potentially change this & CategoricalDtype.type to
            #  something more representative of the scalar
            return FakeCategoricalDtypeType
        elif pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
            return list
        elif pa.types.is_fixed_size_list(pa_type):
            return list
        elif pa.types.is_map(pa_type):
            return list
        elif pa.types.is_struct(pa_type):
            return dict
        elif pa.types.is_null(pa_type):
            # TODO: None? pd.NA? pa.null?
            return type(pa_type)
        elif isinstance(pa_type, pa.ExtensionType):
            return type(self)(pa_type.storage_type).type
        raise NotImplementedError(pa_type)

    @property
    def name(self) -> str:
        """
        A string identifying the data type.
        """
        return f"{str(self.pyarrow_dtype)}[pyarrow]"

    @cached_property
    def numpy_dtype(self) -> np.dtype:
        """Return an instance of the related numpy dtype"""
        if pa.types.is_timestamp(self.pyarrow_dtype):
            # pa.timestamp(unit).to_pandas_dtype() returns ns units
            # regardless of the pyarrow timestamp units.
            # This can be removed if/when pyarrow addresses it:
            # https://github.com/apache/arrow/issues/34462
            return np.dtype(f"datetime64[{self.pyarrow_dtype.unit}]")
        if pa.types.is_duration(self.pyarrow_dtype):
            # pa.duration(unit).to_pandas_dtype() returns ns units
            # regardless of the pyarrow duration units
            # This can be removed if/when pyarrow addresses it:
            # https://github.com/apache/arrow/issues/34462
            return np.dtype(f"timedelta64[{self.pyarrow_dtype.unit}]")
        if pa.types.is_string(self.pyarrow_dtype) or pa.types.is_large_string(
            self.pyarrow_dtype
        ):
            # pa.string().to_pandas_dtype() = object which we don't want
            return np.dtype(str)
        try:
            return np.dtype(self.pyarrow_dtype.to_pandas_dtype())
        except (NotImplementedError, TypeError):
            return np.dtype(object)

    @cached_property
    def kind(self) -> str:
        if pa.types.is_timestamp(self.pyarrow_dtype):
            # To mirror DatetimeTZDtype
            return "M"
        return self.numpy_dtype.kind

    @cached_property
    def itemsize(self) -> int:
        """Return the number of bytes in this dtype"""
        return self.numpy_dtype.itemsize

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return super().__eq__(other)
        return self.pyarrow_dtype == other.pyarrow_dtype

    def __hash__(self) -> int:
        # make myself hashable
        return hash(str(self))

    def __from_arrow__(self, array):
        array_class = self.construct_array_type()
        arr = array.cast(self.pyarrow_dtype, safe=True)
        return array_class(arr)


class FakeDatetimeTZExtensionArray(FakeArrowExtensionArray):
    """
    Workaround array class for DatetimeTZDtype in pandas when
    arrow type is timestamp[ns]
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._ndim = 1

    @property
    def shape(self):
        return (
            (self._pa_array.length(),)
            if self._ndim == 1
            else (1, self._pa_array.length())
        )

    @property
    def ndim(self) -> int:
        return self._ndim

    def __len__(self) -> int:
        return len(self._pa_array) if self._ndim == 1 else self.shape[0]

    def __getitem__(self, item):
        if self._ndim == 2 and item == 0:
            return self._from_pyarrow_array(self._pa_array)
        return super().__getitem__(item)

    def reshape(self, *args, **kwargs):
        if args != (1, -1):
            raise ValueError("Only support reshape to (1, -1)")
        new_arr = self._from_pyarrow_array(self._pa_array)
        new_arr._ndim = 2
        return new_arr


class FakeDatetimeTZArrowDtype(DatetimeTZDtype, FakeArrowDtype):
    """
    Workaround dtype class for DatetimeTZDtype in pandas when
    arrow type is timestamp[ns]
    """

    def __init__(self, pyarrow_type):
        from ... import options

        FakeArrowDtype.__init__(self, pyarrow_type)
        DatetimeTZDtype.__init__(
            self, unit=pyarrow_type.unit, tz=pyarrow_type.tz or options.local_timezone
        )

    @classmethod
    def construct_array_type(cls):
        return FakeDatetimeTZExtensionArray
