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

import dataclasses
import functools
import inspect
from typing import Any, Callable, Generic, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd
import pyarrow as pa

from maxframe.core import OutputType
from maxframe.dataframe.type_infer import InferredDataFrameMeta
from maxframe.dataframe.utils import parse_index
from maxframe.lib.dtypes_extension import ArrowDtype, ExternalBlobDtype
from maxframe.lib.dtypes_extension.blob import AbstractExternalBlob
from maxframe.typing_ import PandasDType
from maxframe.utils import arrow_type_from_str, make_dtype, wrap_arrow_dtype

try:
    from odps import types as odps_types
except ImportError:
    odps_types = None

# TypeVars
T = TypeVar("T")


@dataclasses.dataclass
class _FieldDef:
    name: Any
    dtype: PandasDType


def _item_to_field_def(item_):
    if isinstance(item_, tuple):
        tp = make_dtype(item_[1])
        return _FieldDef(name=item_[0], dtype=tp)
    elif isinstance(item_, slice):
        assert item_.step is None, "Should not specify step when specifying type hints"
        return _FieldDef(name=item_.start, dtype=item_.stop)
    else:
        tp = make_dtype(item_)
        return _FieldDef(name=None, dtype=tp)


class IndexType:
    def __init__(self, index_fields: List[_FieldDef]):
        self.index_fields = index_fields

    def __repr__(self):
        return f"IndexType({[f.dtype for f in self.index_fields]})"

    def __reduce__(self):
        # When picked to executors, detailed field types are not needed,
        #  and we should not introduce MaxFrame itself as a dependency.
        return getattr, (pd, "Index")

    @classmethod
    def from_getitem_args(cls, item) -> "IndexType":
        if isinstance(item, (dict, pd.Series)):
            item = list(item.items())

        if isinstance(item, list) or (
            item and isinstance(item, tuple) and isinstance(item[0], slice)
        ):
            return IndexType([_item_to_field_def(tp) for tp in item])
        else:
            return IndexType([_item_to_field_def(item)])


class SeriesType(Generic[T]):
    def __init__(
        self,
        index_fields: Optional[List[_FieldDef]],
        name_and_dtype: _FieldDef,
        element_fields: Optional[List[_FieldDef]] = None,
    ):
        self.index_fields = index_fields
        self.name_and_dtype = name_and_dtype
        self.element_fields = element_fields  # NEW: for dict-based element types

    def __repr__(self) -> str:
        if self.element_fields:
            return "SeriesType[{{{}}}]".format(
                ", ".join(f"{fd.name}: {fd.dtype}" for fd in self.element_fields)
            )
        return "SeriesType[{}]".format(self.name_and_dtype.dtype)

    def __reduce__(self):
        # When picked to executors, detailed field types are not needed,
        #  and we should not introduce MaxFrame itself as a dependency.
        return getattr, (pd, "Series")

    @classmethod
    def from_getitem_args(cls, item) -> "SeriesType":
        if not isinstance(item, tuple):
            item = (item,)

        # NEW: Check if item[0] is a dict (element type specification)
        if len(item) == 1 and isinstance(item[0], dict):
            # Parse dict into element_fields
            if len(item[0]) == 0:
                raise ValueError("Element type annotation dict cannot be empty")
            element_fields = [
                _FieldDef(name=k, dtype=make_dtype(v)) for k, v in item[0].items()
            ]
            return SeriesType(
                index_fields=None,
                name_and_dtype=_FieldDef(name=None, dtype=None),
                element_fields=element_fields,
            )

        # Existing logic for other cases
        if len(item) == 1:
            tp = _item_to_field_def(item[0])
            return SeriesType(None, tp)
        else:
            tp = _item_to_field_def(item[1])
            idx_fields = IndexType.from_getitem_args(item[0]).index_fields
            return SeriesType(idx_fields, tp)


class DataFrameType:
    def __init__(
        self,
        index_fields: Optional[List[_FieldDef]],
        data_fields: List[_FieldDef],
    ):
        self.index_fields = index_fields
        self.data_fields = data_fields

    def __repr__(self) -> str:
        types = [field.dtype for field in self.data_fields]
        return f"DataFrameType[{types}]"

    def __reduce__(self):
        # When picked to executors, detailed field types are not needed,
        #  and we should not introduce MaxFrame itself as a dependency.
        return getattr, (pd, "DataFrame")

    @classmethod
    def from_getitem_args(cls, item) -> "DataFrameType":
        if not isinstance(item, tuple):
            item = (item,)
        if isinstance(item[0], slice):
            value_defs = item
            idx_defs = None
        else:
            value_defs = item[-1]
            idx_defs = item[0] if len(item) > 1 else None
        fields = IndexType.from_getitem_args(value_defs).index_fields
        if idx_defs is None:
            return DataFrameType(None, fields)
        else:
            idx_fields = IndexType.from_getitem_args(item[0]).index_fields
            return DataFrameType(idx_fields, fields)


def get_function_output_meta(
    func: Callable, df_obj=None, axis: Optional[int] = None
) -> Optional[InferredDataFrameMeta]:
    try:
        func_argspec = inspect.getfullargspec(func)
        ret_type = (func_argspec.annotations or {}).get("return")
        if ret_type is None:
            return None
    except:
        return None

    dtypes = dtype = name = None
    index_fields = None

    # Handle SeriesType with element_fields when axis=1
    if isinstance(ret_type, SeriesType) and ret_type.element_fields and axis == 1:
        # Convert to DataFrame output
        output_type = OutputType.dataframe
        dtypes = pd.Series(
            [fd.dtype for fd in ret_type.element_fields],
            index=[fd.name for fd in ret_type.element_fields],
        )
        index_fields = None  # Will be inherited from input DataFrame
    elif isinstance(ret_type, DataFrameType):
        output_type = OutputType.dataframe
        dtypes = pd.Series(
            [fd.dtype for fd in ret_type.data_fields],
            index=[fd.name for fd in ret_type.data_fields],
        )
        index_fields = ret_type.index_fields
    elif isinstance(ret_type, SeriesType):
        output_type = OutputType.series
        dtype = ret_type.name_and_dtype.dtype
        name = ret_type.name_and_dtype.name
        index_fields = ret_type.index_fields
    elif isinstance(ret_type, IndexType):
        output_type = OutputType.index
        index_fields = ret_type.index_fields
    else:
        output_type = OutputType.scalar
        try:
            dtype = make_dtype(ret_type)
        except:
            return None

    if index_fields is not None:
        if len(index_fields) == 1:
            mock_idx = pd.Index(
                [], dtype=index_fields[0].dtype, name=index_fields[0].name
            )
        else:
            col_names = [index_field.name for index_field in index_fields]
            col_dtypes = pd.Series(
                [index_field.dtype for index_field in index_fields], index=col_names
            )
            mock_df = pd.DataFrame([], columns=col_names).astype(col_dtypes)
            mock_idx = pd.MultiIndex.from_frame(mock_df)
        index_value = parse_index(mock_idx, df_obj, store_data=False)
    else:
        index_value = None

    return InferredDataFrameMeta(
        output_type=output_type,
        index_value=index_value,
        dtypes=dtypes,
        dtype=dtype,
        name=name,
    )


def _dtype(type) -> Union[pd.api.extensions.ExtensionDtype]:
    from maxframe.io.odpsio.schema import odps_type_to_arrow_type

    # Handle blob type special case for ODPS DataType
    if odps_types is not None and hasattr(odps_types, "blob"):
        if type is odps_types.blob or (
            isinstance(type, odps_types.DataType) and type == odps_types.blob
        ):
            return ExternalBlobDtype()

    # Handle Python built-in type classes
    # Check if type is one of the supported Python type classes directly
    if type is str:
        return pd.StringDtype("pyarrow")
    elif type is bytes:
        return wrap_arrow_dtype(pa.binary())
    elif type in (int, float, bool):
        type = np.dtype(type)
        # Fall through to np.dtype handling below

    # Convert to PyArrow type based on input type
    if isinstance(type, str):
        # String type representation
        # First, try to parse as ODPS type if odps is available
        if odps_types is not None:
            try:
                odps_type = odps_types.validate_data_type(type)
                # Check if it's blob type
                if hasattr(odps_types, "blob") and odps_type == odps_types.blob:
                    return ExternalBlobDtype()
                # Convert ODPS type to Arrow type
                arrow_type = odps_type_to_arrow_type(odps_type, "column")
            except (ValueError, AttributeError):
                # Not an ODPS type, try Arrow type string
                arrow_type = arrow_type_from_str(type)
        else:
            # odps not installed, use Arrow type string parsing
            arrow_type = arrow_type_from_str(type)

    elif isinstance(type, np.dtype):
        # NumPy dtype
        if pa is None:
            raise TypeError(
                "PyArrow is required to convert NumPy dtypes. "
                "Please install pyarrow package."
            )
        arrow_type = pa.from_numpy_dtype(type)

    elif isinstance(type, pd.api.extensions.ExtensionDtype):
        # Pandas ExtensionDtype
        if isinstance(type, (ArrowDtype, ExternalBlobDtype)):
            # Already an ArrowDtype or an ExternalBlobDtype, return as-is
            return type
        elif isinstance(type, pd.StringDtype):
            # Convert StringDtype to pyarrow-backed string dtype
            try:
                return pd.StringDtype("pyarrow")
            except (TypeError, ImportError):
                # Fall back to ArrowDtype if StringDtype("pyarrow") not available
                arrow_type = pa.string()
        elif hasattr(type, "pyarrow_dtype"):
            # Extension dtype with PyArrow backing
            arrow_type = type.pyarrow_dtype
        else:
            # Try to convert via pandas
            try:
                arrow_type = pa.from_numpy_dtype(type.numpy_dtype)
            except (AttributeError, TypeError):
                raise TypeError(
                    f"Cannot convert pandas ExtensionDtype '{type}' to MaxFrame dtype. "
                    f"Supported types include ArrowDtype, StringDtype, and numeric dtypes."
                )

    elif odps_types is not None and isinstance(type, odps_types.DataType):
        # ODPS type
        arrow_type = odps_type_to_arrow_type(type, "column")

    elif pa is not None and isinstance(type, pa.DataType):
        # Already a PyArrow type
        arrow_type = type

    else:
        # Unsupported type
        raise TypeError(
            f"Unsupported type '{type}' of type '{type.__class__.__name__}'. "
            f"Supported types are: str, np.dtype, pd.ExtensionDtype, "
            f"odps.types.DataType (if odps installed), or pa.DataType (if pyarrow installed)."
        )

    # Wrap Arrow type in ArrowDtype
    return wrap_arrow_dtype(arrow_type)


_cached_dtype = functools.lru_cache(1000)(_dtype)


def dtype(type):
    """
    Recommend dtype for MaxFrame DataFrame operations.

    Converts various type representations into MaxFrame-compatible dtypes.
    Returns pd.ArrowDtype for most types and ExternalBlobDtype for blob types.

    Parameters
    ----------
    type : str, np.dtype, pd.ExtensionDtype, odps.types.DataType, pa.DataType, or Python type class
        The type to convert to MaxFrame-compatible dtype.
        Supported input types:
        - str: Type string like 'int64', 'string', 'blob', 'list<item: int64>', etc.
        - np.dtype: NumPy dtype objects
        - pd.ExtensionDtype: Pandas extension dtypes
        - odps.types.DataType: ODPS data types (requires odps package)
        - pa.DataType: PyArrow data types
        - Python type classes: int, float, bool, str, bytes
          (int -> int64, float -> float64, bool -> bool, str -> StringDtype(pyarrow), bytes -> binary)

    Returns
    -------
    ArrowDtype or ExternalBlobDtype
        Recommended dtype for use in MaxFrame operations.
        Returns ExternalBlobDtype for blob types, ArrowDtype for all others.

    Raises
    ------
    TypeError
        If the input type is not supported or cannot be converted.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> md.dtype("int64")
    ArrowDtype(int64[pyarrow])

    >>> md.dtype("blob")
    ExternalBlobDtype()

    >>> import numpy as np
    >>> md.dtype(np.dtype('float32'))
    ArrowDtype(float[pyarrow])

    >>> from odps import types as odps_types
    >>> md.dtype(odps_types.string)
    ArrowDtype(string[pyarrow])

    >>> md.dtype(int)
    ArrowDtype(int64[pyarrow])

    >>> md.dtype(float)
    ArrowDtype(double[pyarrow])

    >>> md.dtype(str)
    StringDtype(pyarrow)

    >>> md.dtype(bytes)
    ArrowDtype(binary[pyarrow])
    """
    try:
        return _cached_dtype(type)
    except TypeError as ex:
        if "unhashable" in str(ex):
            # raise actual error
            return _dtype(type)
        raise


def infer_dtype(obj: Any) -> Union[ArrowDtype, ExternalBlobDtype]:
    """
    Infer MaxFrame-compatible dtype from a Python object.

    Creates a single-element PyArrow array from the object to infer its type,
    then converts to MaxFrame-compatible dtype using the dtype() function.

    Parameters
    ----------
    obj : Any
        Python object to infer dtype from. Can be:
        - Scalar values (int, float, bool, str, bytes)
        - Lists/tuples (for list/array types)
        - Dicts (for struct types)
        - Blob objects (SolidBlob)
        - Any other Python object supported by PyArrow

    Returns
    -------
    ArrowDtype or ExternalBlobDtype
        Inferred dtype for the object.

    Raises
    ------
    TypeError
        If the object type is not supported or dtype cannot be inferred.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> md.infer_dtype(42)
    ArrowDtype(int64[pyarrow])

    >>> md.infer_dtype(3.14)
    ArrowDtype(double[pyarrow])

    >>> md.infer_dtype("hello")
    StringDtype(pyarrow)

    >>> md.infer_dtype([1, 2, 3])
    ArrowDtype(list<item: int64>[pyarrow])

    >>> md.infer_dtype({"a": 1, "b": 2})
    ArrowDtype(struct<a: int64, b: int64>[pyarrow])
    """
    if obj is None:
        raise TypeError("Cannot infer dtype from None")

    # Handle blob objects specially - they need ExternalBlobDtype
    if isinstance(obj, AbstractExternalBlob):
        return ExternalBlobDtype()

    try:
        # Create a single-element PyArrow array to infer type
        arr = pa.array([obj])
        arrow_type = arr.type

        # Use existing dtype() to convert Arrow type to MaxFrame dtype
        # This ensures consistency and handles special cases like blob types
        return dtype(arrow_type)

    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise TypeError(
            f"Cannot infer dtype from object of type {type(obj).__name__}: {exc}"
        ) from exc


def register_pandas_typing_funcs():
    def _cls_getitem_func(cls, item, type_cls):
        return type_cls.from_getitem_args(item)

    for pd_cls, type_cls in [
        (pd.DataFrame, DataFrameType),
        (pd.Series, SeriesType),
        (pd.Index, IndexType),
    ]:
        if hasattr(pd_cls, "__class_getitem__"):  # pragma: no cover
            continue
        pd_cls.__class_getitem__ = classmethod(
            functools.partial(_cls_getitem_func, type_cls=type_cls)
        )
