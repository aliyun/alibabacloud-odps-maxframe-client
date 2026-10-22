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

"""Type inference utilities for LiteFrame UDF APIs.

Provides:
- LiteFrameType: annotation class for LiteFrame["col":dtype, ...] syntax
- build_mock_dataframe: creates a small mock DataFrame for inference
- parse_return_annotation: extracts dtypes from return annotations
- infer_output_dtypes: unified inference with priority order
"""

import datetime
import inspect
from typing import List, Optional, get_type_hints

import numpy as np
import pandas as pd
import pyarrow as pa

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

from maxframe.dataframe.typing_ import IndexType, _FieldDef
from maxframe.lib.dtypes_extension import ArrowDtype
from maxframe.liteframe.datatypes import _is_fake_arrow_dtype, _to_numpy_dtype
from maxframe.udf import MarkedFunction

try:
    from maxframe.lib.dtypes_extension.blob import (
        ExternalBlobDtype,
        ExternalBlobExtensionArray,
        SolidBlob,
        is_polars_blob_type,
        pandas_to_polars_with_blobs,
    )
except ImportError:  # pragma: no cover
    ExternalBlobDtype = None
    ExternalBlobExtensionArray = None
    SolidBlob = None
    is_polars_blob_type = None
    pandas_to_polars_with_blobs = None
from maxframe.utils import make_dtypes, unwrap_function, wrap_arrow_dtype


class LiteFrameType:
    """Type annotation class for LiteFrame outputs.

    Enables type hints like:
        def func(df) -> LiteFrame["a":int, "b":float]: ...

    Analogous to DataFrameType but without index fields.
    """

    def __init__(self, data_fields: List[_FieldDef]):
        self.data_fields = data_fields

    @classmethod
    def from_getitem_args(cls, item) -> "LiteFrameType":
        """Parse __class_getitem__ args into LiteFrameType."""
        if not isinstance(item, tuple):
            item = (item,)
        fields = IndexType.from_getitem_args(item).index_fields
        return LiteFrameType(fields)

    def to_dtypes(self) -> pd.Series:
        """Convert data_fields to a pd.Series of dtypes."""
        names = []
        dtypes = []
        for field in self.data_fields:
            names.append(field.name)
            dtypes.append(_normalize_to_arrow_dtype(field.dtype))
        return pd.Series(dtypes, index=names)


def _normalize_to_arrow_dtype(dtype):
    """Convert a dtype specification to ArrowDtype when possible."""
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return dtype
    # Polars dtype classes/instances: round-trip a 0-row Series through
    # pyarrow so polars dtypes enter the pipeline as arrow types, never
    # reaching downstream pandas_dtype sites raw.
    if pl is not None:
        is_pl = isinstance(dtype, pl.DataType) or (
            isinstance(dtype, type) and issubclass(dtype, pl.DataType)
        )
        if is_pl:
            is_decimal = dtype is pl.Decimal or isinstance(dtype, pl.Decimal)
            try:
                pa_type = pl.Series("__c", [], dtype=dtype).to_arrow().type
            except pa.ArrowInvalid:
                # pyarrow < 2.0 cannot import polars decimal arrays via the
                # C ABI (``Invalid or unsupported format string: 'd:38,0'``),
                # so the 0-row round-trip that recovers the arrow type for
                # other dtypes crashes here.  For decimal, the crash itself
                # proves the legacy stack cannot honor the declared schema,
                # so surface the same "cannot preserve" rejection instead of
                # a raw ArrowInvalid.
                if is_decimal:
                    precision = getattr(dtype, "precision", None) or 38
                    scale = getattr(dtype, "scale", None) or 0
                    raise ValueError(
                        f"polars {pl.__version__} cannot preserve the declared "
                        f"Decimal(precision={precision}, scale={scale}) schema "
                        "upgrade polars to >= 1.0 or declare the decimal via "
                        "pd.ArrowDtype(pa.decimal128(...)) instead."
                    )
                raise
            if is_decimal:
                # Legacy polars (< 1.0) drops the declared
                # precision/scale in the Arrow round trip and returns
                # decimal128(38, 0).  Publishing different parameters from the
                # explicitly declared schema is unacceptable, so reject the
                # declaration instead of drifting silently.
                precision = getattr(dtype, "precision", None) or 38
                scale = getattr(dtype, "scale", None) or 0
                actual_params = (
                    (pa_type.precision, pa_type.scale)
                    if pa.types.is_decimal(pa_type)
                    else None
                )
                if actual_params != (precision, scale):
                    raise ValueError(
                        f"polars {pl.__version__} cannot preserve the declared "
                        f"Decimal(precision={precision}, scale={scale}) schema "
                        "upgrade polars to >= 1.0 or declare the decimal via "
                        "pd.ArrowDtype(pa.decimal128(...)) instead."
                    )
            return wrap_arrow_dtype(pa_type)
    try:
        if isinstance(dtype, np.dtype):
            pa_type = pa.from_numpy_dtype(dtype)
        elif dtype is int:
            pa_type = pa.int64()
        elif dtype is float:
            pa_type = pa.float64()
        elif dtype is str:
            pa_type = pa.string()
        elif dtype is bool:
            pa_type = pa.bool_()
        else:
            pa_type = pa.from_numpy_dtype(np.dtype(dtype))
        return wrap_arrow_dtype(pa_type)
    except (TypeError, ValueError):
        if isinstance(dtype, np.dtype):
            return dtype
        return np.dtype(dtype)


def parse_return_annotation(annotation) -> Optional[List[pd.Series]]:
    """Parse a function's return type annotation into a list of dtypes Series.

    Returns:
        List of pd.Series (one per output LiteFrame), or None if not parseable.
    """
    if isinstance(annotation, LiteFrameType):
        return [annotation.to_dtypes()]

    # Normalize typing.Tuple[...] and bare tuple into a uniform tuple of items.
    items = None
    if isinstance(annotation, tuple):
        items = annotation
    elif getattr(annotation, "__origin__", None) is tuple:
        items = getattr(annotation, "__args__", None)

    if items is not None:
        results = []
        for item in items:
            if isinstance(item, LiteFrameType):
                results.append(item.to_dtypes())
            else:
                return None
        return results if results else None

    return None


def build_mock_value(pa_type) -> object:
    """Return a representative scalar value for the given Arrow type.

    Used for mock-based type inference in element-wise (map) and
    DataFrame-wise contexts.  Nested types (list, large_list,
    fixed_size_list, struct, map) are generated recursively so a UDF
    that indexes a list element or struct field can run against the
    mock frame and still produce an inferable result.
    """
    # Nested types are handled first and recurse into the value/item/field
    # types so UDFs that dereference list elements or struct fields (e.g.
    # row["l"][0], row["s"]["a"]) get a representative value at inference
    # time instead of <NA>, which would make inference fail.
    if pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
        return [build_mock_value(pa_type.value_type)]
    elif pa.types.is_fixed_size_list(pa_type):
        return [build_mock_value(pa_type.value_type)] * pa_type.list_size
    elif pa.types.is_struct(pa_type):
        return {
            pa_type.field(i).name: build_mock_value(pa_type.field(i).type)
            for i in range(pa_type.num_fields)
        }
    elif pa.types.is_map(pa_type):
        return [
            (
                build_mock_value(pa_type.key_type),
                build_mock_value(pa_type.item_type),
            )
        ]
    if pa.types.is_integer(pa_type):
        return 1
    elif pa.types.is_floating(pa_type):
        return 1.0
    elif pa.types.is_boolean(pa_type):
        return True
    elif pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
        return "a"
    elif pa.types.is_date(pa_type):
        return datetime.date.today()
    elif pa.types.is_timestamp(pa_type):
        return pd.Timestamp.now()
    else:
        return None


def build_mock_dataframe(
    input_dtypes: pd.Series, n_rows: int = 2, input_frame_type: str = "pandas"
):
    """Build a small mock DataFrame for mock-based type inference.

    Uses Arrow-backed dtypes to match LiteFrame semantics.  When
    ``input_frame_type='polars'`` a ``polars.DataFrame`` is returned so the
    UDF receives the same frame type it will see at execution time (the API
    promises polars UDFs get a polars frame); otherwise a ``pandas.DataFrame``
    is returned.
    """
    data = {}
    for col_name, dtype in input_dtypes.items():
        pa_type = _get_pa_type(dtype)
        if ExternalBlobDtype is not None and isinstance(dtype, ExternalBlobDtype):
            mock_blobs = [SolidBlob(b"")] * n_rows
            data[col_name] = ExternalBlobExtensionArray(mock_blobs)
        else:
            mock_val = build_mock_value(pa_type)
            # FakeArrowExtensionArray (pandas < 1.5 fallback) does not implement
            # arithmetic, so mock-based inference crashes on UDFs like
            # ``d["a"] + d["b"]``.  Build the mock column with the numpy
            # equivalent instead -- enough to infer the result dtype.
            mock_dtype = (
                _to_numpy_dtype(dtype) if _is_fake_arrow_dtype(dtype) else dtype
            )
            data[col_name] = pd.array([mock_val] * n_rows, dtype=mock_dtype)
    pdf = pd.DataFrame(data)
    if input_frame_type == "polars" and pl is not None:
        # Use the blob-aware pandas->polars bridge so blob columns become
        # PolarsObjectContainerType carrying AbstractExternalBlob objects,
        # matching what UDFs see at execution time.  ``pl.from_pandas``
        # would turn blob columns into plain Binary.
        if pandas_to_polars_with_blobs is not None:
            return pandas_to_polars_with_blobs(pdf)
        return pl.from_pandas(pdf)
    return pdf


def _get_pa_type(dtype):
    """Extract the pyarrow type from a pandas dtype."""
    if ExternalBlobDtype is not None and isinstance(dtype, ExternalBlobDtype):
        return pa.binary()
    if isinstance(dtype, ArrowDtype):
        return dtype.pyarrow_dtype
    try:
        return pa.from_numpy_dtype(dtype)
    except (TypeError, ValueError):
        return pa.string()


def _get_return_annotation(func):
    """Get the return type annotation from a function, handling MarkedFunction.

    Follows ``functools.wraps`` / inference-wrapper chains (via
    ``unwrap_function``) back to the user's original callable so type-hint
    names resolve against the user's own module globals, not the wrapper's.
    """
    # Descend through partial/__wrapped__ (a bound CustomAgg hook arrives
    # as functools.partial(MarkedFunction, instance)) — inspect.unwrap
    # alone does not cross partial, so use unwrap_function.
    actual_func = unwrap_function(func)
    if isinstance(actual_func, MarkedFunction):
        actual_func = actual_func.func
    if isinstance(actual_func, type):
        # Callable class: check __call__ method
        actual_func = actual_func.__call__

    try:
        hints = get_type_hints(actual_func)
    except Exception:
        return None
    return hints.get("return", None)


def infer_output_dtypes(
    func,
    input_dtypes: pd.Series,
    dtypes=None,
    skip_infer: bool = False,
    args: tuple = (),
    kwargs: dict = None,
    input_frame_type: str = "pandas",
) -> Optional[List[pd.Series]]:
    """Infer output schema for a UDF.

    Priority:
    1. Explicit dtypes argument
    2. Return type annotation
    3. Mock-based inference
    4. Error (unless skip_infer=True)

    Returns:
        List of pd.Series (one per output), or None if skip_infer and
        inference fails.
    """
    kwargs = kwargs or {}

    # 1. Explicit dtypes
    if dtypes is not None:
        return _parse_explicit_dtypes(dtypes)

    # 2. Return annotation
    annotation = _get_return_annotation(func)
    if annotation is not None:
        parsed = parse_return_annotation(annotation)
        if parsed is not None:
            return parsed

    # 3. Mock inference
    if not skip_infer:
        try:
            mock_df = build_mock_dataframe(
                input_dtypes, input_frame_type=input_frame_type
            )
            # Callable classes are instantiated once, called with end=True
            # (mirroring make_callable_wrapper), and close() runs in finally;
            # a decorated method arrives as a partial that binds the instance,
            # so unwrap MarkedFunction without losing the binding.
            actual_func = func
            if isinstance(func, MarkedFunction):
                actual_func = func.func
            if isinstance(actual_func, type):
                instance = actual_func()
                sig = inspect.signature(instance.__call__)
                has_end = "end" in sig.parameters
                try:
                    if has_end:
                        result = instance(mock_df, *args, end=True, **kwargs)
                    else:
                        result = instance(mock_df, *args, **kwargs)
                finally:
                    if hasattr(instance, "close"):
                        instance.close()
            else:
                result = actual_func(mock_df, *args, **kwargs)

            if isinstance(result, tuple):
                # Multiple outputs
                return [_extract_dtypes_from_df(r) for r in result]
            else:
                return [_extract_dtypes_from_df(result)]
        except Exception:
            pass

    if skip_infer:
        return None

    raise TypeError(
        "Cannot infer output schema for UDF. Please specify 'dtypes' explicitly "
        "or add a return type annotation."
    )


def _canonicalize_dtypes_series(dtypes: pd.Series) -> pd.Series:
    """Canonicalize a single output's dtype ``Series`` (boundary contract).

    LiteFrame column names are always strings, so an explicit ``dtypes`` dict
    with a numeric key (``{0: int}``) or a positional list of bare dtypes
    (which receives an integer index from ``make_dtypes``) must be
    stringified to ``"0"`` before the schema is stored on the operator --
    matching the string-labeled runtime output and the ODPS round trip.
    Collisions introduced by canonicalization (e.g. ``{0: int, "0": float}``
    collapsing to duplicate ``"0"``) are rejected, since a duplicate label
    makes the schema ambiguous.
    """
    if dtypes is None or not isinstance(dtypes, pd.Series) or len(dtypes) == 0:
        return dtypes
    str_index = [str(c) for c in dtypes.index]
    dupes = {c for c in str_index if str_index.count(c) > 1}
    if dupes:
        raise ValueError(
            f"Duplicate column labels after string normalization in "
            f"declared dtypes: {sorted(dupes)}. Distinct labels (e.g. int 0 "
            "and str '0') collapsed onto the same name."
        )
    if str_index != list(dtypes.index):
        dtypes = dtypes.copy()
        dtypes.index = str_index
    return dtypes


def _parse_explicit_dtypes(dtypes) -> List[pd.Series]:
    """Parse explicit dtypes argument into list of pd.Series.

    Handles:
    - dict/pd.Series -> single output
    - list of ``(name, dtype)`` tuples -> single output (named schema)
    - list of dicts/pd.Series -> multiple outputs (one schema per output)
    - list of bare dtypes -> single output (positional, integer index)
    """
    if isinstance(dtypes, list) and dtypes:
        # Multiple outputs: a list of dicts/pd.Series, one schema per output.
        # This is distinct from a list of (name, dtype) tuples, which denotes a
        # *single* output's named schema.
        if isinstance(dtypes[0], (dict, pd.Series)):
            return [_canonicalize_dtypes_series(make_dtypes(d)) for d in dtypes]
        # Single output as a list of (name, dtype) tuples -> named schema.
        # Require *every* element to be a 2-tuple so this does not collide with
        # a positional list of bare dtypes (handled by make_dtypes with an
        # integer index) or with the multi-output list-of-dicts form above.
        if all(isinstance(t, tuple) and len(t) == 2 for t in dtypes):
            return [
                _canonicalize_dtypes_series(
                    make_dtypes({name: dtype for name, dtype in dtypes})
                )
            ]
    # Single output (dict/pd.Series/positional list of bare dtypes/scalar)
    return [_canonicalize_dtypes_series(make_dtypes(dtypes))]


def _extract_dtypes_from_df(df) -> pd.Series:
    if isinstance(df, pd.DataFrame):
        # LiteFrame column names are always strings (boundary contract); a
        # UDF returning ``pd.DataFrame({0: ...})`` carries an integer label
        # that must be stringified to "0".  Reuse the explicit-dtypes
        # canonicalizer so mock-inferred schemas share the same boundary
        # rule (and collision guard) as declared schemas; this covers
        # ``transform_chunk`` (which calls ``infer_output_dtypes`` directly,
        # without the mode-specific wrappers in ``wrap_inference_func``).
        return _canonicalize_dtypes_series(df.dtypes)
    if pl is not None and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        # Bridge through pyarrow so the inferred dtypes are Arrow-backed,
        # matching LiteFrame semantics (the same convention
        # build_mock_dataframe uses).  Without this, a polars UDF result is
        # treated as "something else" and inference silently returns empty.
        try:
            eager = df.collect() if isinstance(df, pl.LazyFrame) else df
            dtypes = {}
            for col in eager.columns:
                if is_polars_blob_type is not None and is_polars_blob_type(
                    eager[col].dtype
                ):
                    # Blob columns must keep ExternalBlobDtype; the Arrow
                    # bridge would flatten them to ArrowDtype(pa.binary()).
                    dtypes[col] = ExternalBlobDtype()
                else:
                    pa_type = eager[col].to_arrow().type
                    dtypes[col] = ArrowDtype(pa_type)
            return pd.Series(dtypes)
        except Exception:
            return pd.Series(dtype="object")
    # If it's something else, try to convert
    return pd.Series(dtype="object")
