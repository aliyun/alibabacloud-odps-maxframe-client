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

import abc
import contextlib
import io
from typing import Any, ContextManager, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_extension_dtype,
)

from maxframe.lib.dtypes_extension.dtypes import ArrowDtype
from maxframe.utils import tokenize
from maxframe.utils.polars import (
    PolarsObjectContainerType,
    is_polars_object_container_type,
)

try:
    import polars as pl
except ImportError:
    pl = None


class ArrowBlobType(pa.ExtensionType):
    type_str = "maxframe.blob"

    def __init__(self):
        super().__init__(pa.binary(), self.type_str)

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return ArrowBlobType()

    def __eq__(self, other):
        return isinstance(other, ArrowBlobType)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.type_str

    def to_pandas_dtype(self):
        return ExternalBlobDtype()


pa.register_extension_type(ArrowBlobType())


class AbstractExternalBlob(metaclass=abc.ABCMeta):
    _blob_types = {}

    @property
    def reference(self) -> Any:
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        if isinstance(other, AbstractExternalBlob):
            return self.reference == other.reference
        else:
            return False

    def __hash__(self):
        return hash((type(self), self.reference))

    def __repr__(self):
        return f"<{type(self).__name__} reference={self.reference}>"

    def __reduce__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def open(self, mode: str = "r") -> ContextManager[io.IOBase]:
        raise NotImplementedError

    def copy(self):
        tp, args = self.__reduce__()
        return tp(*args)

    def __copy__(self):
        return self.copy()

    @classmethod
    def register(cls):
        cls._blob_types[cls.__name__.lower()] = cls

    @classmethod
    def get_cls_by_name(cls, cls_name: str):
        return cls._blob_types[cls_name.lower()]

    def to_serializable(self):
        """
        Called by MaxFrame serializer to convert the blob into new objects
        """
        # Return the blob object itself by default for serialization
        return self


class SolidBlob(AbstractExternalBlob):
    def __init__(self, content: Optional[bytes] = None):
        self._reference = content

    @property
    def reference(self) -> bytes:
        return self._reference

    def __reduce__(self):
        return type(self), (self._reference,)

    @contextlib.contextmanager
    def open(self, mode: str = "r"):
        if "w" in mode:
            sio = io.BytesIO()
        else:
            sio = io.BytesIO(self._reference)
        try:
            yield sio
        except Exception:
            raise
        else:
            if "w" in mode:
                self._reference = sio.getvalue()

    def __maxframe_tokenize__(self):
        return type(self), self._reference


SolidBlob.register()


def ensure_blob_value(v):
    """Convert a single value to an AbstractExternalBlob or None.

    - None → None
    - AbstractExternalBlob → as-is
    - str → SolidBlob(v.encode("utf-8"))
    - bytes/bytearray → SolidBlob(bytes(v))
    - else → SolidBlob(bytes(v))
    """
    if v is None:
        return None
    if isinstance(v, AbstractExternalBlob):
        return v
    if isinstance(v, str):
        return SolidBlob(v.encode("utf-8"))
    if isinstance(v, (bytes, bytearray)):
        return SolidBlob(bytes(v))
    return SolidBlob(bytes(v))


def new_blob(content: Optional[bytes] = None):
    # todo this function currently maps directly to SolidBlob. It should
    #  provide appropriate instance base on running envs.
    return SolidBlob(content)


def blob_from_reference(reference: Optional[bytes] = None):
    # todo this function currently maps directly to SolidBlob. It should
    #  provide appropriate instance base on running envs.
    return SolidBlob(reference)


@register_extension_dtype
class ExternalBlobDtype(ExtensionDtype):
    """
    Extension dtype for ExternalBlob data.
    """

    name = "blob"

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.
        """
        return ExternalBlobExtensionArray

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    @property
    def type(self):
        return AbstractExternalBlob

    def __repr__(self) -> str:
        return self.name

    def __from_arrow__(self, array):
        # ``array`` is a pa.ExtensionArray whose storage is pa.binary(); null
        # slots are Arrow ``None``.  ``blob_from_reference(None)`` would yield
        # ``SolidBlob(None)`` (a non-null blob wrapping empty content), losing
        # the null, so map null slots to ``None`` directly.
        return ExternalBlobExtensionArray(
            [
                blob_from_reference(a) if a is not None else None
                for a in array.to_pylist()
            ]
        )


class ExternalBlobExtensionArray(ExtensionArray):
    """
    Extension array for storing AbstractExternalBlob objects.
    """

    _data: List[AbstractExternalBlob]

    def __init__(self, values: Union[List[AbstractExternalBlob], np.ndarray]):
        if not isinstance(
            values, (list, np.ndarray, pd.api.extensions.ExtensionArray, pa.Array)
        ):
            raise TypeError(
                "values must be a list, a numpy array, a pandas ExtensionArray or an Arrow array"
            )

        if isinstance(values, (pa.Array, pd.api.extensions.ExtensionArray)):
            values = values.tolist()
        elif isinstance(values, np.ndarray):
            if values.ndim != 1:
                raise ValueError("values must be a 1-dimensional array")
            values = values.tolist()

        new_values = [None] * len(values)
        for idx, val in enumerate(values):
            if isinstance(val, str):
                val = val.encode()
            if isinstance(val, (bytes, bytearray)):
                val = new_blob(val)

            if pd.isna(val):
                val = None
            if val is not None and not isinstance(val, AbstractExternalBlob):
                raise TypeError(
                    "All values must be instances of AbstractExternalBlob or None"
                )
            else:
                new_values[idx] = val

        # Store with python list instead of numpy array
        self._data = new_values
        self._dtype = ExternalBlobDtype()

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy: bool = False):
        """
        Construct a new ExtensionArray from a sequence of scalars.
        """
        return cls(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        """
        Reconstruct an ExtensionArray after factorization.
        """
        return cls(values)

    @property
    def dtype(self) -> ExtensionDtype:
        """Return the dtype of the array."""
        return self._dtype

    def __len__(self) -> int:
        """Return length of the array."""
        return len(self._data)

    def __getitem__(self, item):
        """Select a subset of self."""
        if isinstance(item, int):
            return self._data[item]
        else:
            # For slices and other index types, return a new array with the selected items
            return type(self)([self._data[i] for i in np.arange(len(self._data))[item]])

    def __maxframe_tokenize__(self):
        return type(self), tokenize(*self._data)

    def isna(self):
        """A 1-D array indicating if each value is missing."""
        return np.array([x is None for x in self._data])

    def copy(self):
        """Return a copy of the array."""
        return type(self)(self._data.copy())

    @property
    def nbytes(self) -> int:
        """The number of bytes needed to store this object in memory."""
        # Since we're storing references, we'll return a minimal size estimate
        return len(self._data) * 8

    def take(self, indices, allow_fill=False, fill_value=None):
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of int
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.
        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
        """
        if allow_fill:
            # Handle negative indices
            values = []
            for i in indices:
                if i < 0:
                    if not allow_fill:
                        raise IndexError("negative index not allowed")
                    values.append(fill_value)
                else:
                    values.append(self._data[i])
            return type(self)(values)
        else:
            return type(self)([self._data[i] for i in indices])

    @classmethod
    def _concat_same_type(
        cls, to_concat: Sequence["ExternalBlobExtensionArray"]
    ) -> "ExternalBlobExtensionArray":
        return cls([x for sublist in to_concat for x in sublist._data])

    def __arrow_array__(self, type=None):
        return pa.ExtensionArray.from_storage(
            ArrowBlobType(),
            pa.array(
                [x.reference if x is not None else None for x in self._data],
                type=pa.binary(),
            ),
        )

    def astype(self, dtype, copy=True):
        if isinstance(dtype, ArrowDtype) and dtype.pyarrow_dtype == pa.binary():
            blobs = []
            for blob in self._data:
                if blob is None:
                    blobs.append(None)
                else:
                    with blob.open("rb") as reader:
                        blobs.append(reader.read())
            return pd.array(blobs, dtype=dtype)
        return super().astype(dtype, copy=copy)


# ---------------------------------------------------------------------------
# Polars object-container type for blob columns in LiteFrame
# ---------------------------------------------------------------------------

_BLOB_ARROW_TYPE_STR = ArrowBlobType.type_str


def make_polars_blob_type():
    """Construct the PolarsObjectContainerType instance for blob columns.

    The metadata is the stringified Arrow type ("maxframe.blob") of the
    AbstractExternalBlob objects stored in the column.
    """
    if PolarsObjectContainerType is None:
        return None
    return PolarsObjectContainerType(
        "maxframe.object_container", pl.Object, _BLOB_ARROW_TYPE_STR
    )


def is_polars_blob_type(dtype):
    """Check if a Polars dtype is a blob object-container column.

    A blob column is an object-container extension whose metadata equals
    the blob Arrow type string ("maxframe.blob").
    """
    if not is_polars_object_container_type(dtype):
        return False
    return dtype.ext_metadata() == _BLOB_ARROW_TYPE_STR


# ---------------------------------------------------------------------------
# Core-safe blob-aware pandas <--> polars converters
# ---------------------------------------------------------------------------
# These live here (in core/) so ``LiteFrame.to_pandas()`` and the type
# inference mock builder can convert blob columns in a layering-safe
# way.  The executor layer re-exports them for its own import path.


def polars_to_pandas_with_blobs(pl_df):
    """Convert a Polars DataFrame to pandas, handling blob columns.

    Blob columns (``PolarsObjectContainerType`` carrying
    ``AbstractExternalBlob`` objects) are wrapped in
    ``ExternalBlobExtensionArray`` so callers receive proper blob dtype
    values with the ``open()`` API.  Non-blob columns convert normally.
    """
    blob_cols = [c for c in pl_df.columns if is_polars_blob_type(pl_df[c].dtype)]
    if not blob_cols:
        return pl_df.to_pandas(use_pyarrow_extension_array=True)

    non_blob_cols = [c for c in pl_df.columns if c not in blob_cols]
    if non_blob_cols:
        pdf = pl_df.select(non_blob_cols).to_pandas(use_pyarrow_extension_array=True)
    else:
        pdf = pd.DataFrame(index=range(len(pl_df)))

    for col in blob_cols:
        blobs = pl_df[col].to_list()
        pdf[col] = ExternalBlobExtensionArray(blobs)

    return pdf[list(pl_df.columns)]


def pandas_to_polars_with_blobs(pdf):
    """Convert a pandas DataFrame to Polars, handling blob columns.

    ``ExternalBlobExtensionArray`` columns become
    ``PolarsObjectContainerType`` Series (``pl.from_pandas`` cannot handle
    them directly); other columns convert normally.
    """
    needs_conversion = any(isinstance(dt, ExternalBlobDtype) for dt in pdf.dtypes)
    if not needs_conversion:
        return pl.from_pandas(pdf)

    blob_dtype = make_polars_blob_type()
    blob_series = {}
    for col in pdf.columns:
        if isinstance(pdf[col].dtype, ExternalBlobDtype):
            objects = pdf[col].values._data
            blob_series[col] = pl.Series(col, list(objects), dtype=blob_dtype)

    non_blob_col_names = [c for c in pdf.columns if c not in blob_series]
    non_blob_pdf = pdf[non_blob_col_names] if non_blob_col_names else None
    if non_blob_pdf is not None and len(non_blob_pdf.columns) > 0:
        result = pl.from_pandas(non_blob_pdf)
    else:
        result = pl.DataFrame()

    for name, series in blob_series.items():
        result = result.with_columns(series)

    return result.select(list(pdf.columns))
