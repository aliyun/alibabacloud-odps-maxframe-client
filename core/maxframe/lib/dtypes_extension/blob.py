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

from ...utils import tokenize
from .dtypes import ArrowDtype


class ArrowBlobType(pa.ExtensionType):
    def __init__(self):
        super().__init__(pa.binary(), "maxframe.blob")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return ArrowBlobType()

    def __eq__(self, other):
        return isinstance(other, ArrowBlobType)

    def __hash__(self):
        return hash(str(self))

    def to_pandas_dtype(self):
        return ExternalBlobDtype()


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

    @abc.abstractmethod
    def open(self, mode: str = "r") -> ContextManager[io.IOBase]:
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    def __copy__(self):
        return self.copy()

    @classmethod
    def register(cls):
        cls._blob_types[cls.__name__.lower()] = cls

    @classmethod
    def get_cls_by_name(cls, cls_name: str):
        return cls._blob_types[cls_name.lower()]


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

    def copy(self) -> "SolidBlob":
        return SolidBlob(self._reference)

    def __maxframe_tokenize__(self):
        return type(self), self._reference


SolidBlob.register()


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
        return ExternalBlobExtensionArray(
            [blob_from_reference(a) for a in array.to_pylist()]
        )


class ExternalBlobExtensionArray(ExtensionArray):
    """
    Extension array for storing AbstractExternalBlob objects.
    """

    _data: List[AbstractExternalBlob]

    def __init__(self, values: Union[List[AbstractExternalBlob], np.ndarray]):
        if not isinstance(values, (list, np.ndarray)):
            raise TypeError("values must be a list or numpy array")

        if isinstance(values, np.ndarray):
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
            pa.array([x.reference for x in self._data], type=pa.binary()),
        )

    def astype(self, dtype, copy=True):
        if isinstance(dtype, ArrowDtype) and dtype.pyarrow_dtype == pa.binary():
            blobs = [None] * len(self)
            for i, blob in enumerate(self._data):
                with blob.open("rb") as reader:
                    blobs[i] = reader.read()
            return pd.array(blobs, dtype=dtype)
        return super().astype(dtype, copy=copy)
