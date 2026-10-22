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

try:
    import polars as pl
except ImportError:
    pl = None

_HAS_PL_EXTENSION = pl is not None and hasattr(pl, "BaseExtension")

# Extension type name for object-container columns. The Polars storage is
# pl.Object; the *logical* type of the contained objects is carried in the
# extension metadata as a stringified Arrow type (e.g. "maxframe.blob").
_OBJECT_CONTAINER_EXT_NAME = "maxframe.object_container"


def has_pl_object_type(dtype):
    """Check if a Polars dtype holds Python objects (pl.Object).

    This covers:
      - pl.Object dtype directly
      - pl.BaseExtension dtypes backed by pl.Object storage
    """
    if pl is None:
        return False
    if dtype == pl.Object:
        return True
    if _HAS_PL_EXTENSION and isinstance(dtype, pl.BaseExtension):
        ext_storage = getattr(dtype, "ext_storage", None)
        ext_storage_val = ext_storage() if callable(ext_storage) else None
        if ext_storage_val == pl.Object:
            return True
    return False


def pl_storage_to_str(storage):
    """Convert a Polars storage dtype to a string for serialization."""
    if pl is None or storage is None:
        return None
    if storage == pl.Object:
        return "Object"
    return str(storage)


def pl_str_to_storage(storage_str):
    """Convert a storage string back to a Polars dtype."""
    if pl is None:
        return None
    if storage_str is None or storage_str == "Object":
        return pl.Object
    return pl.Object


def pl_get_ext_type_name_and_storage(dtype):
    """Extract extension type name and storage string from a Polars dtype.

    Returns (ext_name, storage_str) where both are strings or None.
    """
    ext_name = None
    storage_str = None
    if _HAS_PL_EXTENSION and isinstance(dtype, pl.BaseExtension):
        if hasattr(dtype, "ext_name"):
            ext_name = dtype.ext_name()
        if hasattr(dtype, "ext_storage"):
            storage_str = pl_storage_to_str(dtype.ext_storage())
    return ext_name, storage_str


def pl_get_ext_type_metadata(dtype):
    """Extract the metadata string from a Polars extension dtype.

    Returns None for non-extension or missing metadata.
    """
    if _HAS_PL_EXTENSION and isinstance(dtype, pl.BaseExtension):
        if hasattr(dtype, "ext_metadata"):
            return dtype.ext_metadata()
    return None


def pl_make_ext_or_object_dtype(ext_name, storage_str, metadata=None):
    """Reconstruct the extension dtype from metadata.

    Returns the reconstructed extension type if ext_name is known,
    otherwise falls back to pl.Object.
    """
    if ext_name is not None and _HAS_PL_EXTENSION:
        ext_cls = pl.get_extension_type(ext_name)
        if ext_cls is not None:
            return ext_cls(ext_name, pl_str_to_storage(storage_str), metadata)
    return pl.Object


if _HAS_PL_EXTENSION:

    class PolarsObjectContainerType(pl.BaseExtension):
        """Polars extension type for columns that store arbitrary Python
        objects in ``pl.Object`` storage.

        The extension metadata carries the stringified Arrow type of the
        *logical* values contained in the column (e.g. ``"maxframe.blob"``),
        so the column is self-describing across serialization boundaries.
        """

        pass

    def register_polars_object_container_type():
        """Register the object-container extension type with Polars. Idempotent."""
        if pl.get_extension_type(_OBJECT_CONTAINER_EXT_NAME) is None:
            pl.register_extension_type(
                _OBJECT_CONTAINER_EXT_NAME, PolarsObjectContainerType
            )

    register_polars_object_container_type()

    def is_polars_object_container_type(dtype):
        """Check if a Polars dtype is an object-container extension type."""
        if not isinstance(dtype, pl.BaseExtension):
            return False
        if not hasattr(dtype, "ext_name") or not hasattr(dtype, "ext_storage"):
            raise SystemError(
                f"Polars BaseExtension dtype {dtype!r} is missing ext_name() or "
                f"ext_storage() methods. This may indicate an incompatible Polars version."
            )
        return dtype.ext_name() == _OBJECT_CONTAINER_EXT_NAME

else:
    PolarsObjectContainerType = None
    register_polars_object_container_type = None

    def is_polars_object_container_type(dtype):
        return False
