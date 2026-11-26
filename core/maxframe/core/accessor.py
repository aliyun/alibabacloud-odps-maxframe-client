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

import functools
import sys
import warnings
from typing import Callable, Dict, Tuple, Type, TypeVar

T = TypeVar("T")


class BaseMaxFrameAccessor:
    def __init__(self, obj):
        self.obj = obj
        validate = getattr(self, "_validate", None)
        if validate is not None:
            validate(obj)

    @classmethod
    def _register(cls, name, func, is_property=False):
        @functools.wraps(func)
        def wrapped(self, *args, **kw):
            return func(self.obj, *args, **kw)

        wrapped.__name__ = name
        setattr(cls, name, wrapped if not is_property else property(wrapped))
        if hasattr(cls, "_api_count"):  # pragma: no branch
            cls._api_count += 1


class CachedAccessor:
    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor
        try:
            cache = obj._accessors
        except AttributeError:
            # tileables deserialized on the client never carry the accessor
            # cache, so it is created on first use
            cache = {}
            obj._accessors = cache
        if self._name not in cache:
            cache[self._name] = self._accessor(obj)
        return cache[self._name]


_ACCESSOR_REGISTRY: Dict[Tuple[Tuple[Type, ...], str], Type] = {}


def _find_declared_attr(cls, name):
    for base in cls.__mro__:
        if name in vars(base):
            return base
    return None


def _is_current_module_binding(accessor: Type) -> bool:
    """Whether *accessor* is what its ``(module, qualname)`` resolves to now.

    A reloaded module rebinds the class before registering it (function-call
    form), so the new class is reachable from its module; a dynamically
    generated class reusing another class's metadata is not.
    """
    qualname = accessor.__qualname__
    if "<locals>" in qualname:
        return False
    module = sys.modules.get(accessor.__module__)
    if module is None:
        return False
    obj = module
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    return obj is accessor


def _same_accessor(first: Type, second: Type) -> bool:
    """Whether *second* is a re-registration of the same accessor as *first*.

    True for the same object, and for a module-level class that the module
    currently binds at the same origin as *first* (a function-call reload
    rebinds the new class before registering it).  Any other class sharing the
    metadata -- a function-local class, a decorator's not-yet-assigned class,
    or a dynamically generated class copying another class's path -- stays
    distinct.
    """
    if first is second:
        return True
    same_origin = (
        first.__module__ == second.__module__
        and first.__qualname__ == second.__qualname__
    )
    return same_origin and _is_current_module_binding(second)


def register_accessor(
    name: str, obj_types: Tuple[Type, ...], override: bool = False
) -> Callable[[Type[T]], Type[T]]:
    """
    Register a custom accessor under ``name`` on the given tileable classes.

    The accessor is reached as an attribute of the objects in `obj_types`, e.g.
    ``df.<name>``. It is instantiated once per object and cached there, so
    repeated access returns the same instance.

    Parameters
    ----------
    name : str
        Name the accessor is registered under. Must be a valid identifier that
        does not start with an underscore.
    obj_types : tuple of type
        Classes the accessor is installed on. Pass a type tuple such as
        ``SERIES_TYPE`` so that the public tileable and its data object are
        both covered.
    override : bool, default False
        Replace an accessor previously registered by this function under the
        same name for the same classes. Attributes owned by MaxFrame itself
        cannot be overridden, `override` or not.

    Returns
    -------
    callable
        A class decorator that installs the accessor and returns the class
        unchanged.

    Raises
    ------
    ValueError
        If `name` is not a valid identifier or starts with an underscore.
    AttributeError
        If `name` is already registered for `obj_types` by a *different*
        accessor while `override` is False, or if `name` already exists as an
        attribute of one of the classes in `obj_types`. Re-registering the same
        accessor is idempotent and does not raise only when it is the very
        object registered before, or a module-level class that the module
        currently binds at that same path (a function-call reload); a
        function-local, decorator-form, or dynamically generated class
        collides like any other unless `override` is True.

    See Also
    --------
    register_dataframe_accessor : Register a custom accessor on DataFrame objects.
    register_liteframe_accessor : Register a custom accessor on LiteFrame objects.
    register_tensor_accessor : Register a custom accessor on Tensor objects.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> from maxframe.dataframe.api.extensions import register_dataframe_accessor
    >>> @register_dataframe_accessor("geo")
    ... class GeoAccessor:
    ...     def __init__(self, obj):
    ...         if "geometry" not in obj.columns:
    ...             raise AttributeError("no geometry column")
    ...         self.obj = obj
    ...
    ...     @property
    ...     def area(self):
    ...         return self.obj.geometry.area()
    """
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError(
            f"invalid accessor name {name!r}: it must be a valid identifier "
            f"that does not start with an underscore"
        )
    obj_types = tuple(obj_types)

    def decorator(accessor: Type[T]) -> Type[T]:
        registered = _ACCESSOR_REGISTRY.get((obj_types, name))
        if registered is None:
            for cls in obj_types:
                owner = _find_declared_attr(cls, name)
                if owner is not None:
                    raise AttributeError(
                        f"cannot override existing attribute "
                        f"{owner.__name__}.{name}; choose another accessor name"
                    )
        elif not _same_accessor(registered, accessor):
            # Re-registering the same accessor (e.g. a module reloaded by the
            # import path) is idempotent: same object, or two module-level
            # classes at the same origin, so it stays a silent no-op refresh
            # rather than a duplicate error.  A different provider claiming the
            # namespace still collides.
            if not override:
                raise AttributeError(
                    f"accessor {name!r} is already registered for "
                    f"{[cls.__name__ for cls in obj_types]}; pass override=True "
                    f"to replace it"
                )
            warnings.warn(
                f"overriding existing accessor {name!r} (on "
                f"{[cls.__name__ for cls in obj_types]})",
                UserWarning,
                stacklevel=3,
            )

        descriptor = CachedAccessor(name, accessor)
        for cls in obj_types:
            setattr(cls, name, descriptor)
        _ACCESSOR_REGISTRY[(obj_types, name)] = accessor
        return accessor

    return decorator
