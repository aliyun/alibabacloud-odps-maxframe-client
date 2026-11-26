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

import sys
import types

import pytest

import maxframe.dataframe as md
import maxframe.liteframe as mlf
import maxframe.tensor as mt
from maxframe.core import accessor as accessor_module
from maxframe.core.accessor import (
    BaseMaxFrameAccessor,
    CachedAccessor,
    register_accessor,
)
from maxframe.dataframe.api import extensions as df_ext
from maxframe.dataframe.core import (
    DATAFRAME_TYPE,
    INDEX_TYPE,
    SERIES_TYPE,
    register_dataframe_accessor,
    register_index_accessor,
    register_series_accessor,
)
from maxframe.liteframe.api import extensions as lf_ext
from maxframe.liteframe.core import LITEFRAME_TYPE, register_liteframe_accessor
from maxframe.serialization import deserialize, serialize
from maxframe.tensor.api import extensions as tensor_ext
from maxframe.tensor.core import TENSOR_TYPE, register_tensor_accessor

# Classes every accessor registration may install a descriptor on.  All
# registrations target one of the public/data type tuples.
ACCESSOR_TARGET_CLASSES = tuple(
    dict.fromkeys(
        DATAFRAME_TYPE + SERIES_TYPE + INDEX_TYPE + LITEFRAME_TYPE + TENSOR_TYPE
    )
)


def _cached_accessors(cls) -> dict:
    return {
        name: value
        for name, value in vars(cls).items()
        if isinstance(value, CachedAccessor)
    }


@pytest.fixture
def isolated_registry():
    # The registry is process-global module state, and registering also
    # installs a descriptor permanently on the target classes.  Snapshot and
    # restore both so a name registered by one test does not leak into the
    # next (a leaked descriptor rejects the same name as an existing
    # attribute once the registry has been restored).
    saved_registry = dict(accessor_module._ACCESSOR_REGISTRY)
    saved_attrs = {cls: _cached_accessors(cls) for cls in ACCESSOR_TARGET_CLASSES}
    try:
        yield
    finally:
        accessor_module._ACCESSOR_REGISTRY.clear()
        accessor_module._ACCESSOR_REGISTRY.update(saved_registry)
        for cls, attrs in saved_attrs.items():
            for name in _cached_accessors(cls).keys() - attrs.keys():
                delattr(cls, name)
            for name, value in attrs.items():
                setattr(cls, name, value)


def _simple_accessor():
    class Accessor:
        def __init__(self, obj):
            self.obj = obj

    return Accessor


def _versioned_accessor(version):
    class Versioned:
        def __init__(self, obj):
            self.obj = obj

        def version(self):
            return version

    return Versioned


FRAME_CASES = [
    (
        "dataframe",
        register_dataframe_accessor,
        DATAFRAME_TYPE,
        lambda: md.DataFrame({"a": [1, 2, 3]}),
    ),
    ("series", register_series_accessor, SERIES_TYPE, lambda: md.Series([1, 2, 3])),
    (
        "index",
        register_index_accessor,
        INDEX_TYPE,
        lambda: md.DataFrame({"a": [1, 2, 3]}).index,
    ),
    (
        "liteframe",
        register_liteframe_accessor,
        LITEFRAME_TYPE,
        lambda: mlf.LiteFrame({"a": [1, 2, 3]}),
    ),
    ("tensor", register_tensor_accessor, TENSOR_TYPE, lambda: mt.tensor([1, 2, 3])),
]
FRAME_IDS = [name for name, *_ in FRAME_CASES]


@pytest.mark.parametrize(
    "name, register, obj_types, make_obj", FRAME_CASES, ids=FRAME_IDS
)
def test_register_accessor_attaches_and_caches(
    name, register, obj_types, make_obj, isolated_registry
):
    accessor_name = f"acc_{name}"
    Accessor = _simple_accessor()
    register(accessor_name)(Accessor)

    obj = make_obj()
    assert isinstance(obj, obj_types)
    accessor = getattr(obj, accessor_name)
    assert isinstance(accessor, Accessor)
    assert accessor.obj is obj
    # instance is cached on the object, class access yields the accessor class
    assert getattr(obj, accessor_name) is accessor
    assert getattr(type(obj), accessor_name) is Accessor
    # the descriptor covers the internal data object as well
    assert isinstance(getattr(obj._data, accessor_name), Accessor)


@pytest.mark.parametrize(
    "name, register, obj_types, make_obj", FRAME_CASES, ids=FRAME_IDS
)
def test_accessor_survives_deserialization(
    name, register, make_obj, obj_types, isolated_registry
):
    accessor_name = f"deser_{name}"
    # the accessor cache is not part of the serialized payload, so the
    # descriptor must recreate it on the deserialized data object
    Accessor = _simple_accessor()
    register(accessor_name)(Accessor)

    data = deserialize(*serialize(make_obj()._data))
    assert not hasattr(data, "_accessors") or not data._accessors
    accessor = getattr(data, accessor_name)
    assert isinstance(accessor, Accessor)
    assert accessor.obj is data
    assert data._accessors[accessor_name] is accessor


def test_attribute_error_renders_accessor_unsupported(isolated_registry):
    @register_series_accessor("float_only")
    class FloatOnlyAccessor(BaseMaxFrameAccessor):
        def _validate(self, obj):
            if obj.dtype.kind != "f":
                raise AttributeError("float series only")

        def doubled(self):
            return self.obj * 2

    floats = md.Series([1.0, 2.0])
    doubled = floats.float_only.doubled()
    assert isinstance(doubled, md.Series)
    assert doubled.shape == (2,)

    ints = md.Series([1, 2])
    assert not hasattr(ints, "float_only")


def test_accessor_composes_with_operators(isolated_registry):
    @register_dataframe_accessor("derived")
    class DerivedAccessor:
        def __init__(self, obj):
            self.obj = obj

        def plus_one(self, column):
            return self.obj[column] + 1

    df = md.DataFrame({"a": [1, 2, 3]})
    result = df.derived.plus_one("a")
    assert isinstance(result, md.Series)
    assert result.shape == (3,)
    # the cache is per object, so a second read sees the same accessor
    assert df.derived.plus_one("a").op is not None


@pytest.mark.parametrize("bad", ["_acc", "a b", "1acc", "", None])
def test_invalid_accessor_name(bad, isolated_registry):
    with pytest.raises(ValueError, match="invalid accessor name"):
        register_accessor(bad, SERIES_TYPE)


def _accessor_class(module_name, qualname="Reloadable", bind_module=False):
    # A dynamic class carrying ``module_name``/``qualname``; with
    # ``bind_module`` it is additionally bound at ``qualname`` on a throwaway
    # module, which is how a function-call reload looks.
    cls = type(qualname, (), {"__init__": lambda self, obj: setattr(self, "obj", obj)})
    cls.__module__ = module_name
    if bind_module:
        module = sys.modules.setdefault(module_name, types.ModuleType(module_name))
        setattr(module, qualname, cls)
    return cls


def _bind_accessor(module_name, qualname="Reloadable"):
    # A class a throwaway module binds at ``qualname``.
    return _accessor_class(module_name, qualname, bind_module=True)


def test_module_origin_reregistration(isolated_registry):
    # A function-call reload rebinds the new class before registering it, so
    # re-registering is a no-op; an unbound class copying the same path is not
    # a reload and must collide.
    module_name = "_mf_reload_accessor_mod"
    try:
        register_series_accessor("origin")(_bind_accessor(module_name))
        register_series_accessor("origin")(_bind_accessor(module_name))

        assert md.Series([1]).origin.obj is not None

        with pytest.raises(AttributeError, match="already registered"):
            register_series_accessor("origin")(_accessor_class(module_name))
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize(
    "make",
    [_simple_accessor, lambda: _accessor_class(__name__, "DynAccessor")],
    ids=["local", "dynamic"],
)
def test_distinct_provider_reregistration_collides(make, isolated_registry):
    # Two distinct classes sharing module/qualname (a function-local class, or
    # an unbound dynamic class) are unrelated providers, so the second
    # registration must collide rather than silently replace the first.
    register_series_accessor("clone")(make())
    with pytest.raises(AttributeError, match="already registered"):
        register_series_accessor("clone")(make())


def test_duplicate_registration_requires_override(isolated_registry):
    First = _versioned_accessor(1)
    Second = _versioned_accessor(2)

    register_series_accessor("versioned")(First)
    with pytest.raises(AttributeError, match="already registered"):
        register_series_accessor("versioned")(Second)
    with pytest.warns(UserWarning, match="overriding existing accessor"):
        register_series_accessor("versioned", override=True)(Second)

    # a fresh object picks the replacement up; already cached instances keep
    # the class they were created with
    assert md.Series([1]).versioned.version() == 2


@pytest.mark.parametrize(
    "register",
    [
        register_dataframe_accessor,
        register_series_accessor,
        register_index_accessor,
        register_liteframe_accessor,
        register_tensor_accessor,
    ],
    ids=FRAME_IDS,
)
def test_cannot_override_existing_attribute(register, isolated_registry):
    Clashing = _simple_accessor()

    with pytest.raises(AttributeError, match="cannot override existing attribute"):
        register("execute")(Clashing)

    with pytest.raises(AttributeError, match="cannot override existing attribute"):
        register("execute", override=True)(Clashing)


def test_api_shortcut_reexports_core_registrator():
    # The per-data-type ``api.extensions`` modules are plain re-exports of the
    # registrators defined next to the type tuples in the data-type core
    # modules, so the shortcut name and the direct name must be the same
    # object (hence one registry).
    assert df_ext.register_dataframe_accessor is register_dataframe_accessor
    assert df_ext.register_series_accessor is register_series_accessor
    assert df_ext.register_index_accessor is register_index_accessor
    assert lf_ext.register_liteframe_accessor is register_liteframe_accessor
    assert tensor_ext.register_tensor_accessor is register_tensor_accessor
