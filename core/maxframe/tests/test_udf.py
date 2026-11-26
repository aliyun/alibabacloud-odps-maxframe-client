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
import inspect
import textwrap

import pytest
from odps import ODPS
from odps.errors import NoSuchObject

from maxframe.config import option_context
from maxframe.core.operator import Operator
from maxframe.tests.utils import tn
from maxframe.udf import (
    BuiltinFunction,
    BuiltinMethod,
    FsMountOptions,
    MarkedFunction,
    ODPSFunction,
    builtin_function,
    copy_func_scheduling_hints,
    discover_marked_functions,
    with_fs_mount,
    with_image_options,
    with_resources,
    with_running_options,
)

# Module-level @builtin_function targets for attribute-access regressions.


def _bf_inner(x):
    return x


@builtin_function
@functools.wraps(_bf_inner)
def _bf_wraps_outermost(x):
    """``@builtin_function`` *outside* ``@functools.wraps`` so the builtin's
    underlying func carries ``__wrapped__ -> _bf_inner``.  Exercises the
    ``BuiltinFunction.__getattr__`` guard: the prior implementation re-wrapped
    callable ``__wrapped__`` as a ``BuiltinMethod`` and ``inspect.unwrap``
    failed by calling the function with no args."""
    return _bf_inner(x)


@builtin_function
@functools.singledispatch
def _bf_sd_outermost(x):
    """``@builtin_function`` *outside* ``@functools.singledispatch`` so
    ``.register`` reaches ``BuiltinFunction.__getattr__``.  The prior
    implementation returned a ``BuiltinMethod`` that failed on reload.
    """
    return x


def test_odps_function():
    func_body = """from odps.udf import annotate
    @annotate("bigint->bigint")
    class MyMul(object):
        def evaluate(self, arg0):
            return arg0 * 2 if arg0 is not None else None"""
    odps_entry = ODPS.from_environments()
    res_name = tn("test_res")
    func_name = tn("test_odps_func")

    def _cleanup():
        try:
            odps_entry.delete_resource(res_name + ".py")
        except NoSuchObject:
            pass
        try:
            odps_entry.delete_function(func_name)
        except NoSuchObject:
            pass

    _cleanup()

    try:
        test_res = odps_entry.create_resource(
            res_name + ".py", "py", fileobj=textwrap.dedent(func_body)
        )
        test_odps_func_obj = odps_entry.create_function(
            func_name, class_type=f"{res_name}.MyMul", resources=[test_res]
        )
        func = ODPSFunction.wrap(test_odps_func_obj)
        assert isinstance(func, ODPSFunction)
        assert func.__name__ == func_name
        assert func.full_function_name in (
            f"{odps_entry.project}:{func_name}",
            f"{odps_entry.project}:default:{func_name}",
        )
    finally:
        _cleanup()


@pytest.mark.parametrize(
    ("path", "mount_path", "storage_options"),
    [
        pytest.param(
            "oss://oss-cn-shanghai-internal.aliyuncs.com/my-bucket/prefix/",
            "/mnt/oss_data",
            {"role_arn": "acs:ram::123456:role/test-role"},
            id="role-arn",
        ),
        pytest.param(
            "oss://oss-cn-hangzhou.aliyuncs.com/another-bucket/",
            "/mnt/data",
            {"access_key_id": "test_ak", "access_key_secret": "test_sk"},
            id="ak-sk",
        ),
        # Bucket-level path without a prefix component is valid too
        pytest.param(
            "oss://oss-cn-beijing.aliyuncs.com/my-bucket/",
            "/mnt/root",
            {"role_arn": "role"},
            id="without-prefix",
        ),
    ],
)
def test_single_fs_mount(path, mount_path, storage_options):
    @with_fs_mount(path, mount_path, storage_options=storage_options)
    def my_func(df):
        return df

    assert isinstance(my_func, MarkedFunction)
    assert len(my_func.fs_mount) == 1

    config = my_func.fs_mount[0]
    assert isinstance(config, FsMountOptions)
    assert config.path == path
    assert config.mount_path == mount_path
    assert config.storage_options == storage_options


@pytest.mark.parametrize(
    ("mount_specs", "storage_options"),
    [
        pytest.param(
            [
                ("oss://oss-cn-shanghai.aliyuncs.com/bucket1/prefix1/", "/mnt/data1"),
                ("oss://oss-cn-hangzhou.aliyuncs.com/bucket2/prefix2/", "/mnt/data2"),
            ],
            {"role_arn": "same_role"},
            id="same-role-arn",
        ),
        pytest.param(
            [
                ("oss://ep1.aliyuncs.com/b1/p1/", "/mnt/d1"),
                ("oss://ep2.aliyuncs.com/b2/p2/", "/mnt/d2"),
                ("oss://ep3.aliyuncs.com/b3/p3/", "/mnt/d3"),
            ],
            {"access_key_id": "same_ak", "access_key_secret": "same_sk"},
            id="same-ak-sk",
        ),
    ],
)
def test_multiple_fs_mount(mount_specs, storage_options):
    def my_func(df):
        return df

    for path, mount_path in mount_specs:
        my_func = with_fs_mount(path, mount_path, storage_options=storage_options)(
            my_func
        )

    assert isinstance(my_func, MarkedFunction)
    assert len(my_func.fs_mount) == len(mount_specs)

    # Mounts accumulate in application order (with stacked decorators that
    # is inner-most first, i.e. bottom-up)
    assert [c.mount_path for c in my_func.fs_mount] == [
        mount_path for _, mount_path in mount_specs
    ]
    assert all(c.storage_options == storage_options for c in my_func.fs_mount)


def test_fs_mount_preserves_other_marked_function_attrs():
    @with_resources("my_project/resources/my_resource.py")
    @with_fs_mount(
        "oss://ep.aliyuncs.com/bucket/prefix/",
        "/mnt/data",
        storage_options={"role_arn": "role"},
    )
    def my_func(df):
        return df

    assert isinstance(my_func, MarkedFunction)
    assert len(my_func.fs_mount) == 1
    assert len(my_func.file_resources) == 1
    assert "my_resource.py" in my_func.file_resources[0]


def test_fs_mount_options_to_dict_and_repr():
    @with_fs_mount(
        "oss://ep.aliyuncs.com/bucket/prefix/",
        "/mnt/data",
        storage_options={"role_arn": "test_role"},
    )
    def my_func(df):
        return df

    config = my_func.fs_mount[0]

    # to_dict returns raw fields for backend to parse
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["path"] == "oss://ep.aliyuncs.com/bucket/prefix/"
    assert config_dict["mount_path"] == "/mnt/data"
    assert config_dict["storage_options"] == {"role_arn": "test_role"}

    # repr includes the class name and both fields
    repr_str = repr(config)
    assert "FsMountOptions" in repr_str
    assert "oss://ep.aliyuncs.com/bucket/prefix/" in repr_str
    assert "/mnt/data" in repr_str


@pytest.mark.parametrize(
    ("mounts", "match"),
    [
        # (path, mount_path, storage_options) triples applied in sequence
        pytest.param(
            [("", "/mnt/data", {"role_arn": "role"})],  # Empty path
            "A valid path string is required",
            id="empty-path",
        ),
        pytest.param(
            [("oss://ep.aliyuncs.com/bucket/prefix/", "/mnt/data", {})],  # No auth
            "Authentication credentials required",
            id="no-auth",
        ),
        pytest.param(
            [
                ("oss://ep1.aliyuncs.com/b1/p1/", "/mnt/d1", {"role_arn": "role1"}),
                ("oss://ep2.aliyuncs.com/b2/p2/", "/mnt/d2", {"role_arn": "role2"}),
            ],  # Conflicting credentials across stacked mounts
            "must use identical storage_options",
            id="different-storage-options",
        ),
    ],
)
def test_fs_mount_invalid_configuration_raises(mounts, match):
    with pytest.raises(ValueError, match=match):

        def my_func(df):
            return df

        for path, mount_path, storage_options in mounts:
            my_func = with_fs_mount(path, mount_path, storage_options=storage_options)(
                my_func
            )


@pytest.mark.parametrize(
    ("legacy_config", "expected_path", "expected_storage_options"),
    [
        (
            {
                "protocol": "oss",
                "oss_endpoint": "oss-cn-hangzhou.aliyuncs.com",
                "oss_bucket": "test-bucket",
                "oss_bucket_prefix": "data/path/",
                "role_arn": "acs:ram::123456789:role/test-role",
                "auth_mode": "role_arn",
                "mount_path": "/mnt/oss",
            },
            "oss://oss-cn-hangzhou.aliyuncs.com/test-bucket/data/path/",
            {"role_arn": "acs:ram::123456789:role/test-role"},
        ),
        (
            {
                "protocol": "oss",
                "oss_endpoint": "oss-cn-shanghai.aliyuncs.com",
                "oss_bucket": "my-bucket",
                "access_key_id": "test_ak",
                "access_key_secret": "test_sk",
                "auth_mode": "ak_sk",
                "mount_path": "/mnt/data",
            },
            "oss://oss-cn-shanghai.aliyuncs.com/my-bucket",
            {"access_key_id": "test_ak", "access_key_secret": "test_sk"},
        ),
    ],
)
def test_fs_mount_options_from_legacy_dict(
    legacy_config, expected_path, expected_storage_options
):
    result = FsMountOptions.from_legacy_dict(legacy_config)

    assert isinstance(result, FsMountOptions)
    assert result.path == expected_path
    assert result.mount_path == legacy_config["mount_path"]
    assert result.storage_options == expected_storage_options


def test_deserialize_fs_mount():
    from maxframe.core.operator.base import _deserialize_fs_mount

    # Test empty values
    assert _deserialize_fs_mount(None) == []
    assert _deserialize_fs_mount([]) == []
    assert _deserialize_fs_mount({}) == []

    # Test legacy dict format from old clients
    legacy_dict = {
        "oss_endpoint": "oss-cn-hangzhou.aliyuncs.com",
        "oss_bucket": "test-bucket",
        "oss_bucket_prefix": "prefix/",
        "role_arn": "acs:ram::123456789:role/test-role",
        "mount_path": "/mnt/oss",
    }
    result = _deserialize_fs_mount(legacy_dict)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], FsMountOptions)
    assert result[0].path == "oss://oss-cn-hangzhou.aliyuncs.com/test-bucket/prefix/"
    assert result[0].mount_path == "/mnt/oss"
    assert result[0].storage_options == {
        "role_arn": "acs:ram::123456789:role/test-role"
    }

    # Test new format (list of FsMountOptions) passes through unchanged
    options = FsMountOptions(
        path="oss://ep.aliyuncs.com/bucket/",
        mount_path="/mnt/data",
        storage_options={"role_arn": "acs:ram::123456789:role/test-role"},
    )
    input_list = [options]
    result = _deserialize_fs_mount(input_list)
    assert result is input_list


def test_with_image_options():
    @with_image_options("python")
    def my_func(x):
        return x

    assert isinstance(my_func, MarkedFunction)
    assert my_func.image_options == {"name": "python"}

    # Test conflict raises error
    with pytest.raises(ValueError, match="image_options already set to 'image_a'"):

        @with_image_options(image_name="image_b")
        @with_image_options(image_name="image_a")
        def conflict_func(x):
            return x

    @with_running_options(image_name="image_d")
    def my_func_with_running_options(x):
        return x

    assert isinstance(my_func_with_running_options, MarkedFunction)
    assert my_func_with_running_options.image_options == {"name": "image_d"}

    with pytest.raises(ValueError, match="image_options already set to 'image_d'"):

        @with_image_options(image_name="image_b")
        @with_running_options(image_name="image_d")
        def my_func_with_conflicting_running_options(x):
            return x


def test_with_running_options_gu_type_marks_gpu_and_requires_quota():
    with pytest.raises(ValueError, match="gu_type requires gu_quota"):
        with_running_options(gu_type="gu.gu8tf-8x.spot")

    @with_running_options(gu_type="gu.gu8tf-8x.spot", gu_quota="spot")
    def spot_func(x):
        return x

    assert spot_func.expect_resources == {
        "gu_quota": ["spot"],
        "gu_type": "gu.gu8tf-8x.spot",
    }
    assert spot_func.gpu


def test_copy_func_scheduling_hints():
    # Test with a regular function (no scheduling hints)
    with option_context() as options:
        options.function.default_running_options = {}  # No default options

        def regular_func(x):
            return x + 1

        op1 = Operator()
        copy_func_scheduling_hints(regular_func, op1)
        # Should not set any attributes since regular function has no hints
        assert not hasattr(op1, "expect_engine") or op1.expect_engine is None
        assert not hasattr(op1, "expect_resources") or op1.expect_resources is None
        assert not hasattr(op1, "gpu") or op1.gpu is None

    # Test with MarkedFunction with scheduling hints

    @with_running_options(engine="DPE", cpu=4, memory="8GiB")
    def marked_func(x):
        return x + 1

    op2 = Operator()
    copy_func_scheduling_hints(marked_func, op2)
    assert op2.expect_engine == "DPE"
    # The expect_resources will include default values for gpu (gu_quota not set when None)
    expected_resources = {"cpu": 4, "memory": "8GiB", "gpu": 0}
    assert op2.expect_resources == expected_resources

    # Test with MarkedFunction with GPU
    @with_running_options(gu=2)
    def gpu_func(x):
        return x + 1

    op3 = Operator()
    copy_func_scheduling_hints(gpu_func, op3)
    assert op3.gpu is True
    # The expect_resources will include the gu value and default values
    # System has default options: {'cpu': 1, 'memory': '4GiB', 'gpu': 0}
    # When GPU is set, copy_func_scheduling_hints auto-sets gu_quota to
    # [options.session.gu_quota_name] which defaults to [None]
    expected_resources = {"gpu": 2, "gu_quota": [None], "cpu": 1, "memory": "4GiB"}
    assert op3.expect_resources == expected_resources

    # Test with MarkedFunction with image_options
    @with_image_options(image_name="python")
    def image_func(x):
        return x + 1

    op4 = Operator()
    copy_func_scheduling_hints(image_func, op4)
    assert op4.image_options == {"name": "python"}


@pytest.mark.parametrize(
    ("default_resources", "decorator_memory_limit", "expected_resources", "raises"),
    [
        pytest.param(
            {
                "cpu": 1,
                "memory": "4GiB",
                "gpu": 0,
                "memory_limit": "4GiB",
            },
            None,
            {"cpu": 1, "memory": "4GiB", "gpu": 0},
            False,
            id="default-equality",
        ),
        pytest.param(
            {"cpu": 1, "memory": "4GiB", "gpu": 0},
            "2GiB",
            None,
            True,
            id="merged-lower-limit",
        ),
    ],
)
def test_copy_func_scheduling_hints_validates_merged_memory_limit(
    default_resources, decorator_memory_limit, expected_resources, raises
):
    expected_default_resources = default_resources.copy()

    with option_context() as options:
        options.function.default_running_options = default_resources

        def func(x):
            return x

        if decorator_memory_limit is not None:
            func = with_running_options(memory_limit=decorator_memory_limit)(func)
            expected_marked_resources = func.expect_resources.copy()

        op = Operator()
        if raises:
            with pytest.raises(
                ValueError,
                match="memory_limit must be greater than or equal to memory",
            ):
                copy_func_scheduling_hints(func, op)
        else:
            copy_func_scheduling_hints(func, op)
            assert op.expect_resources == expected_resources

        assert options.function.default_running_options == expected_default_resources
        if decorator_memory_limit is not None:
            assert func.expect_resources == expected_marked_resources


def test_copy_func_scheduling_hints_recomputes_marked_resources():
    @with_running_options(memory_limit="4GiB")
    def func(x):
        return x

    expected_marked_resources = func.expect_resources.copy()
    cases = (
        ("4GiB", {"cpu": 1, "memory": "4GiB", "gpu": 0}),
        (
            "2GiB",
            {"cpu": 1, "memory": "2GiB", "gpu": 0, "memory_limit": "4GiB"},
        ),
    )

    for memory, expected_resources in cases:
        default_resources = {
            "cpu": 1,
            "memory": memory,
            "gpu": 0,
        }
        expected_default_resources = default_resources.copy()

        with option_context() as options:
            options.function.default_running_options = default_resources

            op = Operator()
            copy_func_scheduling_hints(func, op)

            assert op.expect_resources == expected_resources
            assert (
                options.function.default_running_options == expected_default_resources
            )

        assert func.expect_resources == expected_marked_resources


# Module-level globals for testing global reference
@with_resources("test_res")
def _test_global_marked():
    return 1


def _test_global_caller():
    return _test_global_marked()


def test_discover_marked_functions():
    """Test discovering MarkedFunction in various scenarios."""

    # Test 1: MarkedFunction itself is returned immediately
    @with_resources("res1")
    def marked_itself():
        return 1

    result = discover_marked_functions(marked_itself)
    assert len(result) == 1
    assert result[0] is marked_itself

    # Test 2: Normal function without MarkedFunctions returns empty list
    def normal_func():
        return 1

    result = discover_marked_functions(normal_func)
    assert len(result) == 0

    # Test 3: MarkedFunction in closure
    @with_resources("res1")
    def inner_func():
        return 1

    def outer_func():
        return inner_func()

    result = discover_marked_functions(outer_func)
    assert len(result) == 1
    assert result[0] is inner_func

    # Test 4: Multiple MarkedFunctions in closure
    @with_resources("res1")
    def func_a():
        return 1

    @with_running_options(cpu=2)
    def func_b():
        return 2

    def multi_func():
        return func_a() + func_b()

    result = discover_marked_functions(multi_func)
    assert len(result) == 2
    assert func_a in result
    assert func_b in result

    # Test 5: Nested closures
    @with_resources("res1")
    def nested_inner():
        return 1

    def middle_func():
        def nested_outer():
            return nested_inner()

        return nested_outer

    result = discover_marked_functions(middle_func())
    assert len(result) == 1
    assert result[0] is nested_inner

    # Test 6: MarkedFunction wrapped in functools.partial
    @with_resources("res1")
    def partial_wrapped(x, y):
        return x + y

    partial_func = functools.partial(partial_wrapped, 10)

    result = discover_marked_functions(partial_func)
    assert len(result) == 1
    assert result[0] is partial_wrapped

    # Test 7: MarkedFunction passed as argument to partial
    @with_resources("res1")
    def marked_arg():
        return 1

    def normal_func_with_arg(x):
        return x()

    partial_with_arg = functools.partial(normal_func_with_arg, marked_arg)

    result = discover_marked_functions(partial_with_arg)
    assert len(result) == 1
    assert result[0] is marked_arg

    # Test 8: MarkedFunction passed as keyword argument to partial
    @with_resources("res1")
    def marked_kwarg():
        return 1

    def normal_func_with_kwarg(x=None):
        return x() if x else 0

    partial_with_kwarg = functools.partial(normal_func_with_kwarg, x=marked_kwarg)

    result = discover_marked_functions(partial_with_kwarg)
    assert len(result) == 1
    assert result[0] is marked_kwarg

    # Test 9: Lambda with closure
    @with_resources("res1")
    def marked_for_lambda():
        return 1

    lambda_func = lambda: marked_for_lambda()  # noqa: E731

    result = discover_marked_functions(lambda_func)
    assert len(result) == 1
    assert result[0] is marked_for_lambda

    # Test 10: Nested function with closure
    @with_resources("res1")
    def marked_for_nested():
        return 1

    def outer_with_inner():
        def inner():
            return marked_for_nested()

        return inner

    result = discover_marked_functions(outer_with_inner())
    assert len(result) == 1
    assert result[0] is marked_for_nested

    # Test 11: No duplicate results
    @with_resources("res1")
    def marked_no_dup():
        return 1

    def no_dup_func():
        a = marked_no_dup
        b = marked_no_dup
        return a() + b()

    result = discover_marked_functions(no_dup_func)
    assert len(result) == 1
    assert result[0] is marked_no_dup

    # Test 12: Class method with MarkedFunction
    @with_resources("res1")
    def marked_for_method():
        return 1

    class MyClass:
        def method(self):
            return marked_for_method()

    obj = MyClass()
    result = discover_marked_functions(obj.method)
    assert len(result) == 1
    assert result[0] is marked_for_method

    # Test 13: Global reference
    result = discover_marked_functions(_test_global_caller)
    assert len(result) == 1
    assert result[0] is _test_global_marked


def test_discover_marked_functions_with_wrapped():
    """Test discovering MarkedFunction with __wrapped__ attribute."""

    # Test 1: Wrapped function
    @with_resources("res1")
    def marked_wrapped():
        return 1

    @functools.wraps(marked_wrapped)
    def wrapper():
        return marked_wrapped()

    # Manually set __wrapped__ to simulate decorator behavior
    wrapper.__wrapped__ = marked_wrapped

    result = discover_marked_functions(wrapper)
    assert len(result) == 1
    assert result[0] is marked_wrapped

    # Test 2: Deeply nested wrapped functions
    @with_resources("res1")
    def deep_inner():
        return 1

    def wrapper1():
        return deep_inner()

    def wrapper2():
        return wrapper1()

    wrapper2.__wrapped__ = wrapper1
    wrapper1.__wrapped__ = deep_inner

    result = discover_marked_functions(wrapper2)
    assert len(result) == 1
    assert result[0] is deep_inner


def test_builtin_function_inspect_unwrap():
    """Regression: ``inspect.unwrap`` on a ``@builtin_function``
    function whose underlying func carries ``__wrapped__`` must
    descend to the original function, not raise by re-wrapping
    callable ``__wrapped__`` as a ``BuiltinMethod``.

    ``@builtin_function`` is the *outermost* decorator over
    ``@functools.wraps(_bf_inner)``, so the builtin's ``self.func`` is a
    plain function with ``__wrapped__ -> _bf_inner``.  The prior
    ``__getattr__`` wrapped every callable attribute as a ``BuiltinMethod``
    and ``BuiltinMethod.func`` re-instantiated the (non-class) function with
    no args, so ``inspect.unwrap`` raised.  Restoring the prior
    implementation must fail this assertion; the current guard (only wrap
    when ``self.func`` is a type) passes it.
    """
    unwrapped = inspect.unwrap(_bf_wraps_outermost)
    assert (
        unwrapped is _bf_inner
    ), f"inspect.unwrap did not reach _bf_inner: {unwrapped!r}"


def test_builtin_function_singledispatch_register():
    """Regression: ``register`` delegates to the wrapped ``singledispatch``.

    ``@builtin_function`` must not shadow the dispatcher's ``register``.
    """
    int_handler = _bf_sd_outermost.register(int)
    assert callable(int_handler)
    assert not isinstance(int_handler, BuiltinMethod)
    _bf_sd_outermost.register(str)(object())
    assert _bf_sd_outermost(0) == 0


@pytest.fixture
def builtin_registry():
    """Isolate the ``BuiltinFunction`` registration registry around a test."""
    registered = BuiltinFunction._registered.copy()
    yield BuiltinFunction._registered
    BuiltinFunction._registered.clear()
    BuiltinFunction._registered.update(registered)


def test_builtin_function_register_unimportable_callable(builtin_registry):
    """A registered callable resolves without being importable by name."""
    func_name = "maxframe.tests._unimportable_module#_private_func"

    def _private_func(x):
        return x * 2

    # Referencing by name does not require the function to be registered.
    bf = BuiltinFunction(func_name=func_name)
    assert bf.func_name == func_name

    BuiltinFunction.register_function_body(func_name, _private_func)
    # No module named ``maxframe.tests._unimportable_module`` exists, so
    # resolution can only succeed through the registration.
    assert bf.func is _private_func
    assert bf(3) == 6


def test_builtin_function_register_takes_precedence_over_import(builtin_registry):
    """The registration is checked before the import mechanism."""
    func_name = _bf_sd_outermost.func_name

    def _override(x):
        return x

    BuiltinFunction.register_function_body(func_name, _override)
    assert BuiltinFunction(func_name=func_name).func is _override

    # Unregistered again -> the importable builtin is reloaded by name.
    del builtin_registry[func_name]
    assert BuiltinFunction(func_name=func_name).func is _bf_sd_outermost.func


def test_builtin_method_registered_class(builtin_registry):
    """``BuiltinMethod`` resolves a class available only via registration."""

    class _RegisteredClass:
        def get(self):
            return 42

    func_name = "maxframe.tests._unimportable_module#_RegisteredClass"
    BuiltinFunction.register_function_body(func_name, _RegisteredClass)
    assert BuiltinMethod(func_name=func_name, method_name="get")() == 42


@pytest.mark.parametrize(
    "probe", ["__wrapped__", "__closure__", "__globals__", "__code__"]
)
def test_builtin_function_body_absent_probes_are_opaque(probe, builtin_registry):
    """Callable inspection must not import a body-absent logical name.

    ``unwrap_function`` and the closure-entity checks probe these dunders while
    the client builds a graph, where the private backend module is not
    importable -- answering them by import would fail the graph construction.
    """
    bf = BuiltinFunction(func_name="myapp#scale")

    assert not hasattr(bf, probe)
    # the body stays genuinely unresolved: explicit access still reports the
    # unimportable module rather than returning a placeholder
    with pytest.raises(ValueError, match="not trusted"):
        bf.func


def test_builtin_function_registered_body_still_answers_probes(builtin_registry):
    """A registered body keeps being inspected (backend/worker processes)."""
    func_name = "maxframe.tests._unimportable_module#_private_func"

    def _private_func(x):
        return x

    bf = BuiltinFunction(func_name=func_name)
    BuiltinFunction.register_function_body(func_name, _private_func)

    assert bf.func is _private_func
    assert hasattr(bf, "__code__")
    assert bf(3) == 3
