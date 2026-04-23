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
import textwrap

import pytest
from odps import ODPS
from odps.errors import NoSuchObject

from ..tests.utils import tn
from ..udf import (
    FsMountOptions,
    MarkedFunction,
    ODPSFunction,
    discover_marked_functions,
    with_fs_mount,
    with_image_options,
    with_resources,
    with_running_options,
)


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


class TestWithFsMount:
    def test_single_fs_mount_with_role_arn(self):
        @with_fs_mount(
            "oss://oss-cn-shanghai-internal.aliyuncs.com/my-bucket/prefix/",
            "/mnt/oss_data",
            storage_options={"role_arn": "acs:ram::123456:role/test-role"},
        )
        def my_func(df):
            return df

        assert isinstance(my_func, MarkedFunction)
        assert len(my_func.fs_mount) == 1

        config = my_func.fs_mount[0]
        assert isinstance(config, FsMountOptions)
        assert (
            config.path
            == "oss://oss-cn-shanghai-internal.aliyuncs.com/my-bucket/prefix/"
        )
        assert config.mount_path == "/mnt/oss_data"
        assert config.storage_options == {"role_arn": "acs:ram::123456:role/test-role"}

    def test_single_fs_mount_with_ak_sk(self):
        @with_fs_mount(
            "oss://oss-cn-hangzhou.aliyuncs.com/another-bucket/",
            "/mnt/data",
            storage_options={
                "access_key_id": "test_ak",
                "access_key_secret": "test_sk",
            },
        )
        def my_func(df):
            return df

        assert isinstance(my_func, MarkedFunction)
        assert len(my_func.fs_mount) == 1

        config = my_func.fs_mount[0]
        assert isinstance(config, FsMountOptions)
        assert config.path == "oss://oss-cn-hangzhou.aliyuncs.com/another-bucket/"
        assert config.mount_path == "/mnt/data"
        assert config.storage_options["access_key_id"] == "test_ak"
        assert config.storage_options["access_key_secret"] == "test_sk"

    def test_multiple_fs_mount_same_role_arn(self):
        @with_fs_mount(
            "oss://oss-cn-shanghai.aliyuncs.com/bucket1/prefix1/",
            "/mnt/data1",
            storage_options={"role_arn": "same_role"},
        )
        @with_fs_mount(
            "oss://oss-cn-hangzhou.aliyuncs.com/bucket2/prefix2/",
            "/mnt/data2",
            storage_options={"role_arn": "same_role"},
        )
        def my_func(df):
            return df

        assert isinstance(my_func, MarkedFunction)
        assert len(my_func.fs_mount) == 2

        # First decorator applied last (outer), second decorator applied first (inner)
        # So the order in fs_mount list should be: inner first, outer second
        config1 = my_func.fs_mount[0]
        assert config1.mount_path == "/mnt/data2"
        assert "bucket2" in config1.path

        config2 = my_func.fs_mount[1]
        assert config2.mount_path == "/mnt/data1"
        assert "bucket1" in config2.path

    def test_multiple_fs_mount_same_ak_sk(self):
        @with_fs_mount(
            "oss://ep1.aliyuncs.com/b1/p1/",
            "/mnt/d1",
            storage_options={
                "access_key_id": "same_ak",
                "access_key_secret": "same_sk",
            },
        )
        @with_fs_mount(
            "oss://ep2.aliyuncs.com/b2/p2/",
            "/mnt/d2",
            storage_options={
                "access_key_id": "same_ak",
                "access_key_secret": "same_sk",
            },
        )
        @with_fs_mount(
            "oss://ep3.aliyuncs.com/b3/p3/",
            "/mnt/d3",
            storage_options={
                "access_key_id": "same_ak",
                "access_key_secret": "same_sk",
            },
        )
        def my_func(df):
            return df

        assert isinstance(my_func, MarkedFunction)
        assert len(my_func.fs_mount) == 3

        # Check all three configs exist with correct data
        mount_paths = [c.mount_path for c in my_func.fs_mount]
        assert "/mnt/d1" in mount_paths
        assert "/mnt/d2" in mount_paths
        assert "/mnt/d3" in mount_paths

    def test_multiple_fs_mount_different_role_arn_raises(self):
        with pytest.raises(ValueError, match="must use identical storage_options"):

            @with_fs_mount(
                "oss://ep1.aliyuncs.com/b1/p1/",
                "/mnt/d1",
                storage_options={"role_arn": "role1"},
            )
            @with_fs_mount(
                "oss://ep2.aliyuncs.com/b2/p2/",
                "/mnt/d2",
                storage_options={"role_arn": "role2"},
            )
            def my_func(df):
                return df

    def test_fs_mount_without_prefix(self):
        @with_fs_mount(
            "oss://oss-cn-beijing.aliyuncs.com/my-bucket/",
            "/mnt/root",
            storage_options={"role_arn": "role"},
        )
        def my_func(df):
            return df

        config = my_func.fs_mount[0]
        assert config.path == "oss://oss-cn-beijing.aliyuncs.com/my-bucket/"
        assert config.mount_path == "/mnt/root"

    def test_fs_mount_preserves_other_marked_function_attrs(self):
        from maxframe.udf import with_resources

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

    def test_fs_mount_options_to_dict(self):
        @with_fs_mount(
            "oss://ep.aliyuncs.com/bucket/prefix/",
            "/mnt/data",
            storage_options={"role_arn": "test_role"},
        )
        def my_func(df):
            return df

        config = my_func.fs_mount[0]
        config_dict = config.to_dict()

        # to_dict returns raw fields for backend to parse
        assert isinstance(config_dict, dict)
        assert config_dict["path"] == "oss://ep.aliyuncs.com/bucket/prefix/"
        assert config_dict["mount_path"] == "/mnt/data"
        assert config_dict["storage_options"] == {"role_arn": "test_role"}

    def test_fs_mount_options_repr(self):
        @with_fs_mount(
            "oss://ep.aliyuncs.com/bucket/prefix/",
            "/mnt/data",
            storage_options={"role_arn": "role"},
        )
        def my_func(df):
            return df

        config = my_func.fs_mount[0]
        repr_str = repr(config)
        assert "FsMountOptions" in repr_str
        assert "oss://ep.aliyuncs.com/bucket/prefix/" in repr_str
        assert "/mnt/data" in repr_str

    def test_fs_mount_invalid_path_raises(self):
        with pytest.raises(ValueError, match="A valid path string is required"):

            @with_fs_mount(
                "",  # Empty path
                "/mnt/data",
                storage_options={"role_arn": "role"},
            )
            def my_func(df):
                return df

    def test_fs_mount_no_auth_raises(self):
        with pytest.raises(ValueError, match="Authentication credentials required"):

            @with_fs_mount(
                "oss://ep.aliyuncs.com/bucket/prefix/",
                "/mnt/data",
                storage_options={},  # No auth
            )
            def my_func(df):
                return df

    def test_fs_mount_options_from_legacy_dict_with_role_arn(self):
        legacy_config = {
            "protocol": "oss",
            "oss_endpoint": "oss-cn-hangzhou.aliyuncs.com",
            "oss_bucket": "test-bucket",
            "oss_bucket_prefix": "data/path/",
            "role_arn": "acs:ram::123456789:role/test-role",
            "auth_mode": "role_arn",
            "mount_path": "/mnt/oss",
        }

        result = FsMountOptions.from_legacy_dict(legacy_config)

        assert isinstance(result, FsMountOptions)
        assert (
            result.path == "oss://oss-cn-hangzhou.aliyuncs.com/test-bucket/data/path/"
        )
        assert result.mount_path == "/mnt/oss"
        assert result.storage_options == {
            "role_arn": "acs:ram::123456789:role/test-role"
        }

    def test_fs_mount_options_from_legacy_dict_with_ak_sk(self):
        legacy_config = {
            "protocol": "oss",
            "oss_endpoint": "oss-cn-shanghai.aliyuncs.com",
            "oss_bucket": "my-bucket",
            "access_key_id": "test_ak",
            "access_key_secret": "test_sk",
            "auth_mode": "ak_sk",
            "mount_path": "/mnt/data",
        }

        result = FsMountOptions.from_legacy_dict(legacy_config)

        assert isinstance(result, FsMountOptions)
        assert result.path == "oss://oss-cn-shanghai.aliyuncs.com/my-bucket"
        assert result.mount_path == "/mnt/data"
        assert result.storage_options == {
            "access_key_id": "test_ak",
            "access_key_secret": "test_sk",
        }


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
