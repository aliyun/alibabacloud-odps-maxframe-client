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
import shlex
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
from odps.models import Function as ODPSFunctionObj
from odps.models import Resource as ODPSResourceObj

from .config.validators import is_positive_integer
from .core.mode import is_mock_mode
from .serialization import load_member
from .serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    FieldTypes,
    ListField,
    Serializable,
    StringField,
)
from .typing_ import PandasDType
from .utils import extract_class_name, make_dtype, tokenize, unwrap_function


class PythonPackOptions(Serializable):
    _key_args = ("force_rebuild", "prefer_binary", "pre_release", "no_audit_wheel")

    key = StringField("key")
    requirements = ListField("requirements", FieldTypes.string, default_factory=list)
    force_rebuild = BoolField("force_rebuild", default=False)
    prefer_binary = BoolField("prefer_binary", default=False)
    pre_release = BoolField("pre_release", default=False)
    pack_instance_id = StringField("pack_instance_id", default=None)
    no_audit_wheel = BoolField("no_audit_wheel", default=False)

    def __init__(self, key: str = None, **kw):
        super().__init__(key=key, **kw)
        if self.key is None:
            args = {k: getattr(self, k) for k in self._key_args}
            self.key = tokenize(set(self.requirements), args)

    def __repr__(self):
        args_str = " ".join(f"{k}={getattr(self, k)}" for k in self._key_args)
        return f"<PythonPackOptions {self.requirements} {args_str}>"


class FsMountOptions(Serializable):
    path = StringField("path")
    mount_path = StringField("mount_path")
    storage_options = DictField(
        "storage_options", FieldTypes.string, FieldTypes.any, default_factory=dict
    )

    def __repr__(self):
        return f"<FsMountOptions {self.path} -> {self.mount_path}>"

    def validate(self) -> None:
        if not self.path or not isinstance(self.path, str):
            raise ValueError("A valid path string is required.")
        if not self.mount_path or not isinstance(self.mount_path, str):
            raise ValueError("A valid mount_path string is required.")

        # Check authentication: either role_arn or ak/sk must be provided
        storage_opts = self.storage_options or {}
        has_role_arn = bool(storage_opts.get("role_arn"))
        has_ak_sk = bool(
            storage_opts.get("access_key_id") and storage_opts.get("access_key_secret")
        )
        if not has_role_arn and not has_ak_sk:
            raise ValueError(
                "Authentication credentials required in storage_options: "
                "either 'role_arn' or 'access_key_id'/'access_key_secret' must be provided."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "mount_path": self.mount_path,
            "storage_options": self.storage_options,
        }

    @classmethod
    def from_legacy_dict(cls, config: Dict[str, Any]) -> "FsMountOptions":
        # Reconstruct path: oss://{endpoint}/{bucket}/{prefix}
        endpoint = config.get("oss_endpoint", "")
        bucket = config.get("oss_bucket", "")
        prefix = config.get("oss_bucket_prefix", "")
        if endpoint and bucket:
            path = f"oss://{endpoint}/{bucket}"
            if prefix:
                path = f"{path}/{prefix}"
        else:
            path = ""

        # Collect storage_options from legacy fields
        storage_options = {}
        if config.get("role_arn"):
            storage_options["role_arn"] = config["role_arn"]
        if config.get("access_key_id"):
            storage_options["access_key_id"] = config["access_key_id"]
        if config.get("access_key_secret"):
            storage_options["access_key_secret"] = config["access_key_secret"]

        return cls(
            path=path,
            mount_path=config.get("mount_path", ""),
            storage_options=storage_options,
        )


class BuiltinFunction(Serializable):
    """
    Record reference for builtin functions. The function body
    will NOT be serialized when submitting jobs.
    """

    __slots__ = ("_func",)

    func_name = StringField("func_name", default=None)

    def __init__(self, func: Optional[Callable] = None, **kw):
        self._func = func
        if func is not None:
            func_name = extract_class_name(func)
            if "<" in func_name:
                raise ValueError("Cannot be a local or lambda function")
            kw["func_name"] = kw.get("func_name") or func_name
        super().__init__(**kw)

    @property
    def func(self):
        if getattr(self, "_func", None) is None:
            assert isinstance(self.func_name, str)
            self._func = load_member(self.func_name, type(self)).func
        return self._func

    @property
    def module(self):
        # use func_name instead of func itself to avoid it
        # been imported (might cause infinite recursion!)
        assert isinstance(self.func_name, str)
        return self.func_name.split("#")[0]

    def __getattr__(self, item):
        if item.startswith("_") and not item.endswith("_"):
            return super().__getattribute__(item)
        return getattr(self.func, item)

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)


def builtin_function(func: Callable) -> BuiltinFunction:
    return BuiltinFunction(func=func)


class MarkedFunction(Serializable):
    func = AnyField("func", default=None)
    file_resources = ListField("resources", FieldTypes.string, default_factory=list)
    pythonpacks = ListField("pythonpacks", FieldTypes.reference, default_factory=list)
    expect_engine = StringField("expect_engine", default=None)
    expect_resources = DictField(
        "expect_resources", FieldTypes.string, default_factory=dict
    )
    gpu = BoolField("gpu", default=False)
    fs_mount = ListField("fs_mount", FieldTypes.reference, default_factory=list)
    public_network_whitelist = ListField(
        "public_network_whitelist", FieldTypes.string, default_factory=list
    )
    internal_network_whitelist = ListField(
        "internal_network_whitelist", FieldTypes.string, default_factory=list
    )
    vpc_network_link = StringField("vpc_network_link", default=None)
    # image_options configuration: {"name": "image_name"}
    image_options = DictField("image_options", FieldTypes.string, default=None)

    def __init__(self, func: Optional[Callable] = None, **kw):
        super().__init__(func=func, **kw)

    def __getattr__(self, item):
        return getattr(self.func, item)

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)

    def __repr__(self):
        return f"<MarkedFunction {self.func!r}>"

    @property
    def __wrapped__(self):
        return self.func


class ODPSFunction(Serializable):
    __slots__ = ("_caller_type",)

    full_function_name = StringField("full_function_name", default=None)
    expect_engine = StringField("expect_engine", default=None)
    expect_resources = DictField(
        "expect_resources", FieldTypes.string, default_factory=dict
    )
    result_dtype = AnyField("result_dtype", default=None)

    def __init__(
        self,
        func,
        expect_engine: str = None,
        expect_resources: dict = None,
        dtype: PandasDType = None,
        **kw,
    ):
        full_function_name = None
        if isinstance(func, str):
            full_function_name = func
        elif isinstance(func, ODPSFunctionObj):
            func_parts = [func.project.name]
            if func.schema:
                func_parts.append(func.schema.name)
            func_parts.append(func.name)
            full_function_name = ":".join(func_parts)
        if full_function_name:
            kw["full_function_name"] = full_function_name

        if dtype is not None:
            kw["result_dtype"] = make_dtype(dtype)
        super().__init__(
            expect_engine=expect_engine, expect_resources=expect_resources, **kw
        )

    @property
    def __name__(self):
        return self.full_function_name.rsplit(":", 1)[-1]

    def _detect_caller_type(self) -> Optional[str]:
        if hasattr(self, "_caller_type"):
            return self._caller_type

        frame = sys._getframe(1)
        is_set = False
        while frame.f_back:
            f_mod = frame.f_globals.get("__name__")
            if f_mod and f_mod.startswith("maxframe.dataframe."):
                if f_mod.endswith(".map"):
                    self._caller_type, is_set = "map", True
                elif f_mod.endswith(".aggregation") or ".reduction." in f_mod:
                    self._caller_type, is_set = "agg", True
                if is_set:
                    return self._caller_type
            frame = frame.f_back
        return None

    def __call__(self, obj, *args, **kwargs):
        caller_type = self._detect_caller_type()
        if caller_type == "agg":
            return self._call_aggregate(obj, *args, **kwargs)
        raise NotImplementedError("Need to be referenced inside apply or map functions")

    def _call_aggregate(self, obj, *args, **kwargs):
        from .dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
        from .dataframe.reduction.custom_reduction import build_custom_reduction_result

        if isinstance(obj, (DATAFRAME_TYPE, SERIES_TYPE)):
            return build_custom_reduction_result(obj, self)
        if is_mock_mode():
            ret = obj.iloc[0]
            if self.result_dtype:
                if hasattr(ret, "astype"):
                    ret = ret.astype(self.result_dtype)
                else:  # pragma: no cover
                    ret = np.array(ret).astype(self.result_dtype).item()
            return ret
        raise NotImplementedError("Need to be referenced inside apply or map functions")

    def __repr__(self):
        return f"<ODPSStoredFunction {self.full_function_name}>"

    @classmethod
    def wrap(cls, func):
        if isinstance(func, ODPSFunctionObj):
            return ODPSFunction(func)
        return func


def with_resources(
    *resources: Union[str, ODPSResourceObj], use_wrapper_class: bool = True
):
    def res_to_str(res: Union[str, ODPSResourceObj]) -> str:
        if isinstance(res, str):
            return res
        res_parts = [res.project.name]
        if res.schema:
            res_parts.extend(["schemas", res.schema])
        res_parts.extend(["resources", res.name])
        return "/".join(res_parts)

    def func_wrapper(func):
        str_resources = [res_to_str(r) for r in resources]
        if not use_wrapper_class:
            existing = getattr(func, "file_resources") or []
            func.file_resources = existing + str_resources
            return func

        if isinstance(func, MarkedFunction):
            func.file_resources = func.file_resources + str_resources
            return func
        return MarkedFunction(func, file_resources=str_resources)

    return func_wrapper


def with_python_requirements(
    *requirements: str,
    force_rebuild: bool = False,
    prefer_binary: bool = False,
    pre_release: bool = False,
    no_audit_wheel: bool = False,
):
    result_req = []
    for req in requirements:
        result_req.extend(shlex.split(req))

    def func_wrapper(func):
        pack_item = PythonPackOptions(
            requirements=requirements,
            force_rebuild=force_rebuild,
            prefer_binary=prefer_binary,
            pre_release=pre_release,
            no_audit_wheel=no_audit_wheel,
        )
        if isinstance(func, MarkedFunction):
            func.pythonpacks.append(pack_item)
            return func
        return MarkedFunction(func, pythonpacks=[pack_item])

    return func_wrapper


def with_running_options(
    *,
    engine: Optional[str] = None,
    cpu: Optional[int] = None,
    memory: Optional[Union[str, int]] = None,
    gu: Optional[int] = None,
    gu_quota: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    """
    Set running options for the UDF.

    Parameters
    ----------
    engine: Optional[str]
        The engine to run the UDF.
    cpu: Optional[int]
        The CPU to run the UDF.
    memory: Optional[Union[str, int]]
        The memory to run the UDF. If it is an int, it is in GB.
        If it is a str, it is in the format of "10GiB", "30MiB", etc.
    gu: Optional[int]
        The GU number to run the UDF.
    gu_quota: Optional[Union[str, List[str]]]
        The GU quota nicknames to run the UDF. The order is the priority of the usage.
    image_name: Optional[str]
        The registered image name in MaxCompute to use for running the UDF.
    kwargs
        Other running options.
    """
    engine = engine.upper() if engine else None
    resources = kwargs.copy()

    if cpu is not None and isinstance(cpu, int) and cpu <= 0:
        raise ValueError("cpu must be greater than 0")
    if memory is not None:
        if not isinstance(memory, (int, str)):
            raise TypeError("memory must be an int or str")
        if isinstance(memory, int) and memory <= 0:
            raise ValueError("memory must be greater than 0")
    if gu is not None and gu <= 0:
        raise ValueError("gu must be greater than 0")
    if gu is not None and (cpu or memory):
        raise ValueError("gu can't be specified with cpu or memory")

    if cpu:
        resources["cpu"] = cpu
    if memory:
        resources["memory"] = memory

    if isinstance(gu_quota, str):
        gu_quota = [gu_quota]

    # Only set gpu/gu_quota if they have values, avoid setting None
    if gu is not None:
        resources["gpu"] = gu
    if gu_quota is not None:
        resources["gu_quota"] = gu_quota
    use_gpu = is_positive_integer(gu)

    def func_wrapper(func):
        if not all(v is None for v in (engine, cpu, memory, gu, gu_quota)):
            if isinstance(func, MarkedFunction):
                func.expect_engine = engine
                func.expect_resources = resources
                func.gpu = use_gpu
            else:
                func = MarkedFunction(
                    func,
                    expect_engine=engine,
                    expect_resources=resources,
                    gpu=use_gpu,
                )
        # Delegate other settings
        if "image_name" in kwargs:
            func = with_image_options(**kwargs)(func)
        return func

    return func_wrapper


def with_image_options(
    image_name: Optional[str] = None,
    **kwargs,
):
    """
    Set image options for the UDF.
    The image must be registered in MaxCompute beforehand.

    Parameters
    ----------
    image_name: Optional[str]
        The registered image name in MaxCompute to use for running the UDF.
    kwargs
        Other image options.
    """
    if not image_name:
        raise ValueError("image_name is required")

    image_options_config = {"name": image_name, **kwargs}

    def func_wrapper(func):
        if isinstance(func, MarkedFunction):
            existing = func.image_options
            if existing:
                raise ValueError(
                    f"Function image_options already set to '{existing.get('name')}', "
                    f"cannot set to '{image_name}' again"
                )
            func.image_options = image_options_config
            return func
        return MarkedFunction(
            func,
            image_options=image_options_config,
        )

    return func_wrapper


def with_network_options(
    *,
    vpc_network_link: Optional[str] = None,
    public_whitelist: Optional[List[str]] = None,
    internal_whitelist: Optional[List[str]] = None,
):
    """
    Decorator function to add network whitelist options to UDF functions.

    This function is used to specify a list of network addresses that are
    allowed for the UDF function to access.

    Parameters
    ----------
    vpc_network_link: Optional[str]
        When accessing services under Alibaba Cloud VPC network, such as RDS or self-built services, please refer to
        [https://help.aliyun.com/zh/maxcompute/user-guide/access-vpc-solution-direct-connection](The Access VPC Solution)
        to create network link, and set the network link id to this parameter.

    public_whitelist : List[str]
        Variable number of public network address strings, which can be IP addresses, domains, etc.

    internal_whitelist : List[str]
        Variable number of private network address strings, which can be IP addresses, domains, etc.
        It should be noted that these internal network addresses refer to the internal addresses of
        Alibaba Cloud products, which need to be queried in the product manuals.
        For example, OSS internal address oss-cn-hangzhou-internal.aliyuncs.com.

    Returns
    -------
    func_wrapper : Callable
        Returns a decorator function that wraps the target function and
        adds network whitelist configuration.
    """

    def func_wrapper(func):
        if isinstance(func, MarkedFunction):
            pre_public_whitelist_set = (
                set(func.public_network_whitelist)
                if func.public_network_whitelist
                else set()
            )
            curr_public_whitelisted_set = (
                set(public_whitelist) if public_whitelist else set()
            )
            func.public_network_whitelist = list(
                pre_public_whitelist_set | curr_public_whitelisted_set
            )

            pre_internal_whitelist_set = (
                set(func.internal_network_whitelist)
                if func.internal_network_whitelist
                else set()
            )
            curr_internal_whitelist_set = (
                set(internal_whitelist) if internal_whitelist else set()
            )
            func.internal_network_whitelist = list(
                pre_internal_whitelist_set | curr_internal_whitelist_set
            )

            if vpc_network_link:
                if func.vpc_network_link and func.vpc_network_link != vpc_network_link:
                    raise ValueError(
                        "multiple VPC network connections are not supported"
                    )
                func.vpc_network_link = vpc_network_link

            return func

        marked_func = MarkedFunction(func)
        marked_func.public_network_whitelist = public_whitelist
        marked_func.internal_network_whitelist = internal_whitelist
        marked_func.vpc_network_link = vpc_network_link
        return marked_func

    return func_wrapper


StorageOptions = Optional[Dict[str, Any]]


def with_fs_mount(
    path: str,
    mount_path: str,
    storage_options: StorageOptions = None,
):
    mount_options = FsMountOptions(
        path=path,
        mount_path=mount_path,
        storage_options=storage_options or {},
    )
    mount_options.validate()

    def func_wrapper(func):
        if isinstance(func, MarkedFunction):
            # Validate that all mounts use identical storage_options
            if func.fs_mount:
                existing_opts = func.fs_mount[0].storage_options
                new_opts = mount_options.storage_options
                if existing_opts != new_opts:
                    raise ValueError(
                        f"All fs_mount decorators must use identical storage_options. "
                        f"Existing: {existing_opts}, New: {new_opts}"
                    )

            func.fs_mount.append(mount_options)
            return func

        return MarkedFunction(func, fs_mount=[mount_options])

    return func_wrapper


def get_udf_fs_mount(func: Callable) -> List[FsMountOptions]:
    return getattr(func, "fs_mount", None) or []


with_resource_libraries = with_resources


def get_udf_resources(func: Callable) -> List[Union[ODPSResourceObj, str]]:
    return getattr(func, "file_resources", None) or []


def get_udf_pythonpacks(func: Callable) -> List[PythonPackOptions]:
    return getattr(func, "pythonpacks", None) or []


def discover_marked_functions(func: Callable) -> List[MarkedFunction]:
    """
    Discover all MarkedFunction instances referenced in a function.
    """
    discovered: Set[MarkedFunction] = set()
    visited: set = set()

    def _discover(obj: Any) -> None:
        """Recursively discover MarkedFunction instances in an object."""
        # Avoid infinite recursion
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        # Handle functools.partial - check its func, args, and keywords
        if isinstance(obj, functools.partial):
            _discover(obj.func)
            for arg in obj.args:
                _discover(arg)
            for kwarg in obj.keywords.values() if obj.keywords else []:
                _discover(kwarg)
            return

        obj = unwrap_function(
            obj, stop_predicate=lambda x: isinstance(x, MarkedFunction)
        )

        # Check if it's a MarkedFunction
        if isinstance(obj, MarkedFunction):
            if obj not in discovered:
                discovered.add(obj)
            # Also check the inner function of MarkedFunction
            inner_func = obj.func
            if inner_func is not None and inner_func is not obj:
                _discover(inner_func)
            return

        # If it's a callable, inspect its closures and globals
        if callable(obj) and hasattr(obj, "__code__"):
            code = obj.__code__

            # Inspect closure variables
            if hasattr(obj, "__closure__") and obj.__closure__:
                for cell in obj.__closure__:
                    try:
                        cell_contents = cell.cell_contents
                        _discover(cell_contents)
                    except ValueError:
                        # Cell is empty
                        pass

            # Inspect global variables referenced by the function
            if hasattr(obj, "__globals__"):
                for name in code.co_names:
                    if name in obj.__globals__:
                        _discover(obj.__globals__[name])

            # Also check co_consts for nested functions/lambdas
            for const in code.co_consts:
                if callable(const) and const is not obj:
                    _discover(const)

    _discover(func)
    return list(discovered)
