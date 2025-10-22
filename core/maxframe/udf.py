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

import shlex
import sys
from typing import Callable, List, Optional, Union

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
    FunctionField,
    ListField,
    Serializable,
    StringField,
)
from .typing_ import PandasDType
from .utils import extract_class_name, make_dtype, tokenize


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


class BuiltinFunction(Serializable):
    """
    Record reference for builtin functions. The function body
    will NOT be serialized when submitting jobs.
    """

    __slots__ = ("_func",)

    func_name = StringField("func_name")

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
    func = FunctionField("func")
    resources = ListField("resources", FieldTypes.string, default_factory=list)
    pythonpacks = ListField("pythonpacks", FieldTypes.reference, default_factory=list)
    expect_engine = StringField("expect_engine", default=None)
    expect_resources = DictField(
        "expect_resources", FieldTypes.string, default_factory=dict
    )
    gpu = BoolField("gpu", default=False)

    def __init__(self, func: Optional[Callable] = None, **kw):
        super().__init__(func=func, **kw)

    def __getattr__(self, item):
        return getattr(self.func, item)

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)

    def __repr__(self):
        return f"<MarkedFunction {self.func!r}>"


class ODPSFunction(Serializable):
    __slots__ = ("_caller_type",)

    full_function_name = StringField("full_function_name")
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
            existing = getattr(func, "resources") or []
            func.resources = existing + str_resources
            return func

        if isinstance(func, MarkedFunction):
            func.resources = func.resources + str_resources
            return func
        return MarkedFunction(func, resources=str_resources)

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

    resources["gpu"] = gu
    resources["gu_quota"] = gu_quota
    use_gpu = is_positive_integer(gu)

    def func_wrapper(func):
        if all(v is None for v in (engine, cpu, memory, gu, gu_quota)):
            return func
        if isinstance(func, MarkedFunction):
            func.expect_engine = engine
            func.expect_resources = resources
            func.gpu = use_gpu
            return func
        return MarkedFunction(
            func,
            expect_engine=engine,
            expect_resources=resources,
            gpu=use_gpu,
        )

    return func_wrapper


with_resource_libraries = with_resources


def get_udf_resources(func: Callable) -> List[Union[ODPSResourceObj, str]]:
    return getattr(func, "resources", None) or []


def get_udf_pythonpacks(func: Callable) -> List[PythonPackOptions]:
    return getattr(func, "pythonpacks", None) or []
