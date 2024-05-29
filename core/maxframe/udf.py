# Copyright 1999-2024 Alibaba Group Holding Ltd.
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
from typing import Callable, List, Optional, Union

from odps.models import Resource

from .serialization.serializables import (
    BoolField,
    FieldTypes,
    FunctionField,
    ListField,
    Serializable,
    StringField,
)
from .utils import tokenize


class PythonPackOptions(Serializable):
    key = StringField("key")
    requirements = ListField("requirements", FieldTypes.string, default_factory=list)
    force_rebuild = BoolField("force_rebuild", default=False)
    prefer_binary = BoolField("prefer_binary", default=False)
    pre_release = BoolField("pre_release", default=False)
    pack_instance_id = StringField("pack_instance_id", default=None)

    def __init__(self, key: str = None, **kw):
        super().__init__(key=key, **kw)
        if self.key is None:
            args = {
                "force_rebuild": self.force_rebuild,
                "prefer_binary": self.prefer_binary,
                "pre_release": self.pre_release,
            }
            self.key = tokenize(set(self.requirements), args)

    def __repr__(self):
        return (
            f"<PythonPackOptions {self.requirements} force_rebuild={self.force_rebuild} "
            f"prefer_binary={self.prefer_binary} pre_release={self.pre_release}>"
        )


class MarkedFunction(Serializable):
    func = FunctionField("func")
    resources = ListField("resources", FieldTypes.string, default_factory=list)
    pythonpacks = ListField("pythonpacks", FieldTypes.reference, default_factory=list)

    def __init__(self, func: Optional[Callable] = None, **kw):
        super().__init__(func=func, **kw)

    def __getattr__(self, item):
        return getattr(self.func, item)

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)

    def __repr__(self):
        return f"<MarkedFunction {self.func!r}>"


def with_resources(*resources: Union[str, Resource], use_wrapper_class: bool = True):
    def res_to_str(res: Union[str, Resource]) -> str:
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
        )
        if isinstance(func, MarkedFunction):
            func.pythonpacks.append(pack_item)
            return func
        return MarkedFunction(func, pythonpacks=[pack_item])

    return func_wrapper


with_resource_libraries = with_resources


def get_udf_resources(
    func: Callable,
) -> List[Union[Resource, str]]:
    return getattr(func, "resources", None) or []


def get_udf_pythonpacks(func: Callable) -> List[PythonPackOptions]:
    return getattr(func, "pythonpacks", None) or []
