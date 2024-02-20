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

from typing import Callable, List, Optional, Union

from odps.models import Resource

from .serialization.serializables import (
    FieldTypes,
    FunctionField,
    ListField,
    Serializable,
)


class MarkedFunction(Serializable):
    func = FunctionField("func")
    resource_libraries = ListField(
        "resource_libraries", FieldTypes.string, default_factory=list
    )

    def __init__(self, func: Optional[Callable] = None, **kw):
        super().__init__(func=func, **kw)

    def __getattr__(self, item):
        return getattr(self.func, item)

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)

    def __repr__(self):
        return f"<MarkedFunction {self.func!r}>"


def with_resource_libraries(
    *resources: Union[str, Resource], use_wrapper_class: bool = True
):
    def res_to_str(res: Union[str, Resource]) -> str:
        if isinstance(res, str):
            return res
        res_parts = [res.project]
        if res.schema:
            res_parts.append(res.schema)
        res_parts.append(res.name)
        return ".".join(res_parts)

    def func_wrapper(func):
        str_resources = [res_to_str(r) for r in resources]
        if not use_wrapper_class:
            func.resource_libraries = str_resources
            return func

        if isinstance(func, MarkedFunction):
            func.resource_libraries = str_resources
            return func
        return MarkedFunction(func, resource_libraries=list(resources))

    return func_wrapper


def get_udf_resource_libraries(
    func: Callable,
) -> Optional[List[Union[Resource, str]]]:
    return getattr(func, "resource_libraries", None)
