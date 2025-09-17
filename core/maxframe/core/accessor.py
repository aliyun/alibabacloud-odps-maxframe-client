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

import functools


class BaseMaxFrameAccessor:
    def __init__(self, obj):
        self.obj = obj

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
        if self._name not in obj._accessors:
            obj._accessors[self._name] = self._accessor(obj)
        return obj._accessors[self._name]
