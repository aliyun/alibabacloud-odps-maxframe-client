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

import sys

from ...serialization.serializables import Serializable


def make_import_error_func(package_name):
    def _func(*_, **__):  # pragma: no cover
        raise ImportError(
            f"Cannot import {package_name}, please reinstall that package."
        )

    return _func


def config_mod_getattr(mod_dict, globals_):
    def __getattr__(name):
        import importlib

        if name in mod_dict:
            mod_name, cls_name = mod_dict[name].rsplit(".", 1)
            mod = importlib.import_module(mod_name, globals_["__name__"])
            cls = globals_[name] = getattr(mod, cls_name)
            return cls
        else:  # pragma: no cover
            raise AttributeError(name)

    if sys.version_info[:2] < (3, 7):
        for _mod in mod_dict.keys():
            __getattr__(_mod)

    def __dir__():
        return sorted([n for n in globals_ if not n.startswith("_")] + list(mod_dict))

    globals_.update(
        {
            "__getattr__": __getattr__,
            "__dir__": __dir__,
            "__all__": list(__dir__()),
            "__warningregistry__": dict(),
        }
    )


class TrainingCallback(Serializable):
    _local_cls = None

    @classmethod
    def _load_local_to_remote_mapping(cls, globals_dict):
        if cls._local_to_remote:
            return
        for v in globals_dict.values():
            if isinstance(v, type) and issubclass(v, cls) and v._local_cls is not None:
                cls._local_to_remote[v._local_cls] = v

    @classmethod
    def from_local(cls, callback_obj):
        if isinstance(callback_obj, (list, tuple)):
            return [cls.from_local(x) for x in callback_obj]
        if not type(callback_obj) in cls._local_to_remote:
            return callback_obj

        kw = {}
        remote_cls = cls._local_to_remote[type(callback_obj)]
        for attr in remote_cls._FIELDS:
            try:
                kw[attr] = getattr(callback_obj, attr)
            except AttributeError:
                pass
        return remote_cls(**kw)

    def has_custom_code(self) -> bool:
        return False

    @classmethod
    def remote_to_local(cls, remote_obj):
        if isinstance(remote_obj, (list, tuple)):
            return [cls.remote_to_local(x) for x in remote_obj]
        if not isinstance(remote_obj, TrainingCallback):
            return remote_obj
        return remote_obj.to_local()

    def _extract_kw(self) -> dict:
        kw = {}
        for attr in type(self)._FIELDS:
            val = getattr(self, attr, None)
            if val is not None:
                kw[attr] = val
        return kw

    def to_local(self):
        return type(self)._local_cls(**self._extract_kw())

    def __call__(self, *args, **kwargs):
        return self.to_local()(*args, **kwargs)
