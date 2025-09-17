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

import asyncio
import contextvars
import functools
import io
import os
import pickle as raw_pickle
import sys
from typing import Callable

import cloudpickle

if sys.version_info[:2] < (3, 8):  # pragma: no cover
    try:
        import pickle5 as pickle_mod  # nosec  # pylint: disable=import_pickle
    except ImportError:
        import pickle as pickle_mod  # nosec  # pylint: disable=import_pickle
else:
    import pickle as pickle_mod  # nosec  # pylint: disable=import_pickle

__all__ = [
    "Pickler",
    "Unpickler",
    "dump",
    "dumps",
    "load",
    "loads",
    "PickleError",
    "PicklingError",
    "UnpicklingError",
    "DEFAULT_PROTOCOL",
    "HIGHEST_PROTOCOL",
    "switch_unpickle",
]

Pickler = cloudpickle.Pickler
dump = cloudpickle.dump
dumps = cloudpickle.dumps

PickleError = pickle_mod.PickleError
PicklingError = pickle_mod.PicklingError
UnpicklingError = pickle_mod.UnpicklingError

DEFAULT_PROTOCOL = pickle_mod.DEFAULT_PROTOCOL
HIGHEST_PROTOCOL = pickle_mod.HIGHEST_PROTOCOL


_default_forbidden_val = bool(int(os.getenv("MAXFRAME_FORBIDDEN_PICKLE", "0")))
_unpickle_forbidden_var = contextvars.ContextVar(
    "unpickle_forbidden", default=_default_forbidden_val
)


def is_unpickle_forbidden():
    return _unpickle_forbidden_var.get(False)


class Unpickler(pickle_mod.Unpickler):
    @functools.wraps(pickle_mod.Unpickler.load)
    def load(self):
        if _unpickle_forbidden_var.get(False):
            raise ValueError("Unpickle is forbidden here")
        return super().load()

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ImportError:
            # workaround for pickle incompatibility since numpy>=2.0
            if not module.startswith("numpy._core"):
                raise
            module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)


@functools.wraps(pickle_mod.load)
def load(file, **kwargs):
    return Unpickler(file, **kwargs).load()


@functools.wraps(pickle_mod.loads)
def loads(s, **kwargs):
    if isinstance(s, str):  # pragma: no cover
        raise TypeError("Can't load pickle from unicode string")
    file = io.BytesIO(s)
    return Unpickler(file, **kwargs).load()


# patch original pickle methods
pickle_mod.Unpickler = Unpickler
pickle_mod.load = load
pickle_mod.loads = loads

raw_pickle.Unpickler = Unpickler
raw_pickle.load = load
raw_pickle.loads = loads

cloudpickle.Unpickler = Unpickler
cloudpickle.load = load
cloudpickle.loads = loads


class _UnpickleSwitch:
    def __init__(self, forbidden: bool = True):
        self._token = None
        self._forbidden = forbidden

    def __enter__(self):
        self._token = _unpickle_forbidden_var.set(self._forbidden)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _unpickle_forbidden_var.reset(self._token)

    def __call__(self, func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapped(*args, **kwargs):
                with _UnpickleSwitch(forbidden=self._forbidden):
                    ret = await func(*args, **kwargs)
                return ret

        else:

            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                with _UnpickleSwitch(forbidden=self._forbidden):
                    return func(*args, **kwargs)

        return wrapped


def switch_unpickle(func=None, *, forbidden: bool = True):
    switch_obj = _UnpickleSwitch(forbidden=forbidden)
    if func is None:
        return switch_obj
    return switch_obj(func)
