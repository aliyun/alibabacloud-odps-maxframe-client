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
import os
import sys
import threading

from ..config import options

_internal_mode = threading.local()

_is_in_debug = "VSCODE_PID" in os.environ or "PYCHARM_HOSTED" in os.environ
_debug_thread_mode = type("_DebugThreadMode", (object,), {})
_is_daemon_thread_cache = dict()


def _is_in_ide_repr_thread() -> bool:
    """
    For recent pycharm, repr() is called in a separate thread, thus
    we need to configure mode flags in a separate global object
    """
    if not _is_in_debug:
        return False
    thread_ident = threading.current_thread().ident
    if thread_ident in _is_daemon_thread_cache:
        return _is_daemon_thread_cache[thread_ident]
    cur_frame = sys._getframe(1)
    while cur_frame.f_back is not None:
        if "pydevd_repr_utils" in cur_frame.f_code.co_filename:
            # only cache negative result as daemon thread
            #  seems spawned frequently
            return True
        cur_frame = cur_frame.f_back
    _is_daemon_thread_cache[thread_ident] = False
    return False


def is_eager_mode():
    in_kernel = is_kernel_mode()
    if not in_kernel:
        return options.execution_mode == "eager"
    else:
        # in kernel, eager mode always False
        return False


def _get_mode_value(mode_name):
    val = bool(getattr(_internal_mode, mode_name, False))
    if val:
        return True
    elif _is_in_ide_repr_thread():
        return bool(getattr(_debug_thread_mode, mode_name, False))
    return False


def is_kernel_mode():
    return _get_mode_value("kernel")


def is_build_mode():
    return _get_mode_value("build")


def is_mock_mode():
    return _get_mode_value("mock")


class _EnterModeFuncWrapper:
    def __init__(self, mode_name_to_value):
        self.mode_name_to_value = mode_name_to_value

        # as the wrapper may enter for many times
        # record old values for each time
        self.mode_name_to_value_list = list()

    def __enter__(self):
        mode_name_to_old_value = dict()
        for mode_name, value in self.mode_name_to_value.items():
            # record mode's old values
            mode_name_to_old_value[mode_name] = getattr(_internal_mode, mode_name, None)
            if value is None:
                continue
            # set value
            setattr(_internal_mode, mode_name, value)
            if _is_in_debug:
                setattr(_debug_thread_mode, mode_name, value)
        self.mode_name_to_value_list.append(mode_name_to_old_value)

    def __exit__(self, *_):
        mode_name_to_old_value = self.mode_name_to_value_list.pop()
        for mode_name in self.mode_name_to_value.keys():
            # set back old values
            value = mode_name_to_old_value[mode_name]
            setattr(_internal_mode, mode_name, value)
            if _is_in_debug:
                setattr(_debug_thread_mode, mode_name, value)

    def __call__(self, func):
        mode_name_to_value = self.mode_name_to_value.copy()
        if not inspect.iscoroutinefunction(func):
            # sync
            @functools.wraps(func)
            def _inner(*args, **kwargs):
                with enter_mode(**mode_name_to_value):
                    return func(*args, **kwargs)

        else:
            # async
            @functools.wraps(func)
            async def _inner(*args, **kwargs):
                with enter_mode(**mode_name_to_value):
                    return await func(*args, **kwargs)

        return _inner


def enter_mode(kernel=None, build=None, mock=None):
    mode_name_to_value = {
        "kernel": kernel,
        "build": build,
        "mock": mock,
    }
    mode_name_to_value = {k: v for k, v in mode_name_to_value.items() if v is not None}

    return _EnterModeFuncWrapper(mode_name_to_value)
