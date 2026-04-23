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

import asyncio
import asyncio.events
import concurrent.futures
import contextvars
import functools
import itertools
import sys
import time
from typing import Awaitable, Callable, Optional, Tuple, Type, TypeVar, Union

from tornado import httpclient
from tornado.simple_httpclient import HTTPTimeoutError

from ..typing_ import TimeoutType
from .functional import unwrap_function

_PrimitiveType = TypeVar("_PrimitiveType")
_ToThreadRetType = TypeVar("_ToThreadRetType")
_RetryRetType = TypeVar("_RetryRetType")


async def wait_http_response(
    url: str, *, request_timeout: TimeoutType = None, **kwargs
) -> httpclient.HTTPResponse:
    start_time = time.time()
    while request_timeout is None or time.time() - start_time < request_timeout:
        timeout_left = min(10.0, time.time() - start_time) if request_timeout else None
        try:
            return await httpclient.AsyncHTTPClient().fetch(
                url, request_timeout=timeout_left, **kwargs
            )
        except HTTPTimeoutError:
            pass
    raise TimeoutError


def create_sync_primitive(
    cls: Type[_PrimitiveType], loop: asyncio.AbstractEventLoop
) -> _PrimitiveType:
    """
    Create an asyncio sync primitive (locks, events, etc.)
    in a certain event loop.
    """
    if sys.version_info[1] < 10:
        return cls(loop=loop)

    # From Python3.10 the loop parameter has been removed. We should work around here.
    try:
        old_loop = asyncio.get_event_loop()
    except RuntimeError:
        old_loop = None
    try:
        asyncio.set_event_loop(loop)
        primitive = cls()
    finally:
        asyncio.set_event_loop(old_loop)
    return primitive


class ToThreadCancelledError(asyncio.CancelledError):
    def __init__(self, *args, result=None):
        super().__init__(*args)
        self._result = result

    @property
    def result(self):
        return self._result


class ToThreadMixin:
    _thread_pool_size = 1
    _counter = itertools.count().__next__

    def __del__(self):
        if hasattr(self, "_pool"):
            kw = {"wait": False}
            if sys.version_info[:2] >= (3, 9):
                kw["cancel_futures"] = True
            self._pool.shutdown(**kw)

    async def to_thread(
        self,
        func: Callable[..., _ToThreadRetType],
        *args,
        wait_on_cancel: bool = False,
        timeout: float = None,
        debug_task_name: Optional[str] = None,
        **kwargs,
    ) -> _ToThreadRetType:
        if not hasattr(self, "_pool"):
            self._pool = concurrent.futures.ThreadPoolExecutor(
                self._thread_pool_size,
                thread_name_prefix=f"{type(self).__name__}Pool-{self._counter()}",
            )

        loop = asyncio.events.get_running_loop()
        ctx = contextvars.copy_context()
        func_call = functools.partial(ctx.run, func, *args, **kwargs)
        fut = loop.run_in_executor(self._pool, func_call)

        if loop.get_debug():
            # create a task and mark its name
            default_task_name = None
            try:
                unwrapped = unwrap_function(func)
                default_task_name = unwrapped.__qualname__
                if getattr(unwrapped, "__module__", None):
                    default_task_name = unwrapped.__module__ + "#" + default_task_name
            except:  # noqa # pragma: no cover
                try:
                    default_task_name = repr(func)
                except:  # noqa
                    pass
            debug_task_name = debug_task_name or default_task_name

            async def _wait_fut(aio_fut):
                return await aio_fut

            fut = asyncio.create_task(_wait_fut(fut))
            if sys.version_info[:2] == (3, 7):
                # In Python3.7 we should hack the task name to print it in debug logs.
                setattr(fut, "fd_task_name", debug_task_name)
            else:
                fut.set_name(debug_task_name)

        try:
            coro = fut
            if wait_on_cancel:
                coro = asyncio.shield(coro)
            if timeout is not None:
                coro = asyncio.wait_for(coro, timeout)
            return await coro
        except (asyncio.CancelledError, asyncio.TimeoutError) as ex:
            if not wait_on_cancel:
                raise
            result = await fut
            raise ToThreadCancelledError(*ex.args, result=result)

    def ensure_async_call(
        self,
        func: Callable[..., _ToThreadRetType],
        *args,
        wait_on_cancel: bool = False,
        **kwargs,
    ) -> Awaitable[_ToThreadRetType]:
        if asyncio.iscoroutinefunction(func):
            return func(*args, **kwargs)
        return self.to_thread(func, *args, wait_on_cancel=wait_on_cancel, **kwargs)


def relay_future(
    dest: Union[asyncio.Future, concurrent.futures.Future],
    src: Union[asyncio.Future, concurrent.futures.Future],
) -> None:
    def cb(fut: Union[asyncio.Future, concurrent.futures.Future]):
        try:
            dest.set_result(fut.result())
        except BaseException as ex:
            dest.set_exception(ex)

    src.add_done_callback(cb)


def call_with_retry(
    func: Callable[..., _RetryRetType],
    *args,
    retry_times: Optional[int] = None,
    retry_timeout: TimeoutType = None,
    delay: TimeoutType = None,
    reset_func: Optional[Callable] = None,
    exc_type: Union[
        Type[BaseException], Tuple[Type[BaseException], ...]
    ] = BaseException,
    allow_interrupt: bool = True,
    no_raise: bool = False,
    is_func_async: Optional[bool] = None,
    **kwargs,
) -> _RetryRetType:
    """
    Retry calling function given specified times or timeout.

    Parameters
    ----------
    func: Callable
        function to be retried
    args
        arguments to be passed to the function
    retry_times: Optional[int]
        times to retry the function
    retry_timeout: TimeoutType
        timeout in seconds to retry the function
    delay: TimeoutType
        delay in seconds between every trial
    reset_func: Callable
        Function to call after every trial
    exc_type: Type[BaseException] | Tuple[Type[BaseException], ...]
        Exception type for retrial
    allow_interrupt: bool
        If True, KeyboardInterrupt will stop the retry
    no_raise: bool
        If True, no exception will be raised even if all trials failed
    is_func_async: bool
        If True, func will be treated as async
    kwargs
        keyword arguments to be passed to the function

    Returns
    -------
    Return value of the original function
    """
    from ..config import options

    retry_num = 0
    retry_times = retry_times if retry_times is not None else options.retry_times
    delay = delay if delay is not None else options.retry_delay
    start_time = time.monotonic() if retry_timeout is not None else None

    def raise_or_continue(exc: BaseException):
        nonlocal retry_num
        retry_num += 1
        if allow_interrupt and isinstance(exc, KeyboardInterrupt):
            raise exc from None
        if (retry_times is not None and retry_num > retry_times) or (
            retry_timeout is not None
            and start_time is not None
            and time.monotonic() - start_time > retry_timeout
        ):
            if no_raise:
                return sys.exc_info()
            raise exc from None

    async def async_retry():
        while True:
            try:
                return await func(*args, **kwargs)
            except exc_type as ex:
                await asyncio.sleep(delay)
                res = raise_or_continue(ex)
                if res is not None:
                    return res

                if callable(reset_func):
                    reset_res = reset_func()
                    if asyncio.iscoroutine(reset_res):
                        await reset_res

    def sync_retry():
        while True:
            try:
                return func(*args, **kwargs)
            except exc_type as ex:
                time.sleep(delay)
                res = raise_or_continue(ex)
                if res is not None:
                    return res
                if callable(reset_func):
                    reset_func()

    unwrap_func = func
    if is_func_async is None:
        # unwrap to get true result if func is async
        while isinstance(unwrap_func, functools.partial):
            unwrap_func = unwrap_func.func

    if is_func_async or asyncio.iscoroutinefunction(unwrap_func):
        return async_retry()
    else:
        return sync_retry()
