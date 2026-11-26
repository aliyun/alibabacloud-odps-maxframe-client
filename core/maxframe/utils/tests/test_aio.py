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
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mock
import pytest
from tornado.httpclient import HTTPClientError
from tornado.simple_httpclient import HTTPTimeoutError

from maxframe import utils
from maxframe.config import options
from maxframe.lib.wrapped_pickle import is_unpickle_forbidden, switch_unpickle
from maxframe.utils import aio as aio_module


@pytest.fixture
def http_mock(monkeypatch):
    state = SimpleNamespace(now=0.0, delays=[], fetch=mock.AsyncMock())

    async def sleep(delay):
        state.delays.append(delay)
        state.now += delay

    monkeypatch.setattr(aio_module, "time", SimpleNamespace(time=lambda: state.now))
    monkeypatch.setattr(
        aio_module,
        "asyncio",
        SimpleNamespace(sleep=sleep, iscoroutinefunction=asyncio.iscoroutinefunction),
    )
    monkeypatch.setattr(
        aio_module.httpclient,
        "AsyncHTTPClient",
        lambda: SimpleNamespace(fetch=state.fetch),
    )
    return state


async def test_http_errors_share_original_timeout_budget(http_mock):
    last_error = HTTPClientError(502)

    async def request(*args, **kwargs):
        if http_mock.now < 50:
            http_mock.now += kwargs["request_timeout"]
            raise HTTPTimeoutError("request timed out")
        http_mock.now += 5  # The HTTP error arrives 55s after the helper was entered.
        raise last_error

    http_mock.fetch.side_effect = request
    # The outer retry catches HTTPClientError, not the helper's TimeoutError.
    with pytest.raises(TimeoutError) as exc:
        await utils.call_with_retry(
            utils.wait_http_response,
            "http://service/request",
            request_timeout=60,
            retry_status_codes=(404, 502),
            max_status_retries=0,
            retry_times=1,
            exc_type=HTTPClientError,
        )
    assert http_mock.fetch.await_count == 6
    assert http_mock.now == 60
    assert http_mock.delays == [5]
    assert exc.value.__cause__ is last_error
    assert "url=http://service/request" in str(exc.value)
    assert "last_status=502" in str(exc.value)
    assert "status_retries=1" in str(exc.value)


async def test_wait_without_budget_retries_same_request(http_mock):
    response = object()
    results = iter(
        [
            (0, HTTPClientError(502)),
            (20, HTTPTimeoutError("request timed out")),
            (20, HTTPTimeoutError("request timed out")),
            (20, HTTPClientError(404)),
            (0, response),
        ]
    )

    async def request(*args, **kwargs):
        elapsed, result = next(results)
        http_mock.now += elapsed
        if isinstance(result, Exception):
            raise result
        return result

    http_mock.fetch.side_effect = request
    result = await utils.wait_http_response(
        "http://service/request",
        method="POST",
        body=b"payload",
        headers={"Host": "original-host"},
        retry_status_codes=(404, 502),
        max_status_retries=2,
    )
    assert result is response
    assert http_mock.now == 70
    assert http_mock.delays == [5, 5]
    expected_call = mock.call(
        "http://service/request",
        request_timeout=None,
        method="POST",
        body=b"payload",
        headers={"Host": "original-host"},
    )
    assert http_mock.fetch.await_args_list == [expected_call] * 5


async def test_wait_without_budget_limits_status_retries(http_mock):
    last_error = HTTPClientError(502)
    http_mock.fetch.side_effect = [HTTPClientError(404), HTTPClientError(502)] * 6 + [
        last_error
    ]

    with pytest.raises(HTTPClientError) as exc:
        await utils.wait_http_response(
            "http://service/request", retry_status_codes=(404, 502)
        )

    assert exc.value is last_error
    assert http_mock.fetch.await_count == 13
    assert http_mock.delays == [5] * 12


async def test_other_http_errors_use_outer_retry(http_mock):
    response = object()
    http_mock.fetch.side_effect = [HTTPClientError(502), HTTPClientError(500), response]

    result = await utils.call_with_retry(
        utils.wait_http_response,
        "http://service/request",
        request_timeout=60,
        retry_status_codes=(404, 502),
        retry_times=1,
        exc_type=HTTPClientError,
    )
    assert result is response
    assert http_mock.fetch.await_count == 3
    assert http_mock.delays == [5, options.retry_delay]


async def test_http_errors_are_not_retried_by_default(http_mock):
    http_mock.fetch.side_effect = HTTPClientError(404)
    with pytest.raises(HTTPClientError) as exc:
        await utils.wait_http_response("http://service/request", request_timeout=60)
    assert exc.value.code == 404
    assert http_mock.fetch.await_count == 1
    assert not http_mock.delays


@pytest.mark.parametrize("use_async", [False, True])
async def test_call_with_retry(use_async):
    retry_idx_list = [0]

    def sync_func(delay=0):
        if delay:
            time.sleep(delay)
        if retry_idx_list[0] < 3:
            retry_idx_list[0] += 1
            raise ValueError

    async def async_func(delay=0):
        if delay:
            await asyncio.sleep(delay)
        if retry_idx_list[0] < 3:
            retry_idx_list[0] += 1
            raise ValueError

    func = async_func if use_async else sync_func

    async def wait_coro(res):
        if asyncio.iscoroutine(res):
            return await res
        return res

    # test cases for retry times
    with pytest.raises(ValueError):
        retry_idx_list[0] = 0
        await wait_coro(
            utils.call_with_retry(func, retry_times=1, exc_type=(TypeError, ValueError))
        )
    assert retry_idx_list[0] == 2

    retry_idx_list[0] = 0
    await wait_coro(
        utils.call_with_retry(func, retry_times=3, exc_type=(TypeError, ValueError))
    )
    assert retry_idx_list[0] == 3

    retry_idx_list[0] = 0
    exc_info = await wait_coro(
        utils.call_with_retry(
            func, retry_times=1, exc_type=(TypeError, ValueError), no_raise=True
        )
    )
    assert isinstance(exc_info[1], ValueError)
    assert retry_idx_list[0] == 2

    delay_func = functools.partial(func, delay=0.5)
    with pytest.raises(ValueError):
        retry_idx_list[0] = 0
        await wait_coro(
            utils.call_with_retry(delay_func, retry_times=None, retry_timeout=0.7)
        )
    assert retry_idx_list[0] == 2

    retry_idx_list[0] = 0
    await wait_coro(
        utils.call_with_retry(delay_func, retry_times=None, retry_timeout=2.2)
    )
    assert retry_idx_list[0] == 3

    retry_idx_list[0] = 0
    exc_info = await wait_coro(
        utils.call_with_retry(
            delay_func, retry_times=None, retry_timeout=0.7, no_raise=True
        )
    )
    assert isinstance(exc_info[1], ValueError)
    assert retry_idx_list[0] == 2


def test_debug_to_thread():
    class MixinTestCls(utils.ToThreadMixin):
        @switch_unpickle
        async def run_in_coro(self):
            await self.to_thread(time.sleep, 0.5)
            await self.to_thread(functools.partial(time.sleep), 0.5)
            await self.to_thread(check_unpickle_forbidden)

    def check_unpickle_forbidden():
        assert is_unpickle_forbidden()
        with switch_unpickle():
            pass
        assert is_unpickle_forbidden()

    def thread_body():
        loop = asyncio.new_event_loop()
        loop.set_debug(True)
        asyncio.set_event_loop(loop)
        loop.run_until_complete(MixinTestCls().run_in_coro())

    tpe = ThreadPoolExecutor(max_workers=1)
    tpe.submit(thread_body).result()
    tpe.shutdown()


def test_create_sync_primitive():
    import sys

    loop = asyncio.new_event_loop()

    # Test with Python < 3.10 style (if applicable)
    if sys.version_info[1] < 10:
        lock = utils.create_sync_primitive(asyncio.Lock, loop)
        assert isinstance(lock, asyncio.Lock)
    else:
        # Test with Python >= 3.10 style
        lock = utils.create_sync_primitive(asyncio.Lock, loop)
        assert isinstance(lock, asyncio.Lock)

        # Test that the lock works correctly
        async def test_lock():
            await lock.acquire()
            try:
                # Lock should be held
                assert lock.locked()
            finally:
                lock.release()
            assert not lock.locked()

        loop.run_until_complete(test_lock())

    loop.close()


def test_to_thread_cancelled_error():
    error = utils.ToThreadCancelledError("Test error", result="test_result")

    assert str(error) == "Test error"
    assert error.result == "test_result"

    # Test inheritance
    assert isinstance(error, asyncio.CancelledError)


def test_relay_future():
    async def test_relay():
        # Test with asyncio futures
        src_future = asyncio.Future()
        dest_future = asyncio.Future()

        utils.relay_future(dest_future, src_future)

        # Set result on source
        src_future.set_result("test_result")

        # Result should be relayed to destination
        result = await dest_future
        assert result == "test_result"

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(test_relay())
    finally:
        loop.close()


def test_to_thread_mixin():
    class TestMixin(utils.ToThreadMixin):
        def sync_add(self, a, b):
            return a + b

        async def async_add_via_thread(self, a, b):
            return await self.to_thread(self.sync_add, a, b)

    async def test_mixin():
        mixin = TestMixin()

        # Test basic thread execution
        result = await mixin.async_add_via_thread(2, 3)
        assert result == 5

        # Test ensure_async_call with sync function
        result = await mixin.ensure_async_call(mixin.sync_add, 5, 7)
        assert result == 12

        # Test ensure_async_call with async function
        async def async_add(a, b):
            return a + b

        result = await mixin.ensure_async_call(async_add, 10, 20)
        assert result == 30

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(test_mixin())
    finally:
        loop.close()
