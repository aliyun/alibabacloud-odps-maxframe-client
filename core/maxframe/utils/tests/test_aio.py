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

import pytest

from ... import utils
from ...lib.wrapped_pickle import is_unpickle_forbidden, switch_unpickle


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
