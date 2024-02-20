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

import asyncio
import functools
import os
import queue
import socket
import types
from threading import Thread
from typing import Dict, List, Optional, Set, Tuple

import pytest
from tornado import netutil

from ..core import Tileable, TileableGraph
from ..utils import lazy_import

try:
    from flaky import flaky
except ImportError:

    def flaky(func=None, **_):
        if func is not None:
            return func

        def decorator(f):
            return f

        return decorator


cupy = lazy_import("cupy")
cudf = lazy_import("cudf")
ray = lazy_import("ray")

_test_tables_to_drop = set()


def bind_unused_port(
    reuse_port: bool = False, address: str = "127.0.0.1"
) -> Tuple[socket.socket, int]:
    sock = netutil.bind_sockets(
        0, address, family=socket.AF_INET, reuse_port=reuse_port
    )[0]
    port = sock.getsockname()[1]
    return sock, port


def tn(s, limit=128):
    if os.environ.get("TEST_NAME_SUFFIX") is not None:
        suffix = "_" + os.environ.get("TEST_NAME_SUFFIX").lower()
        if len(s) + len(suffix) > limit:
            s = s[: limit - len(suffix)]
        table_name = s + suffix
        if table_name.count(".") <= 2:
            _test_tables_to_drop.add(table_name)
        return table_name
    else:
        if len(s) > limit:
            s = s[:limit]
        return s


def run_app_in_thread(app_func):
    def app_thread_func(
        loop: asyncio.AbstractEventLoop,
        q: queue.Queue,
        exit_event: asyncio.Event,
        args,
        kwargs,
    ):
        asyncio.set_event_loop(loop)

        async def app_main():
            iterator = app_func(*args, **kwargs)
            is_gen = isinstance(iterator, types.GeneratorType)
            try:
                q.put(next(iterator) if is_gen else iterator)
                await exit_event.wait()
            finally:
                try:
                    if is_gen:
                        list(iterator)
                except StopIteration:
                    pass

        loop.run_until_complete(app_main())

    @functools.wraps(app_func)
    def fixture_func(*args, **kwargs):
        app_loop = asyncio.new_event_loop()
        q = queue.Queue()
        exit_event = asyncio.Event(loop=app_loop)
        app_thread = Thread(
            target=app_thread_func,
            args=(app_loop, q, exit_event, args, kwargs),
        )
        app_thread.start()

        try:
            yield q.get()
        finally:

            async def set_exit_event():
                exit_event.set()

            asyncio.run_coroutine_threadsafe(set_exit_event(), app_loop)
            app_thread.join()

    return fixture_func


def assert_graph_equal(
    graph: TileableGraph,
    expected_node_count: int,
    expected_nodes: Set[str],
    expected_edges: Dict[str, List[str]],
    expected_results: Optional[List[Tileable]],
):
    assert len(graph) == expected_node_count
    key_to_node = {node.key: node for node in graph.iter_nodes()}
    assert set(graph) == {key_to_node[key] for key in expected_nodes}
    for pred, successors in expected_edges.items():
        pred_node = key_to_node[pred]
        assert graph.count_successors(pred_node) == len(successors)
        for successor in successors:
            assert graph.has_successor(pred_node, key_to_node[successor])
    assert graph.results == expected_results


def require_cupy(func):
    if pytest:
        func = pytest.mark.cuda(func)
    func = pytest.mark.skipif(cupy is None, reason="cupy not installed")(func)
    return func


def require_cudf(func):
    if pytest:
        func = pytest.mark.cuda(func)
    func = pytest.mark.skipif(cudf is None, reason="cudf not installed")(func)
    return func


def require_hadoop(func):
    if pytest:
        func = pytest.mark.hadoop(func)
    func = pytest.mark.skipif(
        not os.environ.get("WITH_HADOOP"), reason="Only run when hadoop is installed"
    )(func)
    return func
