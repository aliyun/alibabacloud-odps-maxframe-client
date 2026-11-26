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

"""Shared helpers for LiteFrame reshape compile-time tests.

These tests build operator graphs only and never call ``.execute()``, so the
session-scoped, network-bound ``odps_envs`` autouse fixture inherited from
``core/maxframe/conftest.py`` is replaced with a no-op to keep collection
offline and fast.
"""

import pytest

from maxframe.liteframe.operators.project import LiteFrameProjection


@pytest.fixture(scope="session", autouse=True)
def odps_envs():
    yield


def _walk_nodes(lf):
    """Yield every tileable-data node in ``lf``'s operator graph (DFS)."""
    seen = set()
    stack = [lf.data]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        for inp in getattr(node.op, "inputs", None) or []:
            stack.append(inp)


def _op_types(lf):
    """Return the set of operator class names in ``lf``'s graph."""
    return {type(node.op).__name__ for node in _walk_nodes(lf)}


def _count_expr_type(lf, expr_type):
    """Count expression nodes of *expr_type* across every projection.

    Walks each ``LiteFrameProjection`` node's projections and their
    sub-expressions (operand/expr/multi_expr/...), cycle-safe.
    """
    seen = set()
    total = 0
    for node in _walk_nodes(lf):
        op = node.op
        if not isinstance(op, LiteFrameProjection):
            continue
        for proj in op.projections:
            stack = [proj]
            while stack:
                e = stack.pop()
                if id(e) in seen:
                    continue
                seen.add(id(e))
                if isinstance(e, expr_type):
                    total += 1
                for attr in (
                    "operand",
                    "left",
                    "right",
                    "condition",
                    "true_value",
                    "false_value",
                    "expr",
                    "multi_expr",
                ):
                    v = getattr(e, attr, None)
                    if v is not None and hasattr(v, "_op_type_"):
                        stack.append(v)
    return total
