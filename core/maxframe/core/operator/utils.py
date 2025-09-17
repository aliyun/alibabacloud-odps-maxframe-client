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

import contextlib
import sys

from ...typing_ import EntityType, TileableType
from ..entity import TILEABLE_TYPE


def build_fetch_tileable(tileable: TileableType) -> TileableType:
    if tileable.is_coarse():
        chunks = None
    else:
        chunks = []
        for c in tileable.chunks:
            fetch_chunk = build_fetch(c, index=c.index)
            chunks.append(fetch_chunk)

    tileable_op = tileable.op
    params = tileable.params.copy()

    new_op = tileable_op.get_fetch_op_cls(tileable)(_id=tileable_op.id)
    return new_op.new_tileables(
        None,
        chunks=chunks,
        nsplits=tileable.nsplits,
        _key=tileable.key,
        _id=tileable.id,
        **params,
    )[0]


_type_to_builder = [
    (TILEABLE_TYPE, build_fetch_tileable),
]


def build_fetch(entity: EntityType, **kw) -> EntityType:
    for entity_types, func in _type_to_builder:
        if isinstance(entity, entity_types):
            return func(entity, **kw)
    raise TypeError(f"Type {type(entity)} not supported")


def add_fetch_builder(entity_type, builder_func):
    _type_to_builder.append((entity_type, builder_func))


@contextlib.contextmanager
def rewrite_stop_iteration():
    try:
        yield
    except StopIteration:
        raise RuntimeError("Unexpected StopIteration happened.").with_traceback(
            sys.exc_info()[2]
        ) from None
