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

import itertools
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from weakref import WeakSet

from maxframe.core import (
    ChunkType,
    TileableGraph,
    TileableType,
    build_fetch,
    enter_mode,
)
from maxframe.core.operator import Fetch
from maxframe.session import AbstractSession
from maxframe.utils import copy_tileables

logger = logging.getLogger(__name__)


@dataclass
class ChunkFetchInfo:
    tileable: TileableType
    chunk: ChunkType
    indexes: List[Union[int, slice]]
    data: Any = None


_submitted_tileables = WeakSet()


@enter_mode(build=True, kernel=True)
def gen_submit_tileable_graph(
    session: AbstractSession,
    result_tileables: List[TileableType],
    tileable_to_copied: Dict[TileableType, TileableType] = None,
    warn_duplicated_execution: bool = False,
) -> Tuple[TileableGraph, List[TileableType]]:
    tileable_to_copied = (
        tileable_to_copied if tileable_to_copied is not None else dict()
    )
    indexer = itertools.count()
    result_to_index = {t: i for t, i in zip(result_tileables, indexer)}
    result = list()
    to_execute_tileables = list()
    graph = TileableGraph(result)

    q = list(result_tileables)
    while q:
        tileable = q.pop()
        if tileable in tileable_to_copied:
            continue
        if tileable.cache and tileable not in result_to_index:
            result_to_index[tileable] = next(indexer)
        outputs = tileable.op.outputs
        inputs = tileable.inputs if session not in tileable._executed_sessions else []
        new_inputs = []
        all_inputs_processed = True
        for inp in inputs:
            if inp in tileable_to_copied:
                new_inputs.append(tileable_to_copied[inp])
            elif session in inp._executed_sessions:
                # executed, gen fetch
                fetch_input = build_fetch(inp).data
                tileable_to_copied[inp] = fetch_input
                graph.add_node(fetch_input)
                new_inputs.append(fetch_input)
            else:
                # some input not processed before
                all_inputs_processed = False
                # put back tileable
                q.append(tileable)
                q.append(inp)
                break
        if all_inputs_processed:
            if isinstance(tileable.op, Fetch):
                new_outputs = [tileable]
            elif session in tileable._executed_sessions:
                new_outputs = []
                for out in outputs:
                    fetch_out = tileable_to_copied.get(out, build_fetch(out).data)
                    new_outputs.append(fetch_out)
            else:
                new_outputs = [
                    t.data for t in copy_tileables(outputs, inputs=new_inputs)
                ]
            for out, new_out in zip(outputs, new_outputs):
                tileable_to_copied[out] = new_out
                graph.add_node(new_out)
                for new_inp in new_inputs:
                    graph.add_edge(new_inp, new_out)

    # process results
    result.extend([None] * len(result_to_index))
    for t, i in result_to_index.items():
        result[i] = tileable_to_copied[t]
        to_execute_tileables.append(t)

    if warn_duplicated_execution:
        for n, c in tileable_to_copied.items():
            if not isinstance(c.op, Fetch) and n in _submitted_tileables:
                warnings.warn(
                    f"Tileable {repr(n)} has been submitted before", RuntimeWarning
                )
        # add all nodes into submitted tileables
        _submitted_tileables.update(
            n for n, c in tileable_to_copied.items() if not isinstance(c.op, Fetch)
        )

    return graph, to_execute_tileables
