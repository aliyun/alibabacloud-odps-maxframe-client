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

from abc import ABCMeta, abstractmethod
from typing import Dict, Iterable, List

from ...core import Tileable
from ...serialization.core import buffered, load_type
from ...serialization.serializables import (
    BoolField,
    DictField,
    ListField,
    Serializable,
    StringField,
)
from ...serialization.serializables.core import SerializableSerializer
from ...utils import extract_class_name, tokenize
from .core import DAG


class EntityGraph(DAG, metaclass=ABCMeta):
    @property
    @abstractmethod
    def results(self):
        """
        Return result tileables or chunks.

        Returns
        -------
        results
        """

    @results.setter
    @abstractmethod
    def results(self, new_results):
        """
        Set result tileables or chunks.

        Parameters
        ----------
        new_results

        Returns
        -------

        """

    def copy(self) -> "EntityGraph":
        graph = super().copy()
        graph.results = self.results.copy()
        return graph


class TileableGraph(EntityGraph, Iterable[Tileable]):
    _result_tileables: List[Tileable]
    # logic key is a unique and deterministic key for `TileableGraph`. For
    # multiple runs the logic key will remain same if the computational logic
    # doesn't change. And it can be used to some optimization when running a
    # same `execute`, like HBO.
    _logic_key: str

    def __init__(self, result_tileables: List[Tileable] = None):
        super().__init__()
        self._result_tileables = result_tileables

    @property
    def result_tileables(self):
        return self._result_tileables

    @property
    def results(self):
        return self._result_tileables

    @results.setter
    def results(self, new_results):
        self._result_tileables = new_results

    @property
    def logic_key(self):
        if not hasattr(self, "_logic_key") or self._logic_key is None:
            token_keys = []
            for node in self.bfs():
                logic_key = node.op.get_logic_key()
                if hasattr(node.op, "logic_key") and node.op.logic_key is None:
                    node.op.logic_key = logic_key
                token_keys.append(
                    tokenize(logic_key, **node.extra_params)
                    if node.extra_params
                    else logic_key
                )
            self._logic_key = tokenize(*token_keys)
        return self._logic_key


class SerializableGraph(Serializable):
    _is_chunk = BoolField("is_chunk")
    # TODO(qinxuye): remove this logic when we handle fetch elegantly,
    # now, the node in the graph and inputs for operator may be inconsistent,
    # for example, an operator's inputs may be chunks,
    # but in the graph, the predecessors are all fetch chunks,
    # we serialize the fetch chunks first to make sure when operator's inputs
    # are serialized, they will just be marked as serialized and skip serialization.
    _fetch_nodes = ListField("fetch_nodes")
    _nodes = DictField("nodes")
    _predecessors = DictField("predecessors")
    _successors = DictField("successors")
    _results = ListField("results")
    _graph_cls = StringField("graph_cls")
    _extra_params = DictField("extra_params", default=None)

    @classmethod
    def from_graph(cls, graph: EntityGraph) -> "SerializableGraph":
        kw = dict(
            _is_chunk=False,
            _fetch_nodes=[chunk for chunk in graph if chunk.is_fetch()],
            _nodes=graph._nodes,
            _predecessors=graph._predecessors,
            _successors=graph._successors,
            _results=graph.results,
            _graph_cls=extract_class_name(type(graph)),
        )
        if hasattr(graph, "extra_params"):
            kw["_extra_params"] = graph.extra_params
        return SerializableGraph(**kw)

    def to_graph(self) -> EntityGraph:
        graph_cls = (
            load_type(self._graph_cls, EntityGraph)
            if hasattr(self, "_graph_cls") and self._graph_cls
            else TileableGraph
        )
        graph = graph_cls(self._results)
        graph._nodes.update(self._nodes)
        graph._predecessors.update(self._predecessors)
        graph._successors.update(self._successors)
        if self._extra_params:
            graph.extra_params = self._extra_params
        return graph


class GraphSerializer(SerializableSerializer):
    @buffered
    def serial(self, obj: EntityGraph, context: Dict):
        serializable_graph = SerializableGraph.from_graph(obj)
        return [], [serializable_graph], False

    def deserial(self, serialized: List, context: Dict, subs: List) -> TileableGraph:
        serializable_graph: EntityGraph = subs[0]
        return serializable_graph.to_graph()


GraphSerializer.register(EntityGraph)
SerializableSerializer.register(SerializableGraph)
