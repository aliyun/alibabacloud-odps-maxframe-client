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

import math
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd

from .utils import rewrite_stop_iteration

if TYPE_CHECKING:
    from .. import TileableGraph

try:
    from numpy.core._exceptions import UFuncTypeError
except ImportError:  # pragma: no cover
    UFuncTypeError = None

from ...typing_ import ChunkType, OperatorType, TileableType
from ..entity import (
    ExecutableTuple,
    OutputType,
    get_fetch_class,
    get_output_types,
    get_tileable_types,
)
from ..mode import is_eager_mode

_op_type_to_executor: Dict[Type[OperatorType], Callable] = dict()
_op_type_to_size_estimator: Dict[Type[OperatorType], Callable] = dict()


class TileableOperatorMixin:
    __slots__ = ()

    def check_inputs(self, inputs: List[TileableType]):
        if not inputs:
            return

        for inp in inputs:
            if inp is not None and inp._need_execution():
                raise ValueError(
                    f"{inp} has unknown dtypes, "
                    f"it must be executed first before {str(type(self))}"
                )

    @classmethod
    def _check_if_gpu(cls, inputs: List[TileableType]):
        if not inputs:
            return None
        true_num = 0
        for inp in inputs:
            op = getattr(inp, "op", None)
            if op is None or op.gpu is None:
                return None
            true_num += int(op.gpu)
        if true_num == len(inputs):
            return True
        elif true_num == 0:
            return False
        return None

    def _tokenize_output(self, output_idx: int, **kw):
        return f"{self._key}_{output_idx}"

    @staticmethod
    def _fill_nan_shape(kw: dict):
        nsplits = kw.get("nsplits")
        shape = kw.get("shape")
        if nsplits is not None and shape is not None:
            nsplits = tuple(nsplits)
            shape = list(shape)
            for idx, (s, sp) in enumerate(zip(shape, nsplits)):
                if not pd.isna(s):
                    continue
                s = sum(sp)
                if not np.isnan(s):
                    shape[idx] = s
            kw["shape"] = tuple(shape)
            kw["nsplits"] = nsplits
        return kw

    def _create_tileable(self, output_idx: int, **kw) -> TileableType:
        output_type = kw.pop("output_type", self._get_output_type(output_idx))
        if output_type is None:
            raise ValueError("output_type should be specified")

        if isinstance(output_type, (list, tuple)):
            output_type = output_type[output_idx]

        tileable_type, tileable_data_type = get_tileable_types(output_type)
        kw["_i"] = output_idx
        kw["op"] = self
        if output_type == OutputType.scalar:
            # tensor
            kw["order"] = "C_ORDER"

        kw = self._fill_nan_shape(kw)

        # key of output chunks may only contain keys for its output ids
        if "_key" not in kw:
            kw["_key"] = self._tokenize_output(output_idx, **kw)

        data = tileable_data_type(**kw)
        return tileable_type(data)

    def _new_tileables(
        self, inputs: List[TileableType], kws: List[dict] = None, **kw
    ) -> List[TileableType]:
        assert (
            isinstance(inputs, (list, tuple)) or inputs is None
        ), f"{inputs} is not a list"

        output_limit = kw.pop("output_limit", None)
        if output_limit is None:
            output_limit = getattr(self, "output_limit")

        with rewrite_stop_iteration():
            self._set_inputs(self, inputs)
        if self.gpu is None:
            self.gpu = self._check_if_gpu(self._inputs)
        if getattr(self, "_key", None) is None:
            self._update_key()  # update key when inputs are set

        tileables = []
        for j in range(output_limit):
            create_tensor_kw = kw.copy()
            if kws:
                create_tensor_kw.update(kws[j])
            tileable = self._create_tileable(j, **create_tensor_kw)
            tileables.append(tileable)

        self.outputs = tileables
        if len(tileables) > 1:
            # for each output tileable, hold the reference to the other outputs
            # so that either no one or everyone are gc collected
            for j, t in enumerate(tileables):
                t.data._siblings = [
                    tileable.data for tileable in tileables[:j] + tileables[j + 1 :]
                ]
        return tileables

    def new_tileables(
        self, inputs: List[TileableType], kws: List[dict] = None, **kw
    ) -> List[TileableType]:
        """
        Create tileable objects(Tensors or DataFrames).

        This is a misc function for create tileable objects like tensors or dataframes,
        it will be called inside the `new_tensors` and `new_dataframes`.
        If eager mode is on, it will trigger the execution after tileable objects are created.

        Parameters
        ----------
        inputs : list
            Input tileables
        kws : List[dict]
            Kwargs for each output.
        kw : dict
            Common kwargs for all outputs.

        Returns
        -------
        tileables : list
            Output tileables.

        .. note::
            It's a final method, do not override.
            Override the method `_new_tileables` if needed.
        """
        tileables = self._new_tileables(inputs, kws=kws, **kw)
        if is_eager_mode():
            ExecutableTuple(tileables).execute()
        return tileables

    def new_tileable(
        self, inputs: List[TileableType], kws: List[Dict] = None, **kw
    ) -> TileableType:
        if getattr(self, "output_limit") != 1:
            raise TypeError("cannot new chunk with more than 1 outputs")

        return self.new_tileables(inputs, kws=kws, **kw)[0]

    def get_fetch_op_cls(self, obj: Union[ChunkType, OutputType]):
        from .shuffle import ShuffleProxy

        if isinstance(obj, OutputType):
            output_types = [obj or OutputType.object]
        else:
            output_types = get_output_types(obj, unknown_as=OutputType.object)
        fetch_cls, fetch_shuffle_cls = get_fetch_class(output_types[0])
        if isinstance(self, ShuffleProxy):
            cls = fetch_shuffle_cls
        else:
            cls = fetch_cls

        def _inner(**kw):
            return cls(output_types=output_types, **kw)

        return _inner

    @classmethod
    def register_executor(cls, executor: Callable):
        _op_type_to_executor[cls] = executor

    @classmethod
    def unregister_executor(cls):
        del _op_type_to_executor[cls]

    @classmethod
    def register_size_estimator(cls, size_estimator: Callable):
        _op_type_to_size_estimator[cls] = size_estimator

    @classmethod
    def unregister_size_estimator(cls):
        del _op_type_to_size_estimator[cls]


def execute(results: Dict[str, Any], op: OperatorType):
    try:
        executor = _op_type_to_executor[type(op)]
    except KeyError:
        executor = type(op).execute

    # pre execute
    op.pre_execute(results, op)
    succeeded = False
    try:
        if UFuncTypeError is None:  # pragma: no cover
            return executor(results, op)
        else:
            # Cast `UFuncTypeError` to `TypeError` since subclasses of the former is unpickleable.
            # The `UFuncTypeError` was introduced by numpy#12593 since v1.17.0.
            try:
                result = executor(results, op)
                succeeded = True
                return result
            except UFuncTypeError as e:  # pragma: no cover
                raise TypeError(str(e)).with_traceback(sys.exc_info()[2]) from None
    except NotImplementedError:
        for op_cls in type(op).__mro__:
            if op_cls in _op_type_to_executor:
                executor = _op_type_to_executor[op_cls]
                _op_type_to_executor[type(op)] = executor
                result = executor(results, op)
                succeeded = True
                return result
        raise KeyError(f"No handler found for op: {op}")
    finally:
        if succeeded:
            op.post_execute(results, op)


def estimate_size(results: Dict[str, Any], op: OperatorType):
    try:
        size_estimator = _op_type_to_size_estimator[type(op)]
    except KeyError:
        size_estimator = type(op).estimate_size

    try:
        return size_estimator(results, op)
    except NotImplementedError:
        for op_cls in type(op).__mro__:
            if op_cls in _op_type_to_size_estimator:
                size_estimator = _op_type_to_size_estimator[op_cls]
                _op_type_to_size_estimator[type(op)] = size_estimator
                return size_estimator(results, op)
        raise KeyError(f"No handler found for op: {op} to estimate size")


def estimate_tileable_execution_size(
    tileable_graph: "TileableGraph",
    fetch_sizes: Optional[Dict[str, int]] = None,
) -> Union[int, float]:
    ctx = dict()
    ctx.update(fetch_sizes)
    ref_counts = defaultdict(lambda: 0)
    max_size = 0

    for tileable in tileable_graph:
        for key in set(inp.key for inp in tileable.inputs or ()):
            ref_counts[key] += 1

    for tileable in tileable_graph.topological_iter():
        estimate_size(ctx, tileable.op)
        max_size = max(max_size, sum(ctx.values()))
        if math.isinf(max_size):
            return float("inf")
        for inp in set(inp.key for inp in tileable.inputs or ()):
            ref_counts[inp] -= 1
            if ref_counts[inp] <= 0:
                ctx.pop(inp, None)
    return max_size
