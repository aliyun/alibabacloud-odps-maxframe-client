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
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import ExecutableTuple, OutputType, get_output_types
from ...core.operator import MapReduceOperator
from ...dataframe.utils import parse_index
from ...serialization.serializables import FieldTypes, Int64Field, TupleField
from ...tensor.utils import check_random_state, gen_random_seeds, validate_axis
from ...utils import tokenize
from ..core import LearnOperatorMixin
from . import convert_to_tensor_or_dataframe


def _shuffle_index_value(op, index_value, chunk_index=None):
    key = tokenize((op._values_, chunk_index, index_value.key))
    return parse_index(pd.Index([], index_value.to_pandas().dtype), key=key)


class LearnShuffle(MapReduceOperator, LearnOperatorMixin):
    _op_type_ = opcodes.PERMUTATION

    axes = TupleField("axes", FieldTypes.int32)
    seeds = TupleField("seeds", FieldTypes.uint32)
    n_samples = Int64Field("n_samples", default=None)

    reduce_sizes = TupleField("reduce_sizes", FieldTypes.uint32)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @property
    def output_limit(self):
        if self.stage is None:
            return len(self.output_types)
        return 1

    def _shuffle_index_value(self, index_value):
        return _shuffle_index_value(self, index_value)

    def _shuffle_dtypes(self, dtypes):
        seed = self.seeds[self.axes.index(1)]
        rs = np.random.RandomState(seed)
        shuffled_dtypes = dtypes[rs.permutation(np.arange(len(dtypes)))]
        return shuffled_dtypes

    def _calc_params(self, params):
        axes = set(self.axes)
        for i, output_type, param in zip(itertools.count(0), self.output_types, params):
            if output_type == OutputType.dataframe:
                if 0 in axes:
                    param["index_value"] = self._shuffle_index_value(
                        param["index_value"]
                    )
                if 1 in axes:
                    dtypes = param["dtypes"] = self._shuffle_dtypes(param["dtypes"])
                    param["columns_value"] = parse_index(dtypes.index, store_data=True)
            elif output_type == OutputType.series:
                if 0 in axes:
                    param["index_value"] = self._shuffle_index_value(
                        param["index_value"]
                    )
            param["_position_"] = i
        return params

    def __call__(self, arrays):
        params = self._calc_params([ar.params for ar in arrays])
        return self.new_tileables(arrays, kws=params)


def shuffle(*arrays, random_state=None, n_samples=None, axes=None):
    arrays = [convert_to_tensor_or_dataframe(ar) for ar in arrays]
    axes = axes or (0,)
    if not isinstance(axes, Iterable):
        axes = (axes,)
    elif not isinstance(axes, tuple):
        axes = tuple(axes)
    random_state = check_random_state(random_state).to_numpy()
    if n_samples:
        raise TypeError(f"n_samples argument of shuffle() not supported.")

    max_ndim = max(ar.ndim for ar in arrays)
    axes = tuple(np.unique([validate_axis(max_ndim, ax) for ax in axes]).tolist())
    seeds = gen_random_seeds(len(axes), random_state)

    # verify shape
    for ax in axes:
        shapes = {ar.shape[ax] for ar in arrays if ax < ar.ndim}
        if len(shapes) > 1:
            raise ValueError(f"arrays do not have same shape on axis {ax}")

    op = LearnShuffle(axes=axes, seeds=seeds, output_types=get_output_types(*arrays))
    shuffled_arrays = op(arrays)
    if len(arrays) == 1:
        return shuffled_arrays[0]
    else:
        return ExecutableTuple(shuffled_arrays)
