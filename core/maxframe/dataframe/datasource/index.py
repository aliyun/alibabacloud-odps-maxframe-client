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

import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import BoolField, DataTypeField, IndexField
from ..operators import DataFrameOperatorMixin
from ..utils import is_cudf, parse_index
from .core import PandasDataSourceOperator


class IndexDataSource(PandasDataSourceOperator, DataFrameOperatorMixin):
    """
    Represent data from pandas Index
    """

    _op_type_ = opcodes.INDEX_DATA_SOURCE

    data = IndexField("data")
    dtype = DataTypeField("dtype")
    store_data = BoolField("store_data")

    def __init__(self, data=None, dtype=None, gpu=None, store_data=None, **kw):
        if dtype is None and data is not None:
            dtype = data.dtype
        if gpu is None and is_cudf(data):  # pragma: no cover
            gpu = True
        super().__init__(
            data=data,
            dtype=dtype,
            gpu=gpu,
            store_data=store_data,
            _output_types=[OutputType.index],
            **kw
        )

    def __call__(self, shape=None, chunk_size=None, inp=None, name=None, names=None):
        if inp is None:
            # create from pandas Index
            name = name if name is not None else self.data.name
            names = names if names is not None else self.data.names
            return self.new_index(
                None,
                shape=shape,
                dtype=self.dtype,
                index_value=parse_index(self.data, store_data=self.store_data),
                name=name,
                names=names,
                raw_chunk_size=chunk_size,
            )
        elif hasattr(inp, "index_value"):
            # get index from MaxFrame DataFrame, Series or Index
            name = name if name is not None else inp.index_value.name
            names = names if names is not None else inp.index_value.names
            if inp.index_value.has_value():
                self.data = data = inp.index_value.to_pandas()
                return self.new_index(
                    None,
                    shape=(inp.shape[0],),
                    dtype=data.dtype,
                    index_value=parse_index(data, store_data=self.store_data),
                    name=name,
                    names=names,
                    raw_chunk_size=chunk_size,
                )
            else:
                if self.dtype is None:
                    self.dtype = inp.index_value.to_pandas().dtype
                return self.new_index(
                    [inp],
                    shape=(inp.shape[0],),
                    dtype=self.dtype,
                    index_value=inp.index_value,
                    name=name,
                    names=names,
                )
        else:
            if inp.ndim != 1:
                raise ValueError("Index data must be 1-dimensional")
            # get index from tensor
            dtype = inp.dtype if self.dtype is None else self.dtype
            pd_index = pd.Index([], dtype=dtype)
            if self.dtype is None:
                self.dtype = pd_index.dtype
            return self.new_index(
                [inp],
                shape=inp.shape,
                dtype=self.dtype,
                index_value=parse_index(pd_index, inp, store_data=self.store_data),
                name=name,
                names=names,
            )


def from_pandas(data, chunk_size=None, gpu=None, sparse=False, store_data=False):
    op = IndexDataSource(
        data=data, gpu=gpu, sparse=sparse, dtype=data.dtype, store_data=store_data
    )
    return op(shape=data.shape, chunk_size=chunk_size)


def from_tileable(tileable, dtype=None, name=None, names=None):
    op = IndexDataSource(gpu=tileable.op.gpu, sparse=tileable.issparse(), dtype=dtype)
    return op(inp=tileable, name=name, names=names)
