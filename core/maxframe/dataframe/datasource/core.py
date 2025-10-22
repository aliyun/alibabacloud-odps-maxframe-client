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

import asyncio
from typing import List, MutableMapping, Optional, Union

from ...serialization.serializables import Int64Field, StringField
from ...utils import estimate_pandas_size
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import validate_dtype_backend


class HeadOptimizedDataSource(DataFrameOperator, DataFrameOperatorMixin):
    __slots__ = ()
    # Data source op that optimized for head,
    # First, it will try to trigger first_chunk.head() and raise TilesError,
    # When iterative tiling is triggered,
    # check if the first_chunk.head() meets requirements.
    nrows = Int64Field("nrows", default=None)


class ColumnPruneSupportedDataSourceMixin(DataFrameOperatorMixin):
    __slots__ = ()

    def get_columns(self):  # pragma: no cover
        raise NotImplementedError

    def set_pruned_columns(self, columns, *, keep_order=None):  # pragma: no cover
        raise NotImplementedError


class _IncrementalIndexRecorder:
    _done: List[Optional[asyncio.Event]]
    _chunk_sizes: List[Optional[int]]

    def __init__(self, n_chunk: int):
        self._n_chunk = n_chunk
        self._done = [asyncio.Event() for _ in range(n_chunk)]
        self._chunk_sizes = [None] * n_chunk
        self._waiters = set()

    def _can_destroy(self):
        return all(e.is_set() for e in self._done) and not self._waiters

    def add_waiter(self, i: int):
        self._waiters.add(i)

    async def wait(self, i: int):
        if i == 0:
            return 0, self._can_destroy()
        self._waiters.add(i)
        try:
            await asyncio.gather(*(e.wait() for e in self._done[:i]))
        finally:
            self._waiters.remove(i)
        # all chunk finished and no waiters
        return sum(self._chunk_sizes[:i]), self._can_destroy()

    async def finish(self, i: int, size: int):
        self._chunk_sizes[i] = size
        self._done[i].set()


class IncrementalIndexDatasource(HeadOptimizedDataSource):
    __slots__ = ()

    incremental_index_recorder_name = StringField("incremental_index_recorder_name")


class PandasDataSourceOperator(DataFrameOperator):
    def get_data(self):
        return getattr(self, "data", None)

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "PandasDataSourceOperator"
    ):
        ctx[op.outputs[0].key] = estimate_pandas_size(op.get_data())


class DtypeBackendCompatibleMixin:
    def __on_deserialize__(self):
        self.dtype_backend = validate_dtype_backend(self.dtype_backend)
