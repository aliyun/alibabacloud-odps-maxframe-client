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

from ...serialization.serializables import BoolField
from ..entity import OutputType, register_fetch_class
from .base import Operator
from .core import TileableOperatorMixin
from .fetch import Fetch, FetchMixin
from .fuse import Fuse, FuseChunkMixin


class ObjectOperator(Operator):
    pass


class ObjectOperatorMixin(TileableOperatorMixin):
    _output_type_ = OutputType.object


class ObjectFuseChunkMixin(FuseChunkMixin, ObjectOperatorMixin):
    __slots__ = ()


class ObjectFuseChunk(ObjectFuseChunkMixin, Fuse):
    pass


class ObjectFetch(FetchMixin, ObjectOperatorMixin, Fetch):
    _output_type_ = OutputType.object

    def __init__(self, **kw):
        kw.pop("output_types", None)
        kw.pop("_output_types", None)
        super().__init__(**kw)

    def _new_tileables(self, inputs, kws=None, **kw):
        if "_key" in kw and self.source_key is None:
            self.source_key = kw["_key"]
        return super()._new_tileables(inputs, kws=kws, **kw)


register_fetch_class(OutputType.object, ObjectFetch, None)


class MergeDictOperator(ObjectOperator, ObjectOperatorMixin):
    _merge = BoolField("merge")

    def __init__(self, merge=None, **kw):
        super().__init__(_merge=merge, **kw)

    @property
    def merge(self):
        return self._merge

    @classmethod
    def concat_tileable_chunks(cls, tileable):
        assert not tileable.is_coarse()

        op = cls(merge=True)
        chunk = cls(merge=True).new_chunk(tileable.chunks)
        return op.new_tileable([tileable], chunks=[chunk], nsplits=((1,),))
