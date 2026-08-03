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

from maxframe.core import OutputType, register_fetch_class
from maxframe.core.operator import Fetch, FetchMixin, FetchShuffle
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.operators.core import LiteFrameOperatorMixin
from maxframe.serialization.serializables import ReferenceField, SeriesField


class LiteFrameFetchMixin(LiteFrameOperatorMixin, FetchMixin):
    __slots__ = ()


class LiteFrameFetch(LiteFrameFetchMixin, Fetch):
    dtypes = SeriesField("dtypes")
    frame_metadata = ReferenceField("frame_metadata", FrameMetadata, default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def _new_tileables(self, inputs, kws=None, **kw):
        if "_key" in kw and self.source_key is None:
            self.source_key = kw["_key"]
        return super()._new_tileables(inputs, kws=kws, **kw)


class LiteFrameFetchShuffle(LiteFrameFetchMixin, FetchShuffle):
    dtypes = SeriesField("dtypes")
    frame_metadata = ReferenceField("frame_metadata", FrameMetadata, default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)


register_fetch_class(OutputType.liteframe, LiteFrameFetch, LiteFrameFetchShuffle)
