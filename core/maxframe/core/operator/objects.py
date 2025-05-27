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

from ..entity import OutputType, register_fetch_class
from .base import Operator
from .core import TileableOperatorMixin
from .fetch import Fetch, FetchMixin


class ObjectOperator(Operator):
    pass


class ObjectOperatorMixin(TileableOperatorMixin):
    _output_type_ = OutputType.object


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
