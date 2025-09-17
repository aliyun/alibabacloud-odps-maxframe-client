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

from typing import Dict

from ..lib.dtypes_extension.blob import AbstractExternalBlob, SolidBlob
from .core import Serializer


class ExternalBlobSerializer(Serializer):
    def serial(self, obj: AbstractExternalBlob, context: Dict):
        _, vals = obj.__reduce__()
        return [type(obj).__name__], list(vals), False

    def deserial(self, serialized, context, subs):
        cls_name = serialized[0]
        cls = AbstractExternalBlob.get_cls_by_name(cls_name)
        return cls(*subs)


ExternalBlobSerializer.register(SolidBlob)
