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

import logging

from ..operators import DataFrameOperator, DataFrameOperatorMixin

logger = logging.getLogger(__name__)


class LegacySeriesMethodOperator(DataFrameOperator, DataFrameOperatorMixin):
    _method_name = None
    _method_cls = None

    def __on_deserialize__(self):
        cls = type(self)
        local_fields = {
            f
            for f, name_hash in cls._FIELD_TO_NAME_HASH.items()
            if name_hash == cls._NAME_HASH
        }
        kw = {
            f: getattr(self, f)
            for f in cls._FIELD_TO_NAME_HASH
            if f not in local_fields and hasattr(self, f)
        }

        kw["method"] = self._method_name
        kw["method_kwargs"] = {
            f: getattr(self, f) for f in local_fields if hasattr(self, f)
        }

        logger.warning(f"Using deprecated operator class {cls.__name__}")
        return self._method_cls(**kw)
