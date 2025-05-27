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

import hashlib
from typing import TYPE_CHECKING, Callable, List, Optional

from odps import ODPS
from pandas.core.common import get_callable_name

from ....serialization.serializables import FieldTypes, ListField, StringField
from ....udf import PythonPackOptions
from ...core import AbstractUDF, UserCodeMixin

if TYPE_CHECKING:
    from odpsctx import ODPSSessionContext


class SpeUDF(AbstractUDF, UserCodeMixin):
    _name: str = StringField("name")
    encoded_content: List[str] = ListField("encoded_content", FieldTypes.string)

    def __init__(self, func: Optional[Callable] = None, **kwargs):
        if func is not None:
            kwargs["encoded_content"] = encoded_content = [
                self.generate_pickled_codes(func)
            ]
            content_str = "\n".join(encoded_content)
            kwargs["_name"] = (
                f"user_udf_{get_callable_name(func).replace('<lambda>', 'lambda')}_"
                f"{hashlib.md5(content_str.encode('utf-8')).hexdigest()}"
            )
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return self._name

    def unregister(self, odps: ODPS):
        # Running in SPE, do nothing
        pass

    def register(self, odps: ODPS, overwrite: bool = False):
        # Running in SPE, do nothing
        pass

    def collect_pythonpack(self) -> List[PythonPackOptions]:
        return []

    def load_pythonpack_resources(self, odps_ctx: "ODPSSessionContext") -> None:
        # Running in SPE, do nothing
        pass
