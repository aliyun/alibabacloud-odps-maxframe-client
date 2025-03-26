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

from typing import Any, Dict, List

from ..... import opcodes
from .....serialization.serializables import StringField
from ..core import LLMTextGenOperator
from ..text import TextLLM


class ManagedLLMTextGenOperator(LLMTextGenOperator):
    _op_type_ = opcodes.MANAGED_TEXT_MODAL_GENERATION

    inference_framework: str = StringField("inference_framework", default=None)


class ManagedTextLLM(TextLLM):
    def generate(
        self,
        data,
        prompt_template: List[Dict[str, Any]],
        params: Dict[str, Any] = None,
        **kw
    ):
        return ManagedLLMTextGenOperator(
            model=self, prompt_template=prompt_template, params=params, **kw
        )(data)
