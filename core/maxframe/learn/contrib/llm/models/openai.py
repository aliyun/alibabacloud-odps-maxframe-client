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
from .....serialization.serializables.field import BoolField, Int32Field, StringField
from ..core import LLM, LLMTextGenOperator
from ..text import TextGenLLM


class OpenAICompatibleLLM(LLM):
    """
    Remote OpenAI-compatible text LLM.
    """

    base_url: str = StringField("base_url")
    api_key: str = StringField("api_key")
    batch_size: int = Int32Field("batch_size", default=None)
    batch_timeout: int = Int32Field("batch_timeout", default=None)


class OpenAICompatibleTextLLM(TextGenLLM, OpenAICompatibleLLM):
    def __init__(self, name: str, base_url: str, api_key: str):
        """
        Initialize a remote OpenAI-compatible text LLM.

        Parameters
        ----------
        name : str
            The remote model name to use.
        base_url : str
            The base URL of the OpenAI-compatible API service.
        api_key : str
            The API key for authentication.
        """
        super().__init__(name=name)
        self.base_url = base_url
        self.api_key = api_key

    def generate(
        self,
        data,
        prompt_template: List[Dict[str, Any]],
        simple_output: bool = False,
        params: Dict[str, Any] = None,
        **kw
    ):
        return OpenAICompatibleTextGenOp(
            model=self,
            prompt_template=prompt_template,
            simple_output=simple_output,
            params=params,
            **kw,
        )(data)


class OpenAICompatibleTextGenOp(LLMTextGenOperator):
    _op_type_ = opcodes.OPENAI_COMPATIBLE_TEXT_GENERATION

    simple_output = BoolField("simple_output", default=False)
