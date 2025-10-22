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

from typing import Any, Dict

from ..... import opcodes
from .....serialization.serializables.core import Serializable
from .....serialization.serializables.field import StringField
from ..core import LLMTextGenOperator
from ..multi_modal import MultiModalLLM
from ..text import TextGenLLM


class DashScopeLLMMixin(Serializable):
    __slots__ = ()

    _not_supported_params = {"stream", "incremental_output"}

    def validate_params(self, params: Dict[str, Any]):
        for k in params.keys():
            if k in self._not_supported_params:
                raise ValueError(f"{k} is not supported")


class DashScopeTextLLM(TextGenLLM, DashScopeLLMMixin):
    """
    DashScope text LLM.
    """

    api_key_resource = StringField("api_key_resource", default=None)

    def __init__(self, name: str, api_key_resource: str):
        """
        Initialize a DashScope text LLM.

        Parameters
        ----------
        name : str
            The LLM name to use, check DashScope for `available models <https://help.aliyun.com/zh/model-studio/getting-started/models>`_.
        api_key_resource : str
            The MaxCompute resource file name containing the DashScope API key.
        """
        super().__init__(name=name, api_key_resource=api_key_resource)

    def generate(
        self,
        data,
        prompt_template: Dict[str, Any],
        params: Dict[str, Any] = None,
    ):
        return DashScopeTextGenerationOp(
            model=self,
            prompt_template=prompt_template,
            params=params,
        )(data)


class DashScopeMultiModalLLM(MultiModalLLM, DashScopeLLMMixin):
    """
    DashScope multi-modal LLM.
    """

    api_key_resource = StringField("api_key_resource", default=None)

    def __init__(self, name: str, api_key_resource: str):
        """
        Initialize a DashScope multi-modal LLM.

        Parameters
        ----------
        name : str
            The LLM name to use, check DashScope for `available models <https://help.aliyun.com/zh/model-studio/getting-started/models>`_.
        api_key_resource : str
            The MaxCompute resource file name containing the DashScope API key.
        """
        super().__init__(name=name, api_key_resource=api_key_resource)

    def generate(
        self,
        data,
        prompt_template: Dict[str, Any],
        params: Dict[str, Any] = None,
    ):
        # TODO add precheck here
        return DashScopeMultiModalGenerationOp(
            model=self,
            prompt_template=prompt_template,
            params=params,
        )(data)


class DashScopeTextGenerationOp(LLMTextGenOperator):
    _op_type_ = opcodes.DASHSCOPE_TEXT_GENERATION
    _legacy_name = "DashScopeTextGenerationOperator"


class DashScopeMultiModalGenerationOp(LLMTextGenOperator):
    _op_type_ = opcodes.DASHSCOPE_MULTI_MODAL_GENERATION
    _legacy_name = "DashScopeMultiModalGenerationOperator"


DashScopeTextGenerationOperator = DashScopeTextGenerationOp
DashScopeMultiModalGenerationOperator = DashScopeMultiModalGenerationOp
