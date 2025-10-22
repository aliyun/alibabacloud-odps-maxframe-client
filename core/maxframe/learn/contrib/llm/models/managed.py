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

from typing import Any, Dict, List, Union

from ..... import opcodes
from .....serialization.serializables import BoolField, ReferenceField, StringField
from ..core import LLMTextEmbeddingOp, LLMTextGenOperator
from ..deploy.config import ModelDeploymentConfig
from ..text import TextEmbeddingModel, TextGenLLM


class ManagedLLMTextGenOp(LLMTextGenOperator):
    _op_type_ = opcodes.MANAGED_TEXT_MODAL_GENERATION
    _legacy_name = "ManagedLLMTextGenOperator"

    inference_framework: str = StringField("inference_framework", default=None)
    simple_output: bool = BoolField("simple_output", default=False)


class ManagedTextGenLLM(TextGenLLM):
    """
    Managed text LLM by MaxFrame.
    """

    _legacy_name = "ManagedTextLLM"  # since v2.3.0
    deploy_config: ModelDeploymentConfig = ReferenceField(
        "deploy_config", reference_type=ModelDeploymentConfig, default=None
    )

    def __init__(self, name: str, deploy_config: ModelDeploymentConfig = None):
        """
        Initialize a managed text LLM.

        Parameters
        ----------
        name : str
            The managed text LLM name to use.
        deploy_config : ModelDeploymentConfig
            The model deployment config to use.
        """
        if deploy_config:
            deploy_config.model_name = name
            deploy_config.check_validity()
        super().__init__(name=name, deploy_config=deploy_config)

    def generate(
        self,
        data,
        prompt_template: Union[str, List[Dict[str, Any]]],
        simple_output: bool = False,
        params: Dict[str, Any] = None,
        **kw
    ):
        return ManagedLLMTextGenOp(
            model=self,
            prompt_template=prompt_template,
            simple_output=simple_output,
            params=params,
            **kw,
        )(data)


# since v2.3.0, text llm has more types
ManagedTextLLM = ManagedTextGenLLM  # for old client compatibility


class ManagedLLMTextEmbeddingOp(LLMTextEmbeddingOp):
    _op_type_ = opcodes.LLM_TEXT_EMBEDDING_TASK
    inference_framework: str = StringField("inference_framework", default=None)


class ManagedTextEmbeddingModel(TextEmbeddingModel):
    """
    Managed text embedder by MaxFrame.
    """

    def __init__(self, name: str):
        """
        Initialize a managed text embedder.
        Parameters
        ----------
        name : str
            The managed text embedder name to use.
        """
        super().__init__(name=name)

    def embed(
        self,
        series,
        dimensions: int = None,
        encoding_format: str = None,
        simple_output: bool = False,
        params: Dict[str, Any] = None,
        **kw
    ):
        return ManagedLLMTextEmbeddingOp(
            model=self,
            dimensions=dimensions,
            encoding_format=encoding_format,
            simple_output=simple_output,
            params=params,
            task="text-embedding",
            **kw,
        )(series)


ManagedLLMTextGenOperator = ManagedLLMTextGenOp
