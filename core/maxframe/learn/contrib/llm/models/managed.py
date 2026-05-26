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

from typing import Any, Dict, List, Optional, Union

from maxframe import opcodes
from maxframe.learn.contrib.llm.core import (
    TASK_IMAGE_TEXT_TO_TEXT,
    TASK_MULTI_MODAL_EMBEDDING,
    TASK_SENTENCE_EMBEDDING,
    LLMTextEmbeddingOp,
    LLMTextGenOperator,
)
from maxframe.learn.contrib.llm.deploy.config import ModelDeploymentConfig
from maxframe.learn.contrib.llm.multi_modal import (
    LLMMultiModalGenerationOp,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingOp,
    MultiModalGenLLM,
)
from maxframe.learn.contrib.llm.text import TextEmbeddingModel, TextGenLLM
from maxframe.serialization.serializables import BoolField, ReferenceField, StringField


class ManagedModelMixin:
    __slots__ = ()

    def __init__(
        self, name: str, deploy_config: Optional[ModelDeploymentConfig] = None, **kw
    ):
        if deploy_config:
            deploy_config = deploy_config.copy()
            deploy_config.model_name = name
            deploy_config.check_validity()
        super().__init__(name=name, deploy_config=deploy_config, **kw)


class ManagedLLMTextGenOp(LLMTextGenOperator):
    _op_type_ = opcodes.MANAGED_TEXT_MODAL_GENERATION
    _legacy_name = "ManagedLLMTextGenOperator"

    simple_output = BoolField("simple_output", default=False)
    inference_framework: str = StringField("inference_framework", default=None)


class ManagedTextGenLLM(ManagedModelMixin, TextGenLLM):
    """
    Managed text LLM by MaxFrame.
    """

    _legacy_name = "ManagedTextLLM"  # since v2.3.0
    deploy_config: ModelDeploymentConfig = ReferenceField(
        "deploy_config", reference_type=ModelDeploymentConfig, default=None
    )

    def generate(
        self,
        data,
        prompt_template: Union[str, List[Dict[str, Any]]],
        simple_output: bool = False,
        params: Optional[Dict[str, Any]] = None,
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


class ManagedTextEmbeddingModel(ManagedModelMixin, TextEmbeddingModel):
    """
    Managed text embedder by MaxFrame.
    """

    deploy_config: ModelDeploymentConfig = ReferenceField(
        "deploy_config", reference_type=ModelDeploymentConfig, default=None
    )

    def embed(
        self,
        series,
        dimensions: Optional[int] = None,
        encoding_format: Optional[str] = None,
        simple_output: bool = False,
        params: Optional[Dict[str, Any]] = None,
        *,
        input: Optional[str] = None,
        **kw
    ):
        if input is not None and not isinstance(input, str):
            raise TypeError("input must be a string")
        return ManagedLLMTextEmbeddingOp(
            model=self,
            input=input,
            dimensions=dimensions,
            encoding_format=encoding_format,
            simple_output=simple_output,
            params=params,
            task=TASK_SENTENCE_EMBEDDING,
            **kw,
        )(series)


ManagedLLMTextGenOperator = ManagedLLMTextGenOp


class ManagedLLMMultiModalGenerationOp(LLMMultiModalGenerationOp):
    _op_type_ = opcodes.MANAGED_MULTI_MODAL_GENERATION

    simple_output = BoolField("simple_output", default=False)
    inference_framework: str = StringField("inference_framework", default=None)


class ManagedLLMMultiModalEmbeddingOp(MultiModalEmbeddingOp):
    _op_type_ = opcodes.MANAGED_MULTI_MODAL_EMBEDDING

    inference_framework: str = StringField("inference_framework", default=None)


class ManagedMultiModalGenLLM(ManagedModelMixin, MultiModalGenLLM):
    """
    Managed multimodal generation model by MaxFrame.
    """

    deploy_config: ModelDeploymentConfig = ReferenceField(
        "deploy_config", reference_type=ModelDeploymentConfig, default=None
    )

    def generate(
        self,
        data,
        messages=None,
        prompt_template=None,
        simple_output: bool = False,
        params: Optional[Dict[str, Any]] = None,
        **kw
    ):
        prompt_template = messages if messages is not None else prompt_template
        if prompt_template is None:
            raise ValueError("messages or prompt_template is required")
        return ManagedLLMMultiModalGenerationOp(
            model=self,
            prompt_template=prompt_template,
            simple_output=simple_output,
            params=params,
            task=TASK_IMAGE_TEXT_TO_TEXT,
            **kw,
        )(data)


class ManagedMultiModalEmbeddingModel(ManagedModelMixin, MultiModalEmbeddingModel):
    """
    Managed multimodal embedding model by MaxFrame.
    """

    deploy_config: ModelDeploymentConfig = ReferenceField(
        "deploy_config", reference_type=ModelDeploymentConfig, default=None
    )

    def embed(
        self,
        data,
        input,
        simple_output: bool = False,
        params: Optional[Dict[str, Any]] = None,
        **kw
    ):
        return ManagedLLMMultiModalEmbeddingOp(
            model=self,
            input=input,
            simple_output=simple_output,
            params=params,
            task=TASK_MULTI_MODAL_EMBEDDING,
            **kw,
        )(data)
