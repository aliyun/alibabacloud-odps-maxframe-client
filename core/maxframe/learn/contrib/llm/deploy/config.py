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

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from .....protocol import Serializable
from .....serialization.serializables import StringField
from .....serialization.serializables.field import (
    DictField,
    EnumField,
    Int32Field,
    ListField,
)
from .....serialization.serializables.field_type import FieldTypes
from .framework import InferenceFrameworkEnum


class ModelDeploymentConfig(Serializable):
    """
    Model deployment configuration for extending MaxFrame with custom models.

    This configuration is designed for users who need to deploy models that are not
    available within MaxFrame's built-in model offerings. It provides a way to specify
    custom deployment solutions by informing each MaxFrame worker which framework to use,
    which model path to load, and how to load it.

    The configuration assumes that models are already set up in the container image or
    mounted paths, and uses the current deploy_config to load them. Users are responsible
    for ensuring the runtime environment state and compatibility.

    Parameters
    ----------
    model_name: str
        The name of the model.
    model_file: str
        The file path of the model.
    inference_framework_type: InferenceFrameworkEnum
        The inference framework of the model.
    required_resource_files: List[Union[str, Any]]
        The required resource files of the model.
    load_params: Dict[str, Any]
        The load params of the model.
    required_cpu: int
        The required cpu of the model.
    required_memory: int
        The required memory of the model.
    required_gu: int
        The required gu of the model.
    required_gpu_memory: int
        The required gpu memory of the model.
    device: str
        The device of the model. One of "cpu" or "cuda".
    properties: Dict[str, Any]
        The properties of the model.
    tags: List[str]
        The tags of the model.

    Notes
    -----
    - Preview version for model deployments, all fields could be changed in the future.

    **User Responsibility Notice**: Users must have a complete understanding of what
    they are computing and ensure they fully comprehend the implications of their
    configuration choices. You are responsible for:

    * Ensuring model compatibility with the specified inference framework
    * Verifying that model files exist and are accessible in the runtime environment
    * Confirming that resource requirements (CPU, memory, GPU) are adequate
    * Validating that all dependencies and libraries are properly installed
    * Understanding the computational behavior and characteristics of your chosen model

    Examples
    --------
    >>> from maxframe.learn.contrib.llm.deploy.config import ModelDeploymentConfig
    >>> from maxframe.learn.contrib.llm.deploy.framework import InferenceFrameworkEnum
    >>> from maxframe.learn.contrib.llm.models.managed import ManagedTextGenLLM

    >>> # Configure model deployment with VLLM framework
    >>> model_config = ModelDeploymentConfig(
    ...     # Path to the model files (assumed to be available in container/mount)
    ...     model_file="/models/Qwen3-4B-Instruct-2507-FP8",
    ...     # Use VLLM serving framework for text generation
    ...     inference_framework_type=InferenceFrameworkEnum.VLLM_SERVE_TEXT,
    ...     # Framework-specific loading parameters
    ...     load_params={
    ...         "max_context_tokens": 4096,  # Maximum context length for the model
    ...         "max_startup_wait_seconds": 600  # Max wait time for model startup
    ...     },
    ...     # Target device for inference
    ...     device="cuda",
    ...     # Resource requirements (2 GPU units)
    ...     required_gu=2,
    ...     # Model tags for capabilities
    ...     tags=["text-generation"],
    ... )

    >>> # Create managed LLM instance with the deployment configuration
    >>> llm = ManagedTextGenLLM(name="my-model", deploy_config=model_config)
    >>> # Generate text using the deployed model.
    >>> result_df = llm.generate(
    ...     df,  # Input DataFrame containing prompts
    ...     prompt_template=messages,  # Template for formatting prompts
    ...     running_options={
    ...         "max_context_tokens": 4096,  # Runtime context limit
    ...     },
    ...     params={
    ...         "temperature": 0.7,  # Sampling temperature
    ...         "max_tokens": 2048  # Maximum tokens to generate
    ...     },
    ... )

    To make this model config working with the ManagedTextGenLLM,
    you need to provide a custom image with the required dependencies and model files.

    """

    model_name: str = StringField("model_name")
    model_file: str = StringField("model_file")
    inference_framework_type: InferenceFrameworkEnum = EnumField(
        "inference_framework_type", enum_type=InferenceFrameworkEnum
    )
    required_resource_files: List[Union[str, Any]] = ListField(
        "required_resource_files", field_type=FieldTypes.any, default_factory=list
    )
    load_params: Dict[str, Any] = DictField(
        "load_params",
        key_type=FieldTypes.string,
        value_type=FieldTypes.any,
        default_factory=dict,
    )
    required_cpu: Optional[int] = Int32Field("required_cpu", default=None)
    required_memory: Optional[int] = Int32Field("required_memory", default=None)
    required_gu: Optional[int] = Int32Field("required_gu", default=None)
    required_gpu_memory: Optional[int] = Int32Field("required_gpu_memory", default=None)
    device: str = StringField("device")
    properties: Dict[str, Any] = DictField(
        "properties",
        key_type=FieldTypes.string,
        value_type=FieldTypes.any,
        default_factory=dict,
    )
    tags: List[str] = ListField(
        "tags",
        field_type=FieldTypes.string,
        default_factory=list,
    )

    def is_reasoning_model(self):
        if self.properties is None:
            return False
        return self.properties.get("reasoning_model", False)

    def copy(self) -> "ModelDeploymentConfig":
        return deepcopy(self)

    def __eq__(self, other):
        if not isinstance(other, ModelDeploymentConfig):
            return False

        return (
            self.model_name == other.model_name
            and self.model_file == other.model_file
            and self.inference_framework_type == other.inference_framework_type
            and self.required_resource_files == other.required_resource_files
            and self.load_params == other.load_params
            and self.required_cpu == other.required_cpu
            and self.required_memory == other.required_memory
            and self.required_gu == other.required_gu
            and self.required_gpu_memory == other.required_gpu_memory
            and self.device == other.device
            and self.properties == other.properties
            and self.tags == other.tags
        )

    def __hash__(self):
        return hash(
            (
                self.model_name,
                self.model_file,
                self.inference_framework_type,
                self.required_resource_files,
                self.load_params,
                self.required_cpu,
                self.required_memory,
                self.required_gu,
                self.required_gpu_memory,
                self.device,
                self.properties,
                self.tags,
            )
        )

    def check_validity(self):
        required_fields = [
            "model_name",
            "model_file",
            "inference_framework_type",
            "device",
        ]
        for field in required_fields:
            if getattr(self, field) is None:
                raise ValueError(f"{field} is required")

        one_of_fields = ["required_cpu", "required_gu"]
        if not any(getattr(self, field) is not None for field in one_of_fields):
            raise ValueError(f"At least one of {one_of_fields} is required")

        if not self.tags:
            raise ValueError("tags is required")
