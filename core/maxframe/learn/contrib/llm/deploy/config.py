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

REASONING_MODEL_KEY = "reasoning_model"


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
    model_name : str
        The name of the model.
    model_file : str
        The **local** file path of the model, e.g., ``"/mnt/models/qwen/"``.

        Note: OSS paths (``oss://...``) are NOT supported directly.

    inference_framework_type : InferenceFrameworkEnum
        The inference framework of the model.
    required_resource_files : List[Union[str, Any]]
        The required resource files of the model.
    load_params : Dict[str, Any]
        The load params of the model.
    required_cpu : int
        The required cpu of the model.
    required_memory : int
        The required memory of the model.
    required_gu : int
        The required gu of the model.
    required_gpu_memory : int
        The required gpu memory of the model.
    device : str, optional
        The device of the model. One of "cpu" or "cuda". Defaults to None,
        which allows the server to determine the device at runtime.
    properties : Dict[str, Any]
        The properties of the model.
    tags : List[str]
        The tags of the model.

    envs : Dict[str, str]
        Custom environment variables for the inference subprocess.
        Example: ``{"CUDA_VISIBLE_DEVICES": "0", "HF_HOME": "/mnt/cache"}``

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

    """

    model_name: str = StringField("model_name")
    model_file: str = StringField("model_file")
    inference_framework_type: InferenceFrameworkEnum = EnumField(
        "inference_framework_type", enum_type=InferenceFrameworkEnum, default=None
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
    device: str = StringField("device", default=None)
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
    # Custom environment variables for inference subprocess
    envs: Dict[str, str] = DictField(
        "envs",
        key_type=FieldTypes.string,
        value_type=FieldTypes.string,
        default_factory=dict,
    )
    image: Optional[Dict[str, Any]] = DictField(
        "image",
        key_type=FieldTypes.string,
        value_type=FieldTypes.any,
        default_factory=dict,
    )
    # Inference parameters from model metadata, merged into load_params at runtime
    inference_parameters: Dict[str, Any] = DictField(
        "inference_parameters",
        key_type=FieldTypes.string,
        value_type=FieldTypes.any,
        default_factory=dict,
    )

    def is_reasoning_model(self):
        if not self.properties:
            return False
        return self.properties.get(REASONING_MODEL_KEY, False)

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
            and self.envs == other.envs
            and self.image == other.image
            and self.inference_parameters == other.inference_parameters
        )

    def __hash__(self):
        return hash(
            (
                self.model_name,
                self.model_file,
                self.inference_framework_type,
                (
                    tuple(self.required_resource_files)
                    if self.required_resource_files
                    else None
                ),
                tuple(sorted(self.load_params.items())) if self.load_params else None,
                self.required_cpu,
                self.required_memory,
                self.required_gu,
                self.required_gpu_memory,
                self.device,
                tuple(sorted(self.properties.items())) if self.properties else None,
                tuple(self.tags) if self.tags else None,
                tuple(sorted(self.envs.items())) if self.envs else None,
                tuple(sorted(self.image.items())) if self.image else None,
                (
                    tuple(sorted(self.inference_parameters.items()))
                    if self.inference_parameters
                    else None
                ),
            )
        )

    def check_validity(self):
        """
        Validate the configuration and raise ValueError if invalid.

        This method performs client-side validation to catch configuration
        errors early, before submission to the server.
        """
        required_fields = [
            "model_name",
        ]
        for field in required_fields:
            if getattr(self, field) is None:
                raise ValueError(f"{field} is required")

        required_cpu = self.required_cpu or 0
        required_gu = self.required_gu or 0
        if required_cpu <= 0 and required_gu <= 0:
            raise ValueError(
                "required_cpu or required_gu must be provided and greater than 0"
            )
