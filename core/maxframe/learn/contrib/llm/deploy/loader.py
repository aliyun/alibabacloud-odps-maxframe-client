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

import json
from typing import Any, Dict, List, Optional

from .....protocol import JsonSerializable
from .....serialization.serializables import (
    DictField,
    FieldTypes,
    Int32Field,
    ListField,
    ReferenceField,
    StringField,
)
from .core import ResourceNameTemplate, register_model_deployments

_ALLOWED_DEVICES = {"cpu", "gpu", "ppu", "tpu", "cuda"}


class DeploymentConfigResourceRequirement(JsonSerializable):
    cpu: int = Int32Field("cpu", default=1)
    memory: int = Int32Field("memory", default=4)
    gu: int = Int32Field("gu", default=0)
    gpu_memory: int = Int32Field("gpu_memory", default=0)

    def __init__(self, cpu, memory, **kw):
        gu = kw.get("gu")
        gpu_memory = kw.get("gpu_memory")
        if (
            not isinstance(cpu, int)
            or not isinstance(memory, int)
            or (gu and not isinstance(gu, int))
            or (gpu_memory and not isinstance(gpu_memory, int))
        ):
            raise TypeError("cpu, memory, gpu and gpu_memory must be int")
        super().__init__(cpu=cpu, memory=memory, gu=gu, gpu_memory=gpu_memory)

    def to_json(self) -> dict:
        ret = {
            "cpu": self.cpu,
            "memory": self.memory,
            "gu": self.gu,
            "gpu_memory": self.gpu_memory,
        }
        return ret

    @classmethod
    def from_json(cls, serialized: dict) -> "DeploymentConfigResourceRequirement":
        return cls(
            cpu=serialized.get("cpu", 1),
            memory=serialized.get("memory", 4),
            gu=serialized.get("gu", 0),
            gpu_memory=serialized.get("gpu_memory", 0),
        )


class DeploymentConfigBase(JsonSerializable):
    model_file: Optional[str] = StringField("model_file", default=None)
    model_file_resources: List[ResourceNameTemplate] = ListField(
        "model_file_resources", field_type=FieldTypes.reference, default_factory=list
    )
    resource_requirements: Optional[
        DeploymentConfigResourceRequirement
    ] = ReferenceField(
        "resource_requirements", DeploymentConfigResourceRequirement, default=None
    )
    load_params: Optional[Dict[str, Any]] = DictField(
        "load_params",
        key_type=FieldTypes.string,
        value_type=FieldTypes.any,
        default_factory=dict,
    )
    properties: Dict[str, Any] = DictField(
        "properties",
        key_type=FieldTypes.string,
        value_type=FieldTypes.any,
        default_factory=dict,
    )

    def __init__(
        self,
        model_file=None,
        model_file_resources=None,
        resource_requirements=None,
        load_params=None,
        properties=None,
        **kw,
    ):
        if model_file and not isinstance(model_file, str):
            raise TypeError("model_file must be a string")
        if resource_requirements and not isinstance(
            resource_requirements, DeploymentConfigResourceRequirement
        ):
            raise TypeError(
                "resource_requirements must be a DeploymentConfigResourceRequirement"
            )
        if load_params and not isinstance(load_params, dict):
            raise TypeError("load_params must be a dict")
        if properties and not isinstance(properties, dict):
            raise TypeError("properties must be a dict")
        if model_file_resources:
            if not isinstance(model_file_resources, list):
                raise TypeError("model_file_resources must be a list")
            if not all(
                isinstance(res, ResourceNameTemplate) for res in model_file_resources
            ):
                raise TypeError(
                    "model_file_resources must be a list of ResourceNameTemplate"
                )

        super().__init__(
            model_file=model_file,
            model_file_resources=model_file_resources,
            load_params=load_params,
            resource_requirements=resource_requirements,
            properties=properties,
            **kw,
        )

    @classmethod
    def from_json(cls, serialized: dict) -> "DeploymentConfigBase":
        kw = serialized.copy()
        resource_requirements_data = kw.get("resource_requirements")
        return cls(
            model_file=kw.get("model_file"),
            model_file_resources=[
                ResourceNameTemplate.from_json(res)
                for res in kw.get("model_file_resources", [])
            ],
            load_params=kw.get("load_params", {}),
            resource_requirements=(
                DeploymentConfigResourceRequirement.from_json(
                    resource_requirements_data
                )
                if resource_requirements_data
                else None
            ),
            properties=kw.get("properties", {}),
        )

    def to_json(self) -> dict:
        return {
            "properties": self.properties,
            "load_params": self.load_params,
            "model_file": self.model_file,
            "model_file_resources": [
                res.to_json() for res in self.model_file_resources
            ],
            "resource_requirements": (
                self.resource_requirements.to_json()
                if self.resource_requirements
                else None
            ),
        }


class FrameworkDeploymentConfig(DeploymentConfigBase):
    framework: str = StringField("framework")
    device: str = StringField("device", default="cpu")

    def __init__(
        self,
        **kw,
    ):
        framework = kw.get("framework")
        if not framework:
            raise ValueError("framework is required")

        device = kw.get("device")
        if not device:
            device = "cpu"
        if device not in _ALLOWED_DEVICES:
            raise ValueError(f"device must be one of {_ALLOWED_DEVICES}")

        super().__init__(**kw)

    @classmethod
    def from_json(cls, serialized: dict) -> "FrameworkDeploymentConfig":
        kw = serialized.copy()
        resource_requirements_data = kw.get("resource_requirements")
        return cls(
            model_file=kw.get("model_file"),
            model_file_resources=[
                ResourceNameTemplate.from_json(res)
                for res in kw.get("model_file_resources", [])
            ],
            load_params=kw.get("load_params", {}),
            resource_requirements=(
                DeploymentConfigResourceRequirement.from_json(
                    resource_requirements_data
                )
                if resource_requirements_data
                else None
            ),
            properties=kw.get("properties", {}),
            framework=kw["framework"],
            device=kw.get("device", "cpu"),
        )

    def to_json(self):
        ret = super().to_json()
        ret["framework"] = self.framework
        ret["device"] = self.device
        return ret


class ModelDeploymentConfig(JsonSerializable):
    model_name: str = StringField("model_name")
    common_config: DeploymentConfigBase = ReferenceField(
        "common_config", DeploymentConfigBase, default_factory=DeploymentConfigBase
    )
    deployments: List[FrameworkDeploymentConfig] = ListField(
        "deployments", field_type=FieldTypes.reference, default_factory=list
    )
    tags: List[str] = ListField(
        "tags", field_type=FieldTypes.string, default_factory=list
    )

    def __init__(self, model_name, deployments, **kwargs):
        if not model_name:
            raise ValueError("model_name is required")
        if not deployments:
            raise ValueError("deployments is required")
        else:
            if not isinstance(deployments, list):
                raise TypeError("deployments must be a list")
            if not all(
                isinstance(deployment, FrameworkDeploymentConfig)
                for deployment in deployments
            ):
                raise TypeError(
                    "deployments must be a list of FrameworkDeploymentConfig"
                )
        super().__init__(model_name=model_name, deployments=deployments, **kwargs)

    def to_json(self) -> dict:
        ret = {
            "model_name": self.model_name,
            "tags": self.tags,
        }
        if self.common_config:
            ret["common_config"] = self.common_config.to_json()
        if self.deployments:
            ret["deployments"] = [
                deployment.to_json() for deployment in self.deployments
            ]
        return ret

    @classmethod
    def from_json(cls, serialized: dict) -> "ModelDeploymentConfig":
        kw = serialized.copy()
        return cls(
            model_name=kw["model_name"],
            common_config=(
                DeploymentConfigBase.from_json(kw.get("common_config", {}))
                if kw.get("common_config")
                else None
            ),
            deployments=[
                FrameworkDeploymentConfig.from_json(deployment)
                for deployment in kw.get("deployments", [])
            ],
            tags=kw.get("tags", []),
        )


def _model_config_from_json(json_content: str) -> List[ModelDeploymentConfig]:
    data_list = json.loads(json_content)
    if not isinstance(data_list, list):
        raise ValueError("JSON root must be a list of model configurations.")
    return [ModelDeploymentConfig.from_json(item_data) for item_data in data_list]


def _model_config_to_json(
    model_configs: List[ModelDeploymentConfig],
) -> str:
    data_list = [config.to_json() for config in model_configs]
    return json.dumps(data_list, ensure_ascii=False)


def register_models_from_json_file(config_path):
    """
    Loads model configurations from a JSON file and registers them.
    """

    with open(config_path, "r") as f:
        model_list = _model_config_from_json(f.read())

        if not model_list:
            return

        for model in model_list:
            common_config = model.common_config or DeploymentConfigBase()
            deployments = model.deployments

            for deployment in deployments:
                model_file = deployment.model_file or common_config.model_file
                model_file_resources = (
                    deployment.model_file_resources
                    or common_config.model_file_resources
                )

                properties = deployment.properties or common_config.properties
                load_params = deployment.load_params or common_config.load_params
                calc_resources = (
                    deployment.resource_requirements
                    or common_config.resource_requirements
                )
                tags = list(set(model.tags or []))

                register_model_deployments(
                    model_name=model.model_name,
                    model_file=model_file,
                    inference_frameworks=deployment.framework,
                    required_resources=[
                        ResourceNameTemplate(
                            project=resource_item.project,
                            schema=resource_item.schema,
                            name=resource_item.name,
                        )
                        for resource_item in model_file_resources
                    ],
                    default_load_params=load_params,
                    required_cpu=calc_resources.cpu,
                    required_memory=calc_resources.memory,
                    required_gu=calc_resources.gu,
                    required_gpu_memory=calc_resources.gpu_memory,
                    device=deployment.device,
                    properties=properties,
                    tags=tags,
                )


def register_models_from_dir(config_dir):
    """
    Loads model configurations from a directory and registers them.
    """
    from pathlib import Path

    model_json_files = list(Path(config_dir).glob("*_models.json"))

    for file_path in model_json_files:
        try:
            register_models_from_json_file(file_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load model config from {file_path}, error: {e}"
            )
