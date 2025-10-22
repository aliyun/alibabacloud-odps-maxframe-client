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

from typing import Any, Dict, List, Optional, Union

from odps.models import Resource as ODPSResource

from .....serialization.serializables import Serializable, StringField
from .config import ModelDeploymentConfig
from .framework import InferenceFrameworkEnum

_registered_llm_models = {}


class ResourceNameTemplate(Serializable):
    """
    A template-based resource name builder that supports variable substitution.
    All values in the substitution dict will have their hyphens (-) replaced with underscores (_).

    Example:
        template = ResourceNameTemplate(
            project="my_project",
            schema="schema_{region_id}",
            name="resource_{env}"
        )
        # Render with variables
        full_name = template.build_full_resource_name({"region_id": "cn-hangzhou", "env": "prod"})
        # Result: resource name with cn_hangzhou and prod substituted
    """

    project: str = StringField("project", default=None)
    schema: str = StringField("schema", default=None)
    name: str = StringField("name")

    def build_full_resource_name(self, template_vars: Dict[str, Any]):
        """
        Build full resource name with template variable substitution.
        """
        normalized_vars = {
            key: str(value).replace("-", "_") for key, value in template_vars.items()
        }

        project, schema, name = self.project, self.schema, self.name
        res_name = ODPSResource.build_full_resource_name(
            name=name, project_name=project, schema_name=schema
        )
        try:
            return res_name.format(**normalized_vars)
        except KeyError:
            # If template variable is missing, return as-is
            return res_name

    def __eq__(self, other):
        if not isinstance(other, ResourceNameTemplate):
            return False
        return (
            self.project == other.project
            and self.schema == other.schema
            and self.name == other.name
        )

    def __hash__(self):
        return hash((self.project, self.schema, self.name))

    def to_json(self) -> dict:
        ret = {"name": self.name}
        if self.project:
            ret["project"] = self.project
        if self.schema:
            ret["schema"] = self.schema
        return ret

    @classmethod
    def from_json(cls, serialized: dict) -> "ResourceNameTemplate":
        kw = serialized.copy()
        return cls(
            project=kw.get("project"),
            schema=kw.get("schema"),
            name=kw["name"],
        )


def create_model_deployments(
    model_name: str,
    model_file: str,
    inference_frameworks: Union[
        str, InferenceFrameworkEnum, List[InferenceFrameworkEnum]
    ],
    required_resources: List[Union[str, ResourceNameTemplate]] = None,
    default_load_params: Optional[Dict[str, Any]] = None,
    required_cpu: Optional[float] = None,
    required_memory: Optional[int] = None,
    required_gu: Optional[int] = None,
    required_gpu_memory: Optional[int] = None,
    device: str = None,
    properties: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
):
    frameworks_to_register = (
        [inference_frameworks]
        if isinstance(inference_frameworks, (str, InferenceFrameworkEnum))
        else inference_frameworks
    )

    deployment_configs = []
    for inference_framework in frameworks_to_register:
        inference_framework = (
            InferenceFrameworkEnum.from_string(inference_framework)
            if isinstance(inference_framework, str)
            else inference_framework
        )

        model_config = ModelDeploymentConfig(
            model_name=model_name,
            model_file=model_file,
            inference_framework_type=inference_framework,
            required_resource_files=required_resources,
            load_params=default_load_params,
            required_cpu=required_cpu,
            required_memory=required_memory,
            required_gu=required_gu,
            required_gpu_memory=required_gpu_memory,
            device=device,
            properties=properties,
            tags=tags,
        )
        deployment_configs.append(model_config)
    return deployment_configs


def register_model_deployments(
    model_name: str,
    model_file: str,
    inference_frameworks: Union[
        str, InferenceFrameworkEnum, List[InferenceFrameworkEnum]
    ],
    required_resources: List[Union[str, ResourceNameTemplate]] = None,
    default_load_params: Optional[Dict[str, Any]] = None,
    required_cpu: Optional[float] = None,
    required_memory: Optional[int] = None,
    required_gu: Optional[int] = None,
    required_gpu_memory: Optional[int] = None,
    device: str = None,
    properties: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
):
    """
    Register model deployments

    Notes
    -----
    - Alpha version for model deployments, all fields could be changed in the future.
    """
    model_name = model_name.lower()
    if not isinstance(inference_frameworks, (InferenceFrameworkEnum, List, str)):
        raise TypeError(
            f"inference_framework_type must be type InferenceFrameworkEnum or List[InferenceFrameworkEnum] but {type(inference_frameworks)}"
        )

    model_deployments = create_model_deployments(
        model_name=model_name,
        model_file=model_file,
        inference_frameworks=inference_frameworks,
        required_resources=required_resources,
        default_load_params=default_load_params,
        required_cpu=required_cpu,
        required_memory=required_memory,
        required_gu=required_gu,
        required_gpu_memory=required_gpu_memory,
        device=device,
        properties=properties,
        tags=tags,
    )

    for deployment in model_deployments:
        if model_name not in _registered_llm_models:
            _registered_llm_models[model_name] = {}
        if deployment.device not in _registered_llm_models[model_name]:
            _registered_llm_models[model_name][deployment.device] = {}

        existing = _registered_llm_models[model_name][deployment.device].get(
            deployment.inference_framework_type
        )

        if existing:
            raise ValueError(
                f"Model {model_name} with {deployment.inference_framework_type} on {deployment.device} already registered"
            )

        _registered_llm_models[model_name][deployment.device][
            deployment.inference_framework_type
        ] = deployment

    return model_deployments


def get_registered_model_config(
    model_name: str,
    expected_device: Optional[str] = None,
    expected_inference_framework: Optional[InferenceFrameworkEnum] = None,
) -> Optional[ModelDeploymentConfig]:
    model_name = model_name.lower()

    def _load_model_framework_config(framework_configs, expected_framework):
        if expected_framework is not None:
            return framework_configs[expected_framework]
        else:
            model_config = next(iter(framework_configs.values()))
            return model_config

    try:
        if expected_device is not None:
            model_framework_configs = _registered_llm_models[model_name][
                expected_device
            ]
            return _load_model_framework_config(
                model_framework_configs, expected_inference_framework
            )
        else:
            model_all_configs = _registered_llm_models[model_name]
            for device, model_framework_configs in model_all_configs.items():
                model_config = _load_model_framework_config(
                    model_framework_configs, expected_inference_framework
                )
                if model_config:
                    return model_config
    except KeyError:
        pass
    except StopIteration:
        pass

    return None


def clean_registered_model():
    _registered_llm_models.clear()
