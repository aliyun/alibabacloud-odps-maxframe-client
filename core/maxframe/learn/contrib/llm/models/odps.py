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

"""
Factory for building LLM models from ODPS Model Catalog.

This module provides ODPSLLM which constructs ManagedTextGenLLM or
ManagedTextEmbeddingModel instances from ODPS Model metadata.
"""

import json
from typing import Any, Dict, List, Optional, Set

from ....utils.odpsio import ODPSModelMixin, ReadODPSModel, register_odps_model
from ..core import TASK_SENTENCE_EMBEDDING, TASK_TEXT_GENERATION
from ..deploy.config import ModelDeploymentConfig
from ..deploy.framework import InferenceFrameworkEnum
from .managed import ManagedTextEmbeddingModel, ManagedTextGenLLM

_SCOPE_MAXFRAME = "maxframe"
_SCOPE_DEFAULT = "*"

# keys for ODPS model metadata
_ODPS_PROP_CPU = "cpu"
_ODPS_PROP_MEMORY = "memory"
_ODPS_PROP_GU = "gu"
_ODPS_PROP_GPU = "gpu"
_ODPS_PROP_FRAMEWORK = "framework"
_ODPS_PROP_DEVICE = "device"
_ODPS_PROP_IMAGE = "image"

ODPS_PROP_VERSION_KEY = "model_version"
ODPS_PROP_SOURCE_TYPE_KEY = "model_source_type"
ODPS_PROP_TYPE_KEY = "model_type"
ODPS_PROP_ROLE_ARN = "role_arn"
ODPS_PROP_ACCESS_KEY_ID = "access_key_id"
ODPS_PROP_ACCESS_KEY_SECRET = "access_key_secret"
ODPS_PROP_INFERENCE_PARAMETERS = "inference_parameters"


def _snake_to_camel(snake: str) -> str:
    """Convert snake_case to camelCase. Single words are unchanged."""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


_RESERVED_KEYS = {
    _ODPS_PROP_FRAMEWORK,
    _ODPS_PROP_DEVICE,
    _ODPS_PROP_GU,
    _ODPS_PROP_GPU,
    _ODPS_PROP_CPU,
    _ODPS_PROP_MEMORY,
    _ODPS_PROP_IMAGE,
    ODPS_PROP_ROLE_ARN,
    ODPS_PROP_ACCESS_KEY_ID,
    ODPS_PROP_ACCESS_KEY_SECRET,
    ODPS_PROP_INFERENCE_PARAMETERS,
}


class ODPSModelType:
    """Constants for model types in ODPS Model Catalog."""

    LLM = "LLM"
    MLLM = "MLLM"


@register_odps_model
class ODPSLLM(ODPSModelMixin):
    """
    Factory class for building LLM models from ODPS Model Catalog.

    This class is registered with the ODPS model factory and handles
    LLM model types. When a model of this type is loaded via
    ``read_odps_model()``, this class constructs the appropriate
    ManagedTextGenLLM or ManagedTextEmbeddingModel instance.
    """

    @staticmethod
    def _parse_scope_dict(raw: Any, scope_name: str) -> Dict[str, Any]:
        """Parse a scope value that may be a dict or a JSON string."""
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{scope_name} is not valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{scope_name} must be a dict or JSON dict string")
            return parsed
        raise ValueError(f"{scope_name} must be a dict or JSON dict string")

    @classmethod
    def _build_scoped_sources(
        cls, options: Dict[str, Any], inference_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build a list of dicts to search in precedence order.

        Order: options(top-level) -> options[maxframe] -> options[*]
             -> inference_params(top-level) -> inference_params[maxframe] -> inference_params[*]
        """
        options = options or {}
        inference_params = inference_params or {}

        sources: List[Dict[str, Any]] = []
        for src, label in [
            (options, "options"),
            (inference_params, "inference_parameters"),
        ]:
            sources.append(
                {
                    k: v
                    for k, v in src.items()
                    if k not in (_SCOPE_MAXFRAME, _SCOPE_DEFAULT)
                }
            )
            sources.append(
                cls._parse_scope_dict(src.get(_SCOPE_MAXFRAME), f"{label}['maxframe']")
            )
            sources.append(
                cls._parse_scope_dict(src.get(_SCOPE_DEFAULT), f"{label}['*']")
            )
        return sources

    @staticmethod
    def _get_and_pop_options(
        sources: List[Dict[str, Any]],
        keys: Set[str],
    ) -> Dict[str, Any]:
        """Get specified options from sources by popping them.

        For each key in ``keys``, tries camelCase first then snake_case
        in every source, keeping the first non-None value found.
        Matched keys are removed from sources so they won't be picked
        up again by ``_collect_load_params``.
        """
        result: Dict[str, Any] = {}
        for field in keys:
            camel = _snake_to_camel(field)
            for source in sources:
                # always pop both variants from every source
                val_camel = source.pop(camel, None)
                val_snake = source.pop(field, None)
                if field not in result:
                    picked = val_camel if val_camel is not None else val_snake
                    if picked is not None:
                        result[field] = picked
        return result

    @staticmethod
    def _collect_load_params(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect remaining params from sources (first-seen wins), skipping None values."""
        load_params: Dict[str, Any] = {}
        for source in sources:
            for key, value in source.items():
                if value is None:
                    continue
                if key not in load_params:
                    load_params[key] = value
        return load_params

    @staticmethod
    def _parse_image(raw_image: Any) -> Optional[Dict[str, Any]]:
        """Parse the image field: str -> {"name": str}, dict -> as-is, None -> None."""
        if raw_image is None:
            return None
        if isinstance(raw_image, str):
            return {"name": raw_image}
        if isinstance(raw_image, dict):
            return raw_image
        raise ValueError(f"image must be str or dict, got {type(raw_image).__name__}")

    @classmethod
    def _determine_model_class(cls, tasks: List[str], model_type: str):
        """Return the managed model class, or None if not matched."""
        if model_type != ODPSModelType.LLM:
            return None

        if TASK_TEXT_GENERATION in tasks:
            return ManagedTextGenLLM
        if TASK_SENTENCE_EMBEDDING in tasks:
            return ManagedTextEmbeddingModel
        return None

    @classmethod
    def _build_deployment_config(
        cls,
        model_name: str,
        model_version: Optional[str],
        model_type: str,
        location: str,
        options: Dict[str, Any],
        odps_inference_parameters: Dict[str, Any],
        tasks: List[str],
        source_type: Optional[str],
    ) -> ModelDeploymentConfig:
        """Build a ModelDeploymentConfig from ODPS metadata."""
        sources = cls._build_scoped_sources(options, odps_inference_parameters)

        reserved = cls._get_and_pop_options(sources, _RESERVED_KEYS)
        model_inference_parameters = reserved.get(ODPS_PROP_INFERENCE_PARAMETERS) or {}

        framework = InferenceFrameworkEnum.from_string(
            reserved.get(_ODPS_PROP_FRAMEWORK)
        )

        device = reserved.get(_ODPS_PROP_DEVICE)
        required_gu = reserved.get(_ODPS_PROP_GU) or reserved.get(_ODPS_PROP_GPU)
        required_cpu = reserved.get(_ODPS_PROP_CPU)

        if not required_cpu and not required_gu:
            required_cpu = 1

        required_memory = reserved.get(_ODPS_PROP_MEMORY)

        image = cls._parse_image(reserved.get(_ODPS_PROP_IMAGE))

        # location is always model_file; server-side resolves OSS mounts
        model_file = location or ""

        load_params = cls._collect_load_params(sources)
        tags = list(tasks)

        # Properties for server-side processing
        properties: Dict[str, Any] = {
            ODPS_PROP_SOURCE_TYPE_KEY: source_type,
            ODPS_PROP_TYPE_KEY: model_type,
        }
        if model_version:
            properties[ODPS_PROP_VERSION_KEY] = model_version
        role_arn = reserved.get(ODPS_PROP_ROLE_ARN)
        if role_arn:
            properties[ODPS_PROP_ROLE_ARN] = role_arn
        access_key_id = reserved.get(ODPS_PROP_ACCESS_KEY_ID)
        access_key_secret = reserved.get(ODPS_PROP_ACCESS_KEY_SECRET)
        if access_key_id:
            properties[ODPS_PROP_ACCESS_KEY_ID] = access_key_id
        if access_key_secret:
            properties[ODPS_PROP_ACCESS_KEY_SECRET] = access_key_secret

        return ModelDeploymentConfig(
            model_name=model_name,
            model_file=model_file,
            inference_framework_type=framework,
            required_gu=required_gu,
            required_cpu=required_cpu,
            required_memory=required_memory,
            device=device,
            load_params=load_params,
            inference_parameters=model_inference_parameters,
            tags=tags,
            properties=properties,
            image=image,
        )

    @classmethod
    def _build_odps_source_model(cls, op: ReadODPSModel) -> Any:
        """Build a managed LLM from ODPS metadata, or None if unsupported."""
        if op.format != ODPSModelType.LLM:
            return None

        model_name = op.model_name
        if not model_name:
            raise ValueError("model_name is required")

        options = op.options or {}
        inference_parameters = op.inference_parameters or {}
        tasks = op.tasks or []

        model_class = cls._determine_model_class(tasks, op.format)
        if model_class is None:
            return None

        deploy_config = cls._build_deployment_config(
            model_name=model_name,
            model_version=op.model_version,
            model_type=op.format,
            location=op.location,
            options=options,
            odps_inference_parameters=inference_parameters,
            tasks=tasks,
            source_type=op.source_type,
        )

        return model_class(name=model_name, deploy_config=deploy_config)

    @classmethod
    def is_odps_remote_model(cls, model_config: ModelDeploymentConfig) -> bool:
        source_type = model_config.properties.get(ODPS_PROP_SOURCE_TYPE_KEY, None)
        return source_type == "REMOTE"
