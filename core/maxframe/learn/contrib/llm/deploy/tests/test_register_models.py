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
import os.path
from typing import List

import pytest

from ..core import (
    ResourceNameTemplate,
    _registered_llm_models,
    clean_registered_model,
    get_registered_model_config,
    register_model_deployments,
)
from ..framework import InferenceFrameworkEnum
from ..loader import (
    DeploymentConfigBase,
    DeploymentConfigResourceRequirement,
    FrameworkDeploymentConfig,
    ModelDeploymentConfig,
    _model_config_to_json,
    register_models_from_json_file,
)

current_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def qwen2_5_0_5b_model():
    model_name = "Qwen2.5-0.5B-instruct"
    model_file_resource = ResourceNameTemplate(
        project="bigdata_public_dataset",
        schema="data_plus_ai_{region_id}",
        name="Qwen2.5-0.5B-Instruct.Q8_0.gguf",
    )

    common_load_params = {
        "chat_format": "qwen",
        "flash_attn": True,
        "cache-type-k": "q8_0",
        "cache-type-v": "q8_0",
        "defrag-thold": "1",
    }
    common_config = DeploymentConfigBase.from_json(
        {
            "properties": {"family": "Qwen2.5", "type": "instruct"},
            "load_params": common_load_params,
        }
    )

    load_params = {"n_threads": 2, "n_batch": 1024, "n_ctx": 32768, "verbose": False}
    calc_resources = DeploymentConfigResourceRequirement(cpu=2, memory=8)
    deployment_llama_cpp_python = FrameworkDeploymentConfig(
        model_file="Qwen2.5-0.5B-Instruct.Q8_0.gguf",
        framework=InferenceFrameworkEnum.LLAMA_CPP_PYTHON_TEXT.value,
        device="cpu",
        model_file_resources=[model_file_resource],
        load_params=load_params,
        resource_requirements=calc_resources,
    )
    deployment_llama_server = FrameworkDeploymentConfig(
        model_file="Qwen2.5-0.5B-Instruct.Q8_0.gguf",
        framework=InferenceFrameworkEnum.LLAMA_CPP_SERVE_TEXT.value,
        device="cpu",
        model_file_resources=[model_file_resource],
        load_params=load_params,
        resource_requirements=calc_resources,
    )
    deployments = [deployment_llama_cpp_python, deployment_llama_server]

    return ModelDeploymentConfig(
        model_name=model_name,
        tags=["tag"],
        common_config=common_config,
        deployments=deployments,
    )


@pytest.fixture
def qwen3_8b_model():
    json_config = """
        {
        "model_name": "Qwen3-0.6B",
        "model_identifier": "qwen3-0.6b",
        "common_config": {
        "properties": {
            "family": "Qwen3",
            "reasoning_model": true
        },
        "load_params": {
            "flash_attn": true,
            "cache-type-k": "q8_0",
            "cache-type-v": "q8_0",
            "defrag-thold": "1",
            "n_threads": 2,
            "n_batch": 1024,
            "n_ctx": 32768,
            "verbose": false
        },
        "model_file": "Qwen3-0.6B-Q8_0.gguf",
        "model_file_resources": [
            {
                "project": "bigdata_public_dataset",
                "schema": "data_plus_ai_{region_id}",
                "name": "Qwen3-0.6B-Q8_0.gguf"
            }
        ],
        "resource_requirements": {
            "cpu": 2,
            "memory": 8
            }
        },
        "deployments": [
        {
            "framework": "LLAMA_CPP_SERVE:TEXT",
            "device": "cpu",
            "load_params": {},
            "resource_requirements": {},
            "properties": {}
        },
        {
            "framework": "LLAMA_CPP_PYTHON:TEXT",
            "device": "cpu",
            "load_params": {},
            "resource_requirements": {
            "cpu": 2,
            "memory": 8
            },
            "properties": {}
        }
        ],
        "tags": []
    }
    """
    model = ModelDeploymentConfig.from_json(json.loads(json_config))
    return model


@pytest.fixture
def models(qwen2_5_0_5b_model, qwen3_8b_model):
    model_list = [qwen2_5_0_5b_model, qwen3_8b_model]
    import uuid

    test_json_path = os.path.join(current_dir, f"./test_models_{uuid.uuid4()}.json")
    try:
        with open(test_json_path, "w") as f:
            f.write(_model_config_to_json(model_list))

        yield test_json_path, model_list
    finally:
        clean_registered_model()
        if os.path.exists(test_json_path):
            os.remove(test_json_path)


def test_serialize_and_deserialize(qwen2_5_0_5b_model):
    serialized = qwen2_5_0_5b_model.to_json()
    deserialized = ModelDeploymentConfig.from_json(serialized)
    assert serialized == deserialized.to_json()


def test_register_models():
    clean_registered_model()
    register_model_deployments(
        model_name="mock_model",
        model_file="DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
        inference_frameworks=InferenceFrameworkEnum.LLAMA_CPP_PYTHON_TEXT,
        required_resources=[
            ResourceNameTemplate(
                project="test",
                schema=None,
                name="DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
            ),
        ],
        default_load_params={
            "n_threads": 3,
            "n_threads_batch": 512,
            "n_batch": 512,
            "flash_attn": True,
            "verbose": True,
            "n_ctx": 32 * 1024,
        },
        required_cpu=2,
        required_memory=8,
        device="cpu",
        properties={"reasoning_model": True},
    )

    cpu_model_config = get_registered_model_config(
        "mock_model",
        expected_device="cpu",
        expected_inference_framework=InferenceFrameworkEnum.LLAMA_CPP_PYTHON_TEXT,
    )
    assert cpu_model_config is not None
    assert cpu_model_config is get_registered_model_config(
        "mock_model", expected_device="cpu"
    )
    general_config = get_registered_model_config("mock_model")
    assert cpu_model_config.model_name == general_config.model_name
    assert cpu_model_config.device == general_config.device
    assert cpu_model_config.device == "cpu"
    assert cpu_model_config.required_memory == 8
    assert cpu_model_config.required_cpu == 2
    assert cpu_model_config.required_resource_files == [
        ResourceNameTemplate(
            project="test", schema=None, name="DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
        )
    ]
    assert cpu_model_config.load_params.get("flash_attn") is True

    register_model_deployments(
        model_name="mock_model",
        model_file="DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
        inference_frameworks=InferenceFrameworkEnum.LLAMA_CPP_PYTHON_TEXT,
        required_resources=[],
        default_load_params={},
        required_cpu=2,
        required_memory=8,
        device="gpu",
        properties={"reasoning_model": True},
    )
    gpu_model_config = get_registered_model_config("mock_model", expected_device="gpu")
    cpu_model_config_again = get_registered_model_config(
        "mock_model", expected_device="cpu"
    )
    assert cpu_model_config.model_name == cpu_model_config_again.model_name
    assert cpu_model_config.device == cpu_model_config_again.device
    assert cpu_model_config.device != gpu_model_config.device

    with pytest.raises(ValueError):
        register_model_deployments(
            model_name="mock_model",
            model_file="DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
            inference_frameworks=InferenceFrameworkEnum.LLAMA_CPP_PYTHON_TEXT,
            required_resources=[],
            default_load_params={},
            required_cpu=2,
            required_memory=8,
            device="gpu",
            properties={"reasoning_model": True},
        )


def test_model_config_register(models):
    clean_registered_model()
    import os

    test_json_path, model_list = models
    model_list: List[ModelDeploymentConfig]

    with pytest.raises(FileNotFoundError):
        register_models_from_json_file(
            os.path.join(current_dir, "test_models_not_exist.json")
        )

    test_config_path = os.path.join(current_dir, test_json_path)
    register_models_from_json_file(test_config_path)

    deployments = []
    for model in model_list:
        deployments.extend(model.deployments)

    assert len(_registered_llm_models) == len(model_list)

    for model in model_list:
        model_name = model.model_name.lower()
        assert model_name in _registered_llm_models.keys()

        for deployments in model.deployments:
            assert deployments.device in _registered_llm_models[model_name]
            assert (
                InferenceFrameworkEnum.from_string(deployments.framework)
                in _registered_llm_models[model_name][deployments.device]
            )


def test_resource_name_template():
    """Test ResourceNameTemplate functionality."""
    # Test with single variable
    template = ResourceNameTemplate(
        project="test_project", schema="schema_{region_id}", name="resource_{region_id}"
    )

    result = template.build_full_resource_name({"region_id": "cn-hangzhou"})
    assert "cn_hangzhou" in result

    # Test with multiple variables and hyphen replacement
    template2 = ResourceNameTemplate(
        project="my_project",
        schema="schema_{region_id}_{env}",
        name="resource_{region_id}_{version}",
    )

    result2 = template2.build_full_resource_name(
        {"region_id": "us-west-1", "env": "prod-test", "version": "v1-2-3"}
    )
    assert "us_west_1" in result2
    assert "prod_test" in result2
    assert "v1_2_3" in result2

    # Test equality and hash
    template3 = ResourceNameTemplate("proj", "schema", "name")
    template4 = ResourceNameTemplate("proj", "schema", "name")
    template5 = ResourceNameTemplate("proj2", "schema", "name")

    assert template3 == template4
    assert template3 != template5
    assert hash(template3) == hash(template4)
    assert hash(template3) != hash(template5)


@pytest.mark.parametrize(
    "input_string,expected_enum",
    [
        ("LLAMA_CPP_PYTHON:TEXT", InferenceFrameworkEnum.LLAMA_CPP_PYTHON_TEXT),
        ("LLAMA_CPP_SERVE:TEXT", InferenceFrameworkEnum.LLAMA_CPP_SERVE_TEXT),
        ("DASH_SCOPE:TEXT", InferenceFrameworkEnum.DASH_SCOPE_TEXT),
        ("DASH_SCOPE:MULTIMODAL", InferenceFrameworkEnum.DASH_SCOPE_MULTIMODAL),
        ("VLLM_SERVE:TEXT", InferenceFrameworkEnum.VLLM_SERVE_TEXT),
        ("OPENAI_REMOTE:TEXT", InferenceFrameworkEnum.OPENAI_REMOTE_TEXT),
        ("OTHER", InferenceFrameworkEnum.OTHER),
        (
            InferenceFrameworkEnum.DASH_SCOPE_TEXT,
            InferenceFrameworkEnum.DASH_SCOPE_TEXT,
        ),
    ],
)
def test_inference_framework_enum_success(input_string, expected_enum):
    result = InferenceFrameworkEnum.from_string(input_string)
    assert result == expected_enum


@pytest.mark.parametrize(
    "input_string",
    [
        "INVALID_FRAMEWORK",
        "",
        "llama_cpp_python:text",
        "LLAMA_CPP_PYTHON:Text",
        "UNKNOWN:FRAMEWORK",
    ],
)
def test_inference_framework_enum_failure(input_string):
    """Test that invalid inputs raise appropriate exceptions."""
    with pytest.raises(ValueError):
        InferenceFrameworkEnum.from_string(input_string)
