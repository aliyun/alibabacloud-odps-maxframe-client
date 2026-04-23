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

import json

import pytest

from .....utils.odpsio import ReadODPSModel
from ...core import TASK_SENTENCE_EMBEDDING, TASK_TEXT_GENERATION
from ...deploy.config import REASONING_MODEL_KEY
from ..managed import ManagedTextEmbeddingModel, ManagedTextGenLLM
from ..odps import (
    _ODPS_PROP_CPU,
    _ODPS_PROP_DEVICE,
    _ODPS_PROP_FRAMEWORK,
    _ODPS_PROP_GU,
    _ODPS_PROP_IMAGE,
    _ODPS_PROP_MEMORY,
    _SCOPE_DEFAULT,
    _SCOPE_MAXFRAME,
    ODPS_PROP_ACCESS_KEY_ID,
    ODPS_PROP_ACCESS_KEY_SECRET,
    ODPS_PROP_INFERENCE_PARAMETERS,
    ODPS_PROP_REASONING_MODE,
    ODPS_PROP_ROLE_ARN,
    ODPS_PROP_SOURCE_TYPE_KEY,
    ODPS_PROP_TYPE_KEY,
    ODPS_PROP_VERSION_KEY,
    ODPSLLM,
)


def _make_op(
    model_name="proj.default.test",
    location="/path",
    format="LLM",
    model_version=None,
    options=None,
    tasks=None,
    inference_parameters=None,
    source_type=None,
):
    """Create a ReadODPSModel op with metadata fields pre-populated."""
    return ReadODPSModel(
        model_name=model_name,
        model_version=model_version,
        location=location,
        format=format,
        options=options or {},
        tasks=tasks or [],
        inference_parameters=inference_parameters or {},
        source_type=source_type,
    )


def _llm_scope_options(
    *,
    scope_key=_SCOPE_MAXFRAME,
    gu=1,
    framework="VLLM_SERVE:TEXT",
    device="cuda",
    **extra,
):
    payload = {
        _ODPS_PROP_GU: gu,
        _ODPS_PROP_FRAMEWORK: framework,
        _ODPS_PROP_DEVICE: device,
    }
    payload.update(extra)
    return {scope_key: payload}


# Test 1: Main build flow
def test_build_model_uses_scope_precedence_and_properties():
    # Case 1: OSS path, camelCase roleArn in options -> normalized to snake_case
    oss_path = "oss://oss-cn-hangzhou.aliyuncs.com/bucket/model/"
    op = _make_op(
        location=oss_path,
        options={
            _SCOPE_MAXFRAME: {
                _ODPS_PROP_GU: 8,
                _ODPS_PROP_FRAMEWORK: "VLLM_SERVE:TEXT",
                _ODPS_PROP_DEVICE: "cuda",
                _ODPS_PROP_CPU: 4,
            },
            _SCOPE_DEFAULT: {
                _ODPS_PROP_GU: 2,
                _ODPS_PROP_FRAMEWORK: "OTHER",
                _ODPS_PROP_DEVICE: "cpu",
                _ODPS_PROP_CPU: 1,
            },
            "roleArn": "arn:test:role",
        },
        tasks=[TASK_TEXT_GENERATION],
        inference_parameters={_ODPS_PROP_GU: 1, _ODPS_PROP_MEMORY: 999},
    )
    result = ODPSLLM._build_odps_source_model(op)
    assert isinstance(result, ManagedTextGenLLM)
    assert result.deploy_config.required_gu == 8
    assert result.deploy_config.required_cpu == 4
    assert result.deploy_config.device == "cuda"
    assert result.deploy_config.model_file == oss_path
    assert result.deploy_config.properties[ODPS_PROP_ROLE_ARN] == "arn:test:role"


def test_build_model_from_default_scope():
    # Case 2: Local path -> model_file set directly
    op = _make_op(
        location="/local/path",
        options={
            _SCOPE_DEFAULT: {
                _ODPS_PROP_GU: 4,
                _ODPS_PROP_FRAMEWORK: "VLLM_SERVE:TEXT",
                _ODPS_PROP_DEVICE: "cuda",
                _ODPS_PROP_MEMORY: 16,
            }
        },
        tasks=[TASK_TEXT_GENERATION],
    )
    result = ODPSLLM._build_odps_source_model(op)
    assert isinstance(result, ManagedTextGenLLM)
    assert result.deploy_config.model_file == "/local/path"
    assert result.deploy_config.required_memory == 16


@pytest.mark.parametrize(
    "tasks,options,inference_parameters,expected_class,expected_image",
    [
        (  # image string from inference metadata
            [TASK_TEXT_GENERATION],
            {},
            {
                _ODPS_PROP_GU: 2,
                _ODPS_PROP_FRAMEWORK: "VLLM_SERVE:TEXT",
                _ODPS_PROP_DEVICE: "cuda",
                _ODPS_PROP_IMAGE: "my-image:v1",
            },
            ManagedTextGenLLM,
            {"name": "my-image:v1"},
        ),
        (  # image dict from options metadata
            [TASK_TEXT_GENERATION],
            {
                _SCOPE_MAXFRAME: {
                    _ODPS_PROP_GU: 1,
                    _ODPS_PROP_FRAMEWORK: "VLLM_SERVE:TEXT",
                    _ODPS_PROP_DEVICE: "cuda",
                    _ODPS_PROP_IMAGE: {"name": "img:v1", "tag": "latest"},
                }
            },
            {},
            ManagedTextGenLLM,
            {"name": "img:v1", "tag": "latest"},
        ),
        (  # embedding model dispatch
            [TASK_SENTENCE_EMBEDDING],
            _llm_scope_options(),
            {},
            ManagedTextEmbeddingModel,
            None,
        ),
        (  # both tasks: text-generation takes precedence
            [TASK_TEXT_GENERATION, TASK_SENTENCE_EMBEDDING],
            _llm_scope_options(),
            {},
            ManagedTextGenLLM,
            None,
        ),
    ],
    ids=[
        "image-from-inference-metadata",
        "image-from-options-metadata",
        "embedding-dispatch",
        "text-generation-priority",
    ],
)
def test_model_class_dispatch_and_image_parsing(
    tasks, options, inference_parameters, expected_class, expected_image
):
    result = ODPSLLM._build_odps_source_model(
        _make_op(
            model_name="proj.default.embed",
            options=options,
            tasks=tasks,
            inference_parameters=inference_parameters,
        )
    )
    assert isinstance(result, expected_class)
    if expected_image is not None:
        assert result.deploy_config.image == expected_image


@pytest.mark.parametrize(
    "format,tasks",
    [
        ("BOOSTED_TREE_CLASSIFIER", [TASK_TEXT_GENERATION]),
        ("MLLM", [TASK_TEXT_GENERATION]),
        ("LLM", ["some-unknown-task"]),
        ("LLM", []),
    ],
)
def test_unsupported_or_unmatched_models_return_none(format, tasks):
    result = ODPSLLM._build_odps_source_model(
        _make_op(format=format, tasks=tasks, options=_llm_scope_options())
    )
    assert result is None


# Test 2: Validation errors
def test_validation_and_errors():
    # Empty model_name -> raises
    op = _make_op(
        model_name="",
        options={_SCOPE_MAXFRAME: {_ODPS_PROP_GU: 1, _ODPS_PROP_DEVICE: "cuda"}},
        tasks=[TASK_TEXT_GENERATION],
    )
    with pytest.raises(ValueError, match="model_name is required"):
        ODPSLLM._build_odps_source_model(op)

    # Invalid image type -> raises
    op = _make_op(
        options={
            _SCOPE_MAXFRAME: {
                **_llm_scope_options()[_SCOPE_MAXFRAME],
                _ODPS_PROP_IMAGE: 123,
            }
        },
        tasks=[TASK_TEXT_GENERATION],
    )
    with pytest.raises(ValueError, match="image must be str or dict"):
        ODPSLLM._build_odps_source_model(op)

    # Invalid scope payload -> raises
    op = _make_op(
        options={_SCOPE_MAXFRAME: "not-a-json-dict"},
        tasks=[TASK_TEXT_GENERATION],
    )
    with pytest.raises(ValueError, match="options\\['maxframe'\\] is not valid JSON"):
        ODPSLLM._build_odps_source_model(op)


@pytest.mark.parametrize("scope_key", [_SCOPE_MAXFRAME, _SCOPE_DEFAULT])
def test_build_and_validate_config(scope_key):
    # Case 1: framework absent in metadata, still forms a valid config
    op = _make_op(
        options={scope_key: {_ODPS_PROP_DEVICE: "cuda", _ODPS_PROP_GU: 1}},
        tasks=[TASK_TEXT_GENERATION],
    )
    result = ODPSLLM._build_odps_source_model(op)
    assert isinstance(result, ManagedTextGenLLM)
    result.deploy_config.check_validity()

    # Case 2: explicit metadata values map into config and pass validity checks
    op = _make_op(
        options={
            scope_key: {_ODPS_PROP_GU: 1, _ODPS_PROP_FRAMEWORK: "VLLM_SERVE:TEXT"}
        },
        tasks=[TASK_TEXT_GENERATION],
    )
    result = ODPSLLM._build_odps_source_model(op)
    assert isinstance(result, ManagedTextGenLLM)
    assert result.deploy_config.required_gu == 1
    result.deploy_config.check_validity()


# Test 3: load_params excludes reserved keys
def test_load_params_excludes_reserved_keys():
    op = _make_op(
        options={
            _SCOPE_MAXFRAME: {
                _ODPS_PROP_GU: 2,
                _ODPS_PROP_FRAMEWORK: "VLLM_SERVE:TEXT",
                _ODPS_PROP_DEVICE: "y",
                "custom_param": "value",
            }
        },
        tasks=[TASK_TEXT_GENERATION],
    )
    result = ODPSLLM._build_odps_source_model(op)
    params = result.deploy_config.load_params
    assert "custom_param" in params
    assert _ODPS_PROP_GU not in params
    assert ODPS_PROP_INFERENCE_PARAMETERS not in params


def test_load_params_supports_json_scope_strings():
    op = _make_op(
        options={
            _SCOPE_MAXFRAME: json.dumps({"a": 1}),
            "c": 3,
        },
        inference_parameters={
            _SCOPE_DEFAULT: json.dumps({"b": 2}),
        },
        tasks=[TASK_TEXT_GENERATION],
    )
    result = ODPSLLM._build_odps_source_model(op)
    params = result.deploy_config.load_params
    assert params["a"] == 1
    assert params["b"] == 2
    assert params["c"] == 3


# Test 4: Properties and metadata
def test_properties_and_metadata():
    # Case 1: Full model name with version
    op = _make_op(
        model_name="proj.schema.mymodel",
        location=None,
        model_version="v1.0",
        options=_llm_scope_options(),
        tasks=[TASK_TEXT_GENERATION],
        source_type="IMPORT",
        inference_parameters={ODPS_PROP_INFERENCE_PARAMETERS: {"param1": "value1"}},
    )
    result = ODPSLLM._build_odps_source_model(op)
    props = result.deploy_config.properties
    assert props[ODPS_PROP_SOURCE_TYPE_KEY] == "IMPORT"
    assert props[ODPS_PROP_VERSION_KEY] == "v1.0"
    assert props[ODPS_PROP_TYPE_KEY] == "LLM"
    assert result.deploy_config.inference_parameters == {"param1": "value1"}
    assert result.deploy_config.model_name == "proj.schema.mymodel"

    # Case 2: No version -> MODEL_VERSION_KEY absent
    op = _make_op(
        model_name="proj.default.mymodel",
        options=_llm_scope_options(),
        tasks=[TASK_TEXT_GENERATION],
    )
    result = ODPSLLM._build_odps_source_model(op)
    assert ODPS_PROP_VERSION_KEY not in result.deploy_config.properties

    # Case 3: OSS credentials in camelCase -> extracted as snake_case
    oss_path = "oss://endpoint/bucket/model/"
    op = _make_op(
        location=oss_path,
        options={
            **_llm_scope_options(),
            # ODPS metadata may use camelCase
            "accessKeyId": "myak",
            "accessKeySecret": "mysk",
        },
        tasks=[TASK_TEXT_GENERATION],
    )
    result = ODPSLLM._build_odps_source_model(op)
    props = result.deploy_config.properties
    assert result.deploy_config.model_file == oss_path
    # Properties always use snake_case keys
    assert props[ODPS_PROP_ACCESS_KEY_ID] == "myak"
    assert props[ODPS_PROP_ACCESS_KEY_SECRET] == "mysk"


# Test 5: Parameter type coercion and reasoning_mode handling
@pytest.mark.parametrize(
    "scope_config,expected_reasoning,expected_gu,expected_cpu",
    [
        # reasoning_mode="false" should be False (not True like bool("false"))
        (
            {
                _ODPS_PROP_GU: 1,
                _ODPS_PROP_FRAMEWORK: "VLLM_SERVE:TEXT",
                _ODPS_PROP_DEVICE: "cuda",
                ODPS_PROP_REASONING_MODE: "false",
            },
            False,
            1,
            None,
        ),
        # reasoning_mode="true" -> True
        (
            {
                _ODPS_PROP_GU: 1,
                _ODPS_PROP_FRAMEWORK: "VLLM_SERVE:TEXT",
                _ODPS_PROP_DEVICE: "cuda",
                ODPS_PROP_REASONING_MODE: "true",
            },
            True,
            1,
            None,
        ),
        # reasoning_mode=False needs is not None check to be stored
        (
            {
                _ODPS_PROP_GU: 1,
                _ODPS_PROP_FRAMEWORK: "VLLM_SERVE:TEXT",
                _ODPS_PROP_DEVICE: "cuda",
                ODPS_PROP_REASONING_MODE: False,
            },
            False,
            1,
            None,
        ),
        # reasoning_mode=0 needs is not None check to be stored as False
        (
            {
                _ODPS_PROP_GU: 1,
                _ODPS_PROP_FRAMEWORK: "VLLM_SERVE:TEXT",
                _ODPS_PROP_DEVICE: "cuda",
                ODPS_PROP_REASONING_MODE: 0,
            },
            False,
            1,
            None,
        ),
        # cpu/gu string numbers converted to int
        (
            {
                _ODPS_PROP_GU: "8",
                _ODPS_PROP_FRAMEWORK: "VLLM_SERVE:TEXT",
                _ODPS_PROP_DEVICE: "cuda",
                _ODPS_PROP_CPU: "4",
            },
            None,
            8,
            4,
        ),
    ],
    ids=[
        "reasoning-mode-string-false",
        "reasoning-mode-string-true",
        "reasoning-mode-bool-False",
        "reasoning-mode-int-0",
        "cpu-gu-string-to-int",
    ],
)
def test_parameter_type_coercion(
    scope_config, expected_reasoning, expected_gu, expected_cpu
):
    """Test parameter type coercion: reasoning_mode strings, cpu/gu string->int."""
    op = _make_op(
        options={_SCOPE_MAXFRAME: scope_config},
        tasks=[TASK_TEXT_GENERATION],
    )
    result = ODPSLLM._build_odps_source_model(op)
    assert isinstance(result, ManagedTextGenLLM)
    assert result.deploy_config.required_gu == expected_gu
    assert result.deploy_config.required_cpu == expected_cpu
    if expected_reasoning is not None:
        assert (
            result.deploy_config.properties.get(REASONING_MODEL_KEY)
            == expected_reasoning
        )
    else:
        assert REASONING_MODEL_KEY not in result.deploy_config.properties
