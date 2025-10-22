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

import numpy as np

import maxframe.dataframe as md

from ..models.openai import (
    OpenAICompatibleLLM,
    OpenAICompatibleTextGenOp,
    OpenAICompatibleTextLLM,
)


def test_openai_compatible_llm_field_assignment():
    """Test OpenAICompatibleLLM field assignment."""
    llm = OpenAICompatibleLLM()
    llm.base_url = "https://api.openai.com/v1"
    llm.api_key = "test-key"
    llm.batch_size = 10
    llm.batch_timeout = 300

    assert llm.base_url == "https://api.openai.com/v1"
    assert llm.api_key == "test-key"
    assert llm.batch_size == 10
    assert llm.batch_timeout == 300


def test_openai_compatible_text_llm_initialization():
    """Test OpenAICompatibleTextLLM initialization."""
    model = OpenAICompatibleTextLLM(
        name="gpt-3.5-turbo", base_url="https://api.openai.com/v1", api_key="test-key"
    )

    assert model.name == "gpt-3.5-turbo"
    assert model.base_url == "https://api.openai.com/v1"
    assert model.api_key == "test-key"
    # Test inherited default values
    assert model.batch_size is None
    assert model.batch_timeout is None


def test_openai_compatible_text_llm_generate_method():
    """Test OpenAICompatibleTextLLM generate method."""
    model = OpenAICompatibleTextLLM(
        name="gpt-3.5-turbo", base_url="https://api.openai.com/v1", api_key="test-key"
    )

    # Create test data
    df = md.DataFrame({"query": ["Hello world"]})
    prompt_template = [{"role": "user", "content": "{query}"}]
    params = {"temperature": 0.7}

    # Test that generate method returns a DataFrame (result of operator execution)
    result = model.generate(
        data=df,
        prompt_template=prompt_template,
        simple_output=True,
        params=params,
        extra_param="test",
    )

    # Verify result is a DataFrame
    assert hasattr(result, "index_value")
    assert hasattr(result, "dtypes")

    # Verify the operator that was created has the correct parameters
    # We can access the operator through the result's op attribute
    op = result.op
    assert isinstance(op, OpenAICompatibleTextGenOp)
    assert op.model == model
    assert op.prompt_template == prompt_template
    assert op.simple_output is True
    assert op.params == params


def test_openai_compatible_text_llm_generate_with_defaults():
    """Test OpenAICompatibleTextLLM generate method with default parameters."""
    model = OpenAICompatibleTextLLM(
        name="gpt-3.5-turbo", base_url="https://api.openai.com/v1", api_key="test-key"
    )

    df = md.DataFrame({"query": ["Hello world"]})
    prompt_template = [{"role": "user", "content": "{query}"}]

    # Test that generate method returns a DataFrame (result of operator execution)
    result = model.generate(data=df, prompt_template=prompt_template)

    # Verify result is a DataFrame
    assert hasattr(result, "index_value")
    assert hasattr(result, "dtypes")

    # Verify the operator that was created has the correct parameters
    # We can access the operator through the result's op attribute
    op = result.op
    assert isinstance(op, OpenAICompatibleTextGenOp)
    assert op.model == model
    assert op.prompt_template == prompt_template
    assert op.simple_output is False
    assert op.params is None


def test_openai_compatible_text_generation_operator_default_values():
    """Test OpenAICompatibleTextGenOperator default field values."""
    op = OpenAICompatibleTextGenOp()
    assert op.simple_output is False


def test_openai_compatible_text_generation_operator_field_assignment():
    """Test OpenAICompatibleTextGenOperator field assignment."""
    op = OpenAICompatibleTextGenOp()
    op.simple_output = True
    assert op.simple_output is True


def test_openai_compatible_text_generation_operator_output_dtypes():
    """Test OpenAICompatibleTextGenOperator inherits correct output dtypes."""
    op = OpenAICompatibleTextGenOp()
    dtypes = op.get_output_dtypes()
    assert dtypes["response"] == np.dtype("O")
    assert dtypes["success"] == np.dtype("bool")


def test_openai_compatible_text_generation_operator_with_model():
    """Test OpenAICompatibleTextGenOperator with model parameter."""
    model = OpenAICompatibleTextLLM(
        name="gpt-3.5-turbo", base_url="https://api.openai.com/v1", api_key="test-key"
    )

    prompt_template = [{"role": "user", "content": "Hello"}]
    params = {"temperature": 0.5}

    op = OpenAICompatibleTextGenOp(
        model=model, prompt_template=prompt_template, simple_output=True, params=params
    )

    assert op.model == model
    assert op.prompt_template == prompt_template
    assert op.simple_output is True
    assert op.params == params


def test_openai_compatible_text_llm_inheritance():
    """Test that OpenAICompatibleTextLLM properly inherits from both parent classes."""
    model = OpenAICompatibleTextLLM(
        name="gpt-3.5-turbo", base_url="https://api.openai.com/v1", api_key="test-key"
    )

    # Test TextGenLLM inheritance - should have validate_params method
    assert hasattr(model, "validate_params")
    assert callable(getattr(model, "validate_params"))

    # Test OpenAICompatibleLLM inheritance - should have OpenAI-specific fields
    assert hasattr(model, "base_url")
    assert hasattr(model, "api_key")
    assert hasattr(model, "batch_size")
    assert hasattr(model, "batch_timeout")

    # Test that validate_params doesn't raise error with empty params
    model.validate_params({})


def test_openai_compatible_text_llm_validate_params():
    """Test OpenAICompatibleTextLLM validate_params method."""
    model = OpenAICompatibleTextLLM(
        name="gpt-3.5-turbo", base_url="https://api.openai.com/v1", api_key="test-key"
    )

    # Should not raise exception with valid params
    model.validate_params({"temperature": 0.7, "max_tokens": 100})

    # Should not raise exception with empty params
    model.validate_params({})

    # Should not raise exception with None params
    model.validate_params(None)
