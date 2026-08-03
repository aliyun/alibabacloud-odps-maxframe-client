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

from typing import Any, Dict, Optional

import numpy as np

from maxframe import dataframe as md
from maxframe.learn.contrib.llm.core import (  # noqa: F401
    IMAGE_BASE64_CONTENT,
    IMAGE_BINARY_CONTENT,
    IMAGE_CONTENT_PART,
    IMAGE_URL_CONTENT,
    LLM,
    TEXT_CONTENT_PART,
    ContentPart,
    ImageContentType,
    LLMTaskOperator,
    LLMTextGenOperator,
    validate_llm_input_data,
)
from maxframe.serialization.serializables import BoolField
from maxframe.serialization.serializables.field import AnyField


class LLMMultiModalGenerationOp(LLMTextGenOperator):
    pass


class MultiModalEmbeddingOp(LLMTaskOperator):
    # Multimodal embedding request template.
    input = AnyField("input", default=None)
    simple_output = BoolField("simple_output", default=False)

    def get_output_dtypes(self) -> Dict[str, np.dtype]:
        return {"response": md.dtype("string"), "success": np.dtype("bool")}


class MultiModalModel(LLM):
    pass


class MultiModalGenLLM(MultiModalModel):
    def generate(
        self,
        data,
        messages=None,
        prompt_template=None,
        simple_output: bool = False,
        params: Optional[Dict[str, Any]] = None,
        **kw,
    ):
        raise NotImplementedError


class MultiModalEmbeddingModel(MultiModalModel):
    def embed(
        self,
        data,
        input,
        simple_output: bool = False,
        params: Optional[Dict[str, Any]] = None,
        **kw,
    ):
        raise NotImplementedError


def generate(
    data,
    model: MultiModalGenLLM,
    messages=None,
    prompt_template=None,
    simple_output: bool = False,
    params: Optional[Dict[str, Any]] = None,
    **kw,
):
    """
    Generate text with a multimodal LLM from MaxFrame data.

    Parameters
    ----------
    data : DataFrame or Series
        Input data used to render one request per row. Template placeholders
        reference columns by name, for example ``"{image_url}"``.
    model : MultiModalGenLLM
        Multimodal generation model instance.
    messages : list of dict, optional
        Chat messages template. This is an alias of ``prompt_template`` and
        takes precedence when both arguments are provided.
    prompt_template : str or list of dict, optional
        Prompt template used to build each request. A list value should follow
        chat message shape, where message ``content`` may contain plain text,
        provider-compatible dicts, or ``ContentPart`` values.
    simple_output : bool, default False
        Whether to return the generated text directly when supported by the
        model executor, instead of the raw provider response.
    params : dict, optional
        Additional generation parameters such as temperature or max tokens.

    Returns
    -------
    DataFrame
        A DataFrame with ``response`` and ``success`` columns. Failed requests
        store the error message in ``response``.

    Examples
    --------
    Build chat messages with text and image URL placeholders:

    >>> from maxframe.learn.contrib.llm import ContentPart, ImageContentType
    >>> from maxframe.learn.contrib.llm.models.dashscope import DashScopeMultiModalLLM
    >>> import maxframe.dataframe as md
    >>>
    >>> df = md.DataFrame({"image_url": ["https://example.com/cat.png"]})
    >>> model = DashScopeMultiModalLLM(
    ...     name="qwen-vl-max",
    ...     api_key_resource="<api-key-resource-name>",
    ... )
    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": [
    ...             ContentPart.text("Analyze this image."),
    ...             ContentPart.image(
    ...                 data=df.image_url,
    ...                 type=ImageContentType.URL,
    ...             ),
    ...         ],
    ...     }
    ... ]
    >>> result = model.generate(df, messages=messages)

    Use an OSS object URL with explicit storage options when the image is not
    publicly reachable:

    >>> storage_options = {
    ...     "access_key_id": "<access-key-id>",
    ...     "access_key_secret": "<access-key-secret>",
    ... }
    >>> df = md.DataFrame({"image_url": ["oss://endpoint/bucket/path/cat.png"]})
    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": [
    ...             ContentPart.text("Analyze this OSS image."),
    ...             ContentPart.image(
    ...                 data=df.image_url,
    ...                 type=ImageContentType.URL,
    ...                 storage_options=storage_options,
    ...             ),
    ...         ],
    ...     }
    ... ]
    >>> result = model.generate(df, messages=messages)

    Notes
    -----
    ``ContentPart.image`` supports ``ImageContentType.URL``, ``ImageContentType.BINARY`` and
    ``ImageContentType.BASE64`` image content. ``ImageContentType.BINARY`` and ``ImageContentType.BASE64`` inputs must include ``mime_type``.
    """
    validate_llm_input_data(data)
    if not isinstance(model, MultiModalGenLLM):
        raise ValueError("model must be a MultiModalGenLLM object")
    params = params if params is not None else {}
    model.validate_params(params)
    return model.generate(
        data,
        messages=messages,
        prompt_template=prompt_template,
        simple_output=simple_output,
        params=params,
        **kw,
    )


def embed(
    data,
    model: MultiModalEmbeddingModel,
    input,
    simple_output: bool = False,
    params: Optional[Dict[str, Any]] = None,
    **kw,
):
    """
    Embed multimodal input with a multimodal embedding model.

    Parameters
    ----------
    data : DataFrame or Series
        Input data used to render one embedding request per row.
    model : MultiModalEmbeddingModel
        Multimodal embedding model instance.
    input : list or ContentPart
        Multimodal input template. Values may contain placeholders that
        reference columns in ``data``. The template is rendered row by row and
        sent as a single multimodal embedding input for that row.
    simple_output : bool, default False
        Whether to return embedding vectors directly when supported by the
        model executor, instead of the raw provider response.
    params : dict, optional
        Additional embedding parameters.

    Returns
    -------
    DataFrame
        A DataFrame with ``response`` and ``success`` columns. Failed requests
        store the error message in ``response``.

    Examples
    --------
    >>> from maxframe.learn.contrib.llm import ContentPart, ImageContentType
    >>> input = [
    ...     ContentPart.text("Represent this product image."),
    ...     ContentPart.image(
    ...         data=df.image_url,
    ...         type=ImageContentType.URL,
    ...     ),
    ... ]
    >>> result = model.embed(df, input=input, simple_output=True)
    """
    validate_llm_input_data(data)
    if not isinstance(model, MultiModalEmbeddingModel):
        raise ValueError("model must be a MultiModalEmbeddingModel object")
    params = params if params is not None else {}
    model.validate_params(params)
    return model.embed(
        data, input=input, simple_output=simple_output, params=params, **kw
    )
