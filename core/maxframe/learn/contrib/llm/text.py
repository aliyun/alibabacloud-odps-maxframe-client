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

from typing import Any, Dict, List, Tuple

import numpy as np

from .... import opcodes
from ....dataframe.core import DataFrame, Series
from ....serialization.serializables import (
    DictField,
    FieldTypes,
    ListField,
    StringField,
)
from .core import LLM, LLMTaskOperator


class TextLLMSummarizeOp(LLMTaskOperator):
    _op_type_ = opcodes.LLM_TEXT_SUMMARIZE_TASK
    _legacy_name = "TextLLMSummarizeOperator"  # since v2.3.0

    def get_output_dtypes(self) -> Dict[str, np.dtype]:
        return {
            "summary": np.dtype("O"),
            "success": np.dtype("bool"),
        }


class TextLLMTranslateOp(LLMTaskOperator):
    _op_type_ = opcodes.LLM_TEXT_TRANSLATE_TASK
    _legacy_name = "TextLLMTranslateOperator"  # since v2.3.0

    source_language = StringField("source_language")
    target_language = StringField("target_language")
    description = StringField("description", default=None)
    examples = ListField("examples", FieldTypes.dict, default=None)

    def get_output_dtypes(self) -> Dict[str, np.dtype]:
        return {
            "output": np.dtype("O"),
            "success": np.dtype("bool"),
        }


class TextLLMClassifyOp(LLMTaskOperator):
    _op_type_ = opcodes.LLM_TEXT_CLASSIFY_TASK
    _legacy_name = "TextLLMClassifyOperator"  # since v2.3.0

    labels = ListField("labels")
    description = StringField("description", default=None)
    examples = ListField("examples", FieldTypes.dict, default=None)

    def get_output_dtypes(self) -> Dict[str, np.dtype]:
        return {
            "label": np.dtype("O"),
            "reason": np.dtype("O"),
            "success": np.dtype("bool"),
        }


class TextLLMExtractOp(LLMTaskOperator):
    _op_type_ = opcodes.LLM_TEXT_EXTRACT_TASK
    _legacy_name = "TextLLMExtractOperator"  # since v2.3.0

    schema = DictField("schema", FieldTypes.string, FieldTypes.any, default=None)
    description = StringField("description", default=None)
    examples = ListField("examples", FieldTypes.dict, default_factory=None)

    def get_output_dtypes(self) -> Dict[str, np.dtype]:
        return {
            "output": np.dtype("O"),
            "success": np.dtype("bool"),
        }


class TextGenLLM(LLM):
    _legacy_name = "TextLLM"  # since v2.3.0

    def generate(
        self,
        data,
        prompt_template: List[Dict[str, str]],
        params: Dict[str, Any] = None,
    ):
        raise NotImplementedError

    def summarize(self, series, index=None, **kw):
        return TextLLMSummarizeOp(model=self, task="summarize", **kw)(series, index)

    def translate(
        self,
        series,
        target_language: str,
        source_language: str = None,
        description: str = None,
        examples: List[Dict[str, str]] = None,
        index=None,
        **kw
    ):
        return TextLLMTranslateOp(
            model=self,
            task="translate",
            source_language=source_language,
            target_language=target_language,
            description=description,
            examples=examples,
            **kw
        )(series, index)

    def classify(
        self,
        series,
        labels: List[str],
        description=None,
        examples=None,
        index=None,
        **kw
    ):
        return TextLLMClassifyOp(
            model=self,
            labels=labels,
            task="classify",
            description=description,
            examples=examples,
            **kw
        )(series, index)

    def extract(
        self,
        series,
        schema: Any,
        description: str = None,
        examples: List[Tuple[str, str]] = None,
        index=None,
        **kw
    ):
        import inspect

        from pydantic import BaseModel

        if inspect.isclass(schema) and issubclass(schema, BaseModel):
            schema = schema.model_json_schema()

        return TextLLMExtractOp(
            model=self,
            schema=schema,
            task="extract",
            description=description,
            examples=examples,
            **kw
        )(series, index)


TextLLM = TextGenLLM  # for old client compatibility


class TextEmbeddingModel(LLM):
    def embed(
        self,
        data: Series,
        dimensions: int,
        encoding_format: str,
        simple_output: bool,
        params: Dict[str, Any],
        **kw
    ):
        raise NotImplementedError


def generate(
    data,
    model: TextGenLLM,
    prompt_template: List[Dict[str, Any]],
    params: Dict[str, Any] = None,
):
    """
    Generate text using a text language model based on given data and prompt template.

    Parameters
    ----------
    data : DataFrame or Series
        Input data used for generation. Can be maxframe DataFrame, Series that contain text to be processed.
    model : TextLLM
        Language model instance used for text generation.
    prompt_template : List[Dict[str, str]]
        Dictionary containing the conversation messages template. Use ``{col_name}`` as a placeholder to reference
        column data from input data.

        Usually in format of [{"role": "user", "content": "{query}"}], same with openai api schema.
    params : Dict[str, Any], optional
        Additional parameters for generation configuration, by default None.
        Can include settings like temperature, max_tokens, etc.

    Returns
    -------
    DataFrame
        Generated text raw response and success status. If the success is False, the generated text will return the
        error message.

    Examples
    --------
    >>> from maxframe.learn.contrib.llm.models.managed import ManagedTextGenLLM
    >>> import maxframe.dataframe as md
    >>>
    >>> # Initialize the model
    >>> llm = ManagedTextGenLLM(name="Qwen3-0.6B")
    >>>
    >>> # Prepare prompt template
    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": "Help answer following question: {query}",
    ...     },
    ... ]

    >>> # Create sample data
    >>> df = md.DataFrame({"query": ["What is machine learning?"]})
    >>>
    >>> # Generate response
    >>> result = generate(df, llm, prompt_template=messages)
    >>> result.execute()
    """
    if not isinstance(data, DataFrame) and not isinstance(data, Series):
        raise ValueError("data must be a maxframe dataframe or series object")
    if not isinstance(model, TextGenLLM):
        raise TypeError("model must be a TextLLM object")
    params = params if params is not None else dict()
    model.validate_params(params)
    return model.generate(data, prompt_template=prompt_template, params=params)


def summary(series, model: TextGenLLM, index=None):
    """
    Generate summaries for text content in a series using a language model.

    Parameters
    ----------
    series : Series
        A maxframe Series containing text data to be summarized.
        Each element should be a text string.
    model : TextGenLLM
        Language model instance used for text summarization.
    index : array-like, optional
        Index for the output series, by default None, will generate new index.

    Returns
    -------
    DataFrame
        A DataFrame containing the generated summaries and success status.
        Columns include 'summary' (generated summary text) and 'success' (boolean status).
        If 'success' is False, the 'summary' column will contain error information instead of the expected output.

    Examples
    --------
    >>> from maxframe.learn.contrib.llm.models.managed import ManagedTextGenLLM
    >>> import maxframe.dataframe as md
    >>>
    >>> # Initialize the model
    >>> llm = ManagedTextGenLLM(name="Qwen3-0.6B")
    >>>
    >>> # Create sample data
    >>> texts = md.Series([
    ...     "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
    ...     "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data."
    ... ])
    >>>
    >>> # Generate summaries
    >>> result = summary(texts, llm)
    >>> result.execute()

    Notes
    -----
      **Preview:** This API is in preview state and may be unstable.
      The interface may change in future releases.
    """
    if not isinstance(series, Series):
        raise ValueError("series must be a maxframe series object")

    if series.dtype != np.str_:
        raise ValueError("summary input must be a string series")

    return model.summarize(series, index=index)


def translate(
    series, model: TextGenLLM, source_language: str, target_language: str, index=None
):
    """
    Translate text content in a series using a language model from source language to target language.

    Parameters
    ----------
    series : Series
        A maxframe Series containing text data to translate.
        Each element should be a text string.
    model : TextGenLLM
        Language model instance used for text translation.
    source_language : str
        Source language of the text (e.g., 'en', 'zh', 'ja').
    target_language : str
        Target language for translation (e.g., 'en', 'zh', 'ja').
    index : array-like, optional
        Index for the output series, by default None, will generate new index.

    Returns
    -------
    DataFrame
        A DataFrame containing the generated translations and success status.
        Columns include 'output' (translated text) and 'success' (boolean status).
        If 'success' is False, the 'output' column will contain error information instead of the expected output.

    Examples
    --------
    >>> from maxframe.learn.contrib.llm.models.managed import ManagedTextGenLLM
    >>> import maxframe.dataframe as md
    >>>
    >>> # Initialize the model
    >>> llm = ManagedTextGenLLM(name="Qwen3-0.6B")
    >>>
    >>> # Create sample data
    >>> texts = md.Series([
    ...     "Hello, how are you?",
    ...     "Machine learning is fascinating."
    ... ])
    >>>
    >>> # Translate from English to Chinese
    >>> result = translate(texts, llm, source_language="en", target_language="zh")
    >>> result.execute()

    Notes
    -----
      **Preview:** This API is in preview state and may be unstable.
      The interface may change in future releases.
    """
    if not isinstance(series, Series):
        raise ValueError("series must be a maxframe series object")
    if series.dtype != np.str_:
        raise ValueError("translate input must be a string series")
    return model.translate(
        series,
        source_language=source_language,
        target_language=target_language,
        index=index,
    )


def classify(
    series,
    model: TextGenLLM,
    labels: List[str],
    description: str = None,
    examples: List[Dict[str, str]] = None,
    index=None,
):
    """
    Classify text content in a series with given labels using a language model.

    Parameters
    ----------
    series : Series
        A maxframe Series containing text data to be classified.
        Each element should be a text string.
    model : TextGenLLM
        Language model instance used for text classification.
    labels : List[str]
        List of labels to classify the text into.
    description : str, optional
        Description of the classification task to help the model understand the context.
    examples : List[Dict[str, str]], optional
        Examples of the classification task, like [{"text": "text...", "label": "A", "reason": "reason..."}],
        to help LLM better understand your classification rules.
    index : array-like, optional
        Index for the output series, by default None, will generate new index.

    Returns
    -------
    DataFrame
        A DataFrame containing the generated classification results and success status.
        Columns include 'label' (predicted label), 'reason' (reasoning), and 'success' (boolean status).
        If 'success' is False, the 'label' and 'reason' columns will contain error information instead of the expected output.

    Examples
    --------
    >>> from maxframe.learn.contrib.llm.models.managed import ManagedTextGenLLM
    >>> import maxframe.dataframe as md
    >>>
    >>> # Initialize the model
    >>> llm = ManagedTextGenLLM(name="Qwen3-0.6B")
    >>>
    >>> # Create sample data
    >>> texts = md.Series([
    ...     "I love this product! It's amazing!",
    ...     "This is terrible, worst purchase ever.",
    ...     "It's okay, nothing special."
    ... ])
    >>>
    >>> # Classify sentiment
    >>> labels = ["positive", "negative", "neutral"]
    >>> description = "Classify the sentiment of customer reviews"
    >>> examples = [
    ...     {"text": "Great product!", "label": "positive", "reason": "Expresses satisfaction"},
    ...     {"text": "Poor quality", "label": "negative", "reason": "Expresses dissatisfaction"}
    ... ]
    >>> result = classify(texts, llm, labels=labels, description=description, examples=examples)
    >>> result.execute()

    Notes
    -----
      **Preview:** This API is in preview state and may be unstable.
      The interface may change in future releases.
    """
    if not isinstance(series, Series):
        raise ValueError("series must be a maxframe series object")

    if series.dtype != np.str_:
        raise ValueError("classify input must be a string series")

    if not isinstance(labels, list):
        raise TypeError("labels must be a list")

    if not labels:
        raise ValueError("labels must not be empty")

    return model.classify(
        series, labels=labels, description=description, examples=examples, index=index
    )


def extract(
    series,
    model: TextGenLLM,
    schema: Any,
    description: str = None,
    examples: List[Tuple[str, str]] = None,
    index=None,
):
    """
    Extract structured information from text content in a series using a language model.

    Parameters
    ----------
    series : Series
        A maxframe Series containing text data to extract information from.
        Each element should be a text string.
    model : TextGenLLM
        Language model instance used for information extraction.
    schema : Any
        Schema definition for the extraction. Can be a dictionary defining the structure
        or a Pydantic BaseModel class that will be converted to JSON schema.
    description : str, optional
        Description of the extraction task to help the model understand what to extract.
    examples : List[Tuple[str, str]], optional
        Examples of the extraction task in format [(input_text, expected_output), ...],
        to help LLM better understand the extraction requirements.
    index : array-like, optional
        Index for the output series, by default None, will generate new index.

    Returns
    -------
    DataFrame
        A DataFrame containing the extracted information and success status.
        Columns include 'output' (extracted structured data) and 'success' (boolean status).
        If 'success' is False, the 'output' column will contain error information instead of the expected output.

    Examples
    --------
    >>> from maxframe.learn.contrib.llm.models.managed import ManagedTextGenLLM
    >>> import maxframe.dataframe as md
    >>>
    >>> # Initialize the model
    >>> llm = ManagedTextGenLLM(name="Qwen3-0.6B")
    >>>
    >>> # Create sample data
    >>> texts = md.Series([
    ...     "John Smith, age 30, works as a Software Engineer at Google.",
    ...     "Alice Johnson, 25 years old, is a Data Scientist at Microsoft."
    ... ])
    >>>
    >>> # Define extraction schema
    >>> schema = {
    ...     "name": "string",
    ...     "age": "integer",
    ...     "job_title": "string",
    ...     "company": "string"
    ... }
    >>>
    >>> # Extract structured information
    >>> description = "Extract person information from text"
    >>> examples = [
    ...     ("Bob Brown, 35, Manager at Apple", '{"name": "Bob Brown", "age": 35, "job_title": "Manager", "company": "Apple"}')
    ... ]
    >>> result = extract(texts, llm, schema=schema, description=description, examples=examples)
    >>> result.execute()

    Notes
    -----
      **Preview:** This API is in preview state and may be unstable.
      The interface may change in future releases.
    """
    if not isinstance(series, Series):
        raise ValueError("series must be a maxframe series object")
    if series.dtype != np.str_:
        raise ValueError("extract input must be a string series")
    if not schema:
        raise ValueError("schema must not be empty")
    if (
        examples
        and not isinstance(examples, list)
        or not any(isinstance(x, Tuple) for x in examples)
    ):
        raise ValueError("examples must be a list of tuples, format is (input, output)")
    return model.extract(
        series, schema=schema, description=description, examples=examples, index=index
    )


def embed(
    series,
    model: TextEmbeddingModel,
    dimensions: int = None,
    encoding_format: str = None,
    simple_output: bool = False,
    params: Dict[str, Any] = None,
    index=None,
):
    """
    Embed text content in a series using a text embedding model.

    Parameters
    ----------
    series : Series
        A maxframe Series containing text data to be embedded.
        Each element should be a text string.
    model : TextEmbeddingModel
        Text embedding model instance used for generating embeddings.
    dimensions : int, optional
        Dimensions of the embedding vectors. If not specified, uses model default.
    encoding_format : str, optional
        Encoding format of the embedding (e.g., 'float', 'base64'). If not specified, uses model default.
    simple_output : bool, optional
        Whether to return the embedding data directly without additional metadata, by default False.
    params : Dict[str, Any], optional
        Additional parameters for embedding configuration, by default None.
        Can include model-specific settings.
    index : array-like, optional
        Index for the output series, by default None, will generate new index.

    Returns
    -------
    DataFrame
        A DataFrame containing the generated embeddings and success status.
        Columns include 'response' (embedding vectors) and 'success' (boolean status).
        If 'success' is False, the 'response' column will contain error information instead of the expected output.

    Examples
    --------
    >>> from maxframe.learn.contrib.llm.models.managed import ManagedTextEmbeddingModel
    >>> import maxframe.dataframe as md
    >>>
    >>> # Initialize the embedding model
    >>> embedding_model = ManagedTextEmbeddingModel(name="text-embedding-ada-002")
    >>>
    >>> # Create sample data
    >>> texts = md.Series([
    ...     "Machine learning is a powerful technology.",
    ...     "Natural language processing enables computers to understand text.",
    ...     "Deep learning uses neural networks for pattern recognition."
    ... ])
    >>>
    >>> # Generate embeddings
    >>> result = embed(texts, embedding_model, simple_output=True)
    >>> result.execute()

    Notes
    -----
      **Preview:** This API is in preview state and may be unstable.
      The interface may change in future releases.
    """
    if not isinstance(series, Series):
        raise ValueError("series must be a maxframe series object")
    if series.dtype != np.str_:
        raise ValueError("embed input must be a string series")
    return model.embed(
        series,
        dimensions=dimensions,
        encoding_format=encoding_format,
        simple_output=simple_output,
        params=params,
        index=index,
    )


TextLLMExtractOperator = TextLLMExtractOp
TextLLMSummarizeOperator = TextLLMSummarizeOp
TextLLMTranslateOperator = TextLLMTranslateOp
TextLLMClassifyOperator = TextLLMClassifyOp
