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

from typing import Any, Dict, List

import numpy as np

from .... import opcodes
from ....dataframe.core import DataFrame, Series
from ....serialization.serializables import FieldTypes, ListField, StringField
from .core import LLM, LLMTaskOperator


class TextLLMSummarizeOperator(LLMTaskOperator):
    _op_type_ = opcodes.LLM_TEXT_SUMMARIZE_TASK

    def get_output_dtypes(self) -> Dict[str, np.dtype]:
        return {
            "summary": np.dtype("O"),
            "success": np.dtype("bool"),
        }


class TextLLMTranslateOperator(LLMTaskOperator):
    _op_type_ = opcodes.LLM_TEXT_TRANSLATE_TASK

    source_language = StringField("source_language")
    target_language = StringField("target_language")

    def get_output_dtypes(self) -> Dict[str, np.dtype]:
        return {
            "target": np.dtype("O"),
            "success": np.dtype("bool"),
        }


class TextLLMClassifyOperator(LLMTaskOperator):
    _op_type_ = opcodes.LLM_TEXT_CLASSIFY_TASK

    labels = ListField("labels")
    description = StringField("description", default=None)
    examples = ListField("examples", FieldTypes.dict, default=None)

    def get_output_dtypes(self) -> Dict[str, np.dtype]:
        return {
            "label": np.dtype("O"),
            "reason": np.dtype("O"),
            "success": np.dtype("bool"),
        }


class TextLLM(LLM):
    def generate(
        self,
        data,
        prompt_template: List[Dict[str, str]],
        params: Dict[str, Any] = None,
    ):
        raise NotImplementedError

    def summarize(self, series, index=None, **kw):
        return TextLLMSummarizeOperator(model=self, task="summarize", **kw)(
            series, index
        )

    def translate(
        self,
        series,
        target_language: str,
        source_language: str = None,
        index=None,
        **kw
    ):
        return TextLLMTranslateOperator(
            model=self,
            task="translate",
            source_language=source_language,
            target_language=target_language,
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
        return TextLLMClassifyOperator(
            model=self,
            labels=labels,
            task="classify",
            description=description,
            examples=examples,
            **kw
        )(series, index)


def generate(
    data,
    model: TextLLM,
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
    >>> from maxframe.learn.contrib.llm.models.managed import ManagedTextLLM
    >>> import maxframe.dataframe as md
    >>>
    >>> # Initialize the model
    >>> llm = ManagedTextLLM(name="Qwen2.5-0.5B-instruct")
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
    if not isinstance(model, TextLLM):
        raise TypeError("model must be a TextLLM object")
    params = params if params is not None else dict()
    model.validate_params(params)
    return model.generate(data, prompt_template=prompt_template, params=params)


def summary(series, model: TextLLM, index=None):
    """
    Generate summaries for text content in a series using a language model.

    Parameters
    ----------
    series : Series
        A maxframe Series containing text data to be summarized.
        Each element should be a text string.
    model : TextLLM
        Language model instance used for text summarization.
    index : array-like, optional
        Index for the output series, by default None, will generate new index.

    Returns
    -------
    maxframe.Series
        A pandas Series containing the generated summaries and success status.

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
    series, model: TextLLM, source_language: str, target_language: str, index=None
):
    """
    Translate text content in a series using a language model from source language to target language.

    Parameters
    ----------
    series : pandas.Series
        A maxframe Series containing text data to translate.
        Each element should be a text string.
    model : TextLLM
        Language model instance used for text summarization.
    source_language : str
        Source language of the text.
    target_language : str
        Target language of the text.
    index : array-like, optional
        Index for the output series, by default None, will generate new index.

    Returns
    -------
    maxframe.Series
        A pandas Series containing the generated translation and success status.

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
    model: TextLLM,
    labels: List[str],
    description: str = None,
    examples: List[Dict[str, str]] = None,
    index=None,
):
    """
    Classify text content in a series with given labels.

    Parameters
    ----------
    series : pandas.Series
        A maxframe Series containing text data to be classified.
        Each element should be a text string.
    model : TextLLM
        Language model instance used for text summarization.
    labels : List[str]
        List of labels to classify the text.
    description : str
        Description of the classification task.
    examples : List[Dict[str, Dict[str, str]]]
        Examples of the classification task, like [{ "text": "text...", "label":"A", reason : "reason..."}], help
        LLM to better understand your rules.
    index : array-like, optional
        Index for the output series, by default None, will generate new index.

    Returns
    -------
    maxframe.Series
        A pandas Series containing the generated classification results and success status.

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
