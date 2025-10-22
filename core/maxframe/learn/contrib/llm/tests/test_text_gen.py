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

import mock
import numpy as np
import pytest

import maxframe.dataframe as md

from .. import text as llm_text


def test_generate_invalid_data_type_raises():
    model = mock.create_autospec(llm_text.TextGenLLM, instance=True)
    with pytest.raises(ValueError):
        llm_text.generate(
            123, model, prompt_template=[{"role": "user", "content": "x"}]
        )


def test_generate_invalid_model_type_raises():
    df = md.DataFrame({"query": ["x"]})
    with pytest.raises(TypeError):
        llm_text.generate(
            df, object(), prompt_template=[{"role": "user", "content": "x"}]
        )


def test_generate_calls_validate_params_with_default_and_forwards():
    df = md.DataFrame({"query": ["hello"]})
    model = mock.create_autospec(llm_text.TextGenLLM, instance=True)
    sentinel = object()
    model.generate.return_value = sentinel

    ret = llm_text.generate(
        df, model, prompt_template=[{"role": "user", "content": "x"}]
    )

    assert ret is sentinel
    model.validate_params.assert_called_once()
    assert model.validate_params.call_args[0][0] == {}
    assert model.generate.call_args.kwargs["prompt_template"] == [
        {"role": "user", "content": "x"}
    ]
    assert model.generate.call_args.kwargs["params"] == {}


def test_summary_type_and_dtype_validation_and_forward():
    model = mock.create_autospec(llm_text.TextGenLLM, instance=True)

    with pytest.raises(ValueError):
        llm_text.summary("not_series", model)

    s_wrong = md.Series(np.array([1], dtype=np.int_))
    with pytest.raises(ValueError):
        llm_text.summary(s_wrong, model)

    s_ok = md.Series(np.array(["a"], dtype=np.str_))
    model.summarize.return_value = "OK_SUM"
    with mock.patch.object(llm_text.np, "str_", s_ok.dtype, create=True):
        ret = llm_text.summary(s_ok, model)
    assert ret == "OK_SUM"
    model.summarize.assert_called_once()


def test_translate_type_and_dtype_validation_and_forward():
    model = mock.create_autospec(llm_text.TextGenLLM, instance=True)

    with pytest.raises(ValueError):
        llm_text.translate(
            "not_series", model, source_language="en", target_language="zh"
        )

    s_wrong = md.Series(np.array([1], dtype=np.int_))
    with pytest.raises(ValueError):
        llm_text.translate(s_wrong, model, source_language="en", target_language="zh")

    s_ok = md.Series(np.array(["hello"], dtype=np.str_))
    model.translate.return_value = "OK_TRANS"
    with mock.patch.object(llm_text.np, "str_", s_ok.dtype, create=True):
        ret = llm_text.translate(
            s_ok, model, source_language="en", target_language="zh"
        )
    assert ret == "OK_TRANS"
    model.translate.assert_called_once()


def test_classify_validation_and_forward():
    model = mock.create_autospec(llm_text.TextGenLLM, instance=True)

    with pytest.raises(ValueError):
        llm_text.classify("not_series", model, labels=["A", "B"])

    s_wrong = md.Series(np.array([1], dtype=np.int_))
    with pytest.raises(ValueError):
        llm_text.classify(s_wrong, model, labels=["A", "B"])

    s_ok = md.Series(np.array(["text"], dtype=np.str_))
    with pytest.raises(TypeError):
        with mock.patch.object(llm_text.np, "str_", s_ok.dtype, create=True):
            llm_text.classify(s_ok, model, labels="not_list")

    with pytest.raises(ValueError):
        with mock.patch.object(llm_text.np, "str_", s_ok.dtype, create=True):
            llm_text.classify(s_ok, model, labels=[])

    model.classify.return_value = "OK_CLS"
    with mock.patch.object(llm_text.np, "str_", s_ok.dtype, create=True):
        ret = llm_text.classify(
            s_ok,
            model,
            labels=["A", "B"],
            description="desc",
            examples=[{"text": "t", "label": "A", "reason": "r"}],
        )
    assert ret == "OK_CLS"
    model.classify.assert_called_once()


def test_extract_validation_and_forward():
    model = mock.create_autospec(llm_text.TextGenLLM, instance=True)

    with pytest.raises(ValueError):
        llm_text.extract("not_series", model, schema={"a": "b"})

    s_wrong = md.Series(np.array([1], dtype=np.int_))
    with pytest.raises(ValueError):
        llm_text.extract(s_wrong, model, schema={"a": "b"})

    s_ok = md.Series(np.array(["text"], dtype=np.str_))
    with pytest.raises(ValueError):
        llm_text.extract(s_ok, model, schema=None)

    with pytest.raises(ValueError):
        llm_text.extract(s_ok, model, schema={"a": "b"}, examples="not_list")

    with pytest.raises(ValueError):
        llm_text.extract(s_ok, model, schema={"a": "b"}, examples=[{"not": "tuple"}])

    model.extract.return_value = "OK_EXT"
    with mock.patch.object(llm_text.np, "str_", s_ok.dtype, create=True):
        ret = llm_text.extract(s_ok, model, schema={"a": "b"}, examples=[("in", "out")])
    assert ret == "OK_EXT"
    model.extract.assert_called_once()
