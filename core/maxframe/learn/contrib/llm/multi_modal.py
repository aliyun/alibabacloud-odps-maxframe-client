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
from typing import Any, Dict

from ....dataframe.core import DATAFRAME_TYPE, SERIES_TYPE
from .core import LLM


class MultiModalLLM(LLM):
    def generate(
        self,
        data,
        prompt_template: Dict[str, Any],
        params: Dict[str, Any] = None,
    ):
        raise NotImplementedError


def generate(
    data,
    model: MultiModalLLM,
    prompt_template: Dict[str, Any],
    params: Dict[str, Any] = None,
):
    if not isinstance(data, DATAFRAME_TYPE) and not isinstance(data, SERIES_TYPE):
        raise ValueError("data must be a maxframe dataframe or series object")
    if not isinstance(model, MultiModalLLM):
        raise ValueError("model must be a MultiModalLLM object")
    params = params if params is not None else dict()
    model.validate_params(params)
    return model.generate(data, prompt_template, params)
