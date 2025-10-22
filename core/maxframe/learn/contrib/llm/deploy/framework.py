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

from enum import Enum


class InferenceFrameworkEnum(Enum):
    LLAMA_CPP_PYTHON_TEXT = "LLAMA_CPP_PYTHON:TEXT"
    LLAMA_CPP_SERVE_TEXT = "LLAMA_CPP_SERVE:TEXT"
    DASH_SCOPE_TEXT = "DASH_SCOPE:TEXT"
    DASH_SCOPE_MULTIMODAL = "DASH_SCOPE:MULTIMODAL"
    VLLM_SERVE_TEXT = "VLLM_SERVE:TEXT"
    OPENAI_REMOTE_TEXT = "OPENAI_REMOTE:TEXT"
    OTHER = "OTHER"

    @classmethod
    def from_string(cls, label):
        if label is None:
            return None

        if isinstance(label, cls):
            return label

        return cls(label)
