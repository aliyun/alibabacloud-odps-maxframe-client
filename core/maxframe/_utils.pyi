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

from typing import Callable, Optional, Type

def to_str(s, encoding: Optional[str] = None) -> str: ...
def to_binary(s, encoding: Optional[str] = None) -> bytes: ...
def register_tokenizer(cls: Type, handler: Callable) -> None: ...
def reset_id_random_seed() -> None: ...
def new_random_id(byte_len: int) -> bytes: ...
