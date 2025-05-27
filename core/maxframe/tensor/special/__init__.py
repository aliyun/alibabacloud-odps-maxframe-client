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

from .misc import softmax, softplus
from .statistical import (
    TensorExpit,
    TensorLogExpit,
    TensorLogit,
    expit,
    log_expit,
    logit,
)


def _clear_nones():
    """Remove unsupported functions"""
    erase_key = []
    for member, val in globals().items():
        if val is None:
            erase_key.append(member)
    for key in erase_key:
        globals().pop(key, None)


_clear_nones()
del _clear_nones
