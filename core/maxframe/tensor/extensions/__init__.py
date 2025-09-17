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

from ...core import CachedAccessor
from .accessor import TensorMaxFrameAccessor
from .apply_chunk import apply_chunk
from .rebalance import rebalance


def _install():
    from ..core import TENSOR_TYPE

    TensorMaxFrameAccessor._register("apply_chunk", apply_chunk)
    TensorMaxFrameAccessor._register("rebalance", rebalance)

    if TensorMaxFrameAccessor._api_count:
        for t in TENSOR_TYPE:
            t.mf = CachedAccessor("mf", TensorMaxFrameAccessor)


_install()
del _install
