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

from ....tensor.spatial.distance.cdist import TensorCDist
from ....tensor.spatial.distance.pdist import TensorPDist
from ....tensor.spatial.distance.squareform import TensorSquareForm
from ..utils import build_method_call_adapter

TensorCDistAdapter = build_method_call_adapter(
    TensorCDist,
    "cdist",
    "xa",
    "xb",
    kw_keys=["metric", "p", "w"],
    v="V",
    vi="VI",
    source_module="scipy.spatial.distance",
)
TensorPDistAdapter = build_method_call_adapter(
    TensorPDist,
    "pdist",
    0,
    kw_keys=["metric", "p", "w"],
    v="V",
    vi="VI",
    source_module="scipy.spatial.distance",
)
TensorSquareFormAdapter = build_method_call_adapter(
    TensorSquareForm,
    "squareform",
    0,
    kw_keys=["checks"],
    source_module="scipy.spatial.distance",
)
