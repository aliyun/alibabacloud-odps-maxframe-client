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

from .....learn.metrics.pairwise.cosine import CosineDistances
from .....learn.metrics.pairwise.euclidean import EuclideanDistances
from .....learn.metrics.pairwise.haversine import HaversineDistances
from .....learn.metrics.pairwise.manhattan import ManhattanDistances
from ...utils import build_method_call_adapter

CosineDistancesAdapter = build_method_call_adapter(
    CosineDistances,
    "cosine_distances",
    "x",
    "y",
    source_module="sklearn.metrics.pairwise",
)
EuclideanDistancesAdapter = build_method_call_adapter(
    EuclideanDistances,
    "euclidean_distances",
    "x",
    "y",
    y_norm_squared="Y_norm_squared",
    squared="squared",
    x_norm_squared="X_norm_squared",
    source_module="sklearn.metrics.pairwise",
)
HaversineDistancesAdapter = build_method_call_adapter(
    HaversineDistances,
    "haversine_distances",
    "x",
    "y",
    source_module="sklearn.metrics.pairwise",
)
ManhattanDistancesAdapter = build_method_call_adapter(
    ManhattanDistances,
    "manhattan_distances",
    "x",
    "y",
    source_module="sklearn.metrics.pairwise",
)
