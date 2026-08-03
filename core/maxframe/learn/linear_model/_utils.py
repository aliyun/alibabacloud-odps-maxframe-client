# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

import numpy as np

from maxframe.udf import builtin_function


@builtin_function
def _dict_sample_weight(y_chunk, weight_dict=None):
    """Convert class_weight dict to per-sample weights.

    Looks up each sample's label in *weight_dict* and returns the
    corresponding weight. Samples whose label is not in the dict get
    weight 1.0.

    This function is used via ``apply_chunk`` on the original (pre-encoding)
    y so that dict keys match the user's original class labels.
    """
    result = np.ones(len(y_chunk), dtype=np.float64)
    for label, weight in weight_dict.items():
        result[y_chunk == label] = weight
    return result


@builtin_function
def _apply_balanced_weight(y_chunk, class_weights_arr):
    """Map each sample's encoded class id to its balanced class weight.

    *class_weights_arr* is a 1-D array of length ``n_classes`` where
    ``class_weights_arr[k] = n_samples / (n_classes * count_k)``.
    """
    return class_weights_arr[y_chunk.astype(int)]
