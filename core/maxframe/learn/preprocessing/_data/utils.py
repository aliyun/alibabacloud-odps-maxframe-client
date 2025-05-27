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

import numpy as np

from .... import tensor as mt
from ....tensor.core import TENSOR_TYPE


def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    """Set scales of near constant features to 1.

    The goal is to avoid division by very small or zero values.

    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.

    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):  # pragma: no cover
        if scale == 0.0:
            scale = 1.0
        return scale
    elif hasattr(scale, "ndim") and scale.ndim == 0:  # pragma: no cover
        # scalar that is tensor
        return mt.where(scale == 0.0, 1.0, scale)
    elif isinstance(scale, (np.ndarray, TENSOR_TYPE)):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()

        if constant_mask is None:
            constant_mask = scale < 10 * np.finfo(scale.dtype).eps
        scale[constant_mask] = 1.0
        return scale


def _get_real_sample_size(tileable):
    if not np.isnan(tileable.shape[0]):
        return mt.asarray(tileable.shape[0])

    def get_real_shape(s):
        return s.shape[0]

    slc = [slice(None, None, None)] + [0] * (tileable.ndim - 1)
    return tileable[slc].mf.apply_chunk(get_real_shape, shape=(np.nan,)).sum()


def _is_constant_feature(var, mean, n_samples):
    """Detect if a feature is indistinguishable from a constant feature.

    The detection is based on its computed variance and on the theoretical
    error bounds of the '2 pass algorithm' for variance computation.

    See "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    """
    # In scikit-learn, variance is always computed using float64 accumulators.
    eps = np.finfo(np.float64).eps

    upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2
    return var <= upper_bound
