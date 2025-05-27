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

from ...typing_ import TileableType
from ...utils import calc_nsplits, has_unknown_shape


def refresh_tileable_shape(tileable):
    if has_unknown_shape(tileable):
        # update shape
        nsplits = calc_nsplits({c.index: c.shape for c in tileable.chunks})
        shape = tuple(sum(ns) for ns in nsplits)
        tileable._nsplits = nsplits
        tileable._shape = shape


def fill_chunk_slices(tileable: TileableType) -> None:
    if tileable is None or not tileable.nsplits or not tileable.chunks:
        # errors may occurred
        return
    acc_splits = tuple((0,) + tuple(np.cumsum(sp)) for sp in tileable.nsplits)
    for chunk in tileable.chunks:
        chunk.slices = list(
            (acc_splits[i][chunk.index[i]], acc_splits[i][chunk.index[i] + 1])
            for i in range(tileable.ndim)
        )
