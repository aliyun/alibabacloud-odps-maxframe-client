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

from numbers import Integral
from typing import List, TypeVar, Union

import numpy as np
import pandas as pd
import pyarrow as pa

SlicesType = List[Union[None, Integral, slice]]

TimeoutType = Union[int, float, None]

PandasDType = Union[np.dtype, pd.api.extensions.ExtensionDtype]

ArrowTableType = Union[pa.Table, pa.RecordBatch]
PandasObjectTypes = Union[
    pd.DataFrame,
    pd.Series,
    pd.Index,
]

OperatorType = TypeVar("OperatorType")
TileableType = TypeVar("TileableType")
ChunkType = TypeVar("ChunkType")
EntityType = TypeVar("EntityType")
SessionType = TypeVar("SessionType")

ClusterType = TypeVar("ClusterType")
ClientType = TypeVar("ClientType")
