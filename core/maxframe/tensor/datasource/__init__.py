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

from .arange import TensorArange, arange
from .array import (
    ArrayDataSource,
    CSRMatrixDataSource,
    array,
    asarray,
    ascontiguousarray,
    asfortranarray,
    tensor,
)
from .diag import TensorDiag, diag
from .diagflat import diagflat
from .empty import TensorEmpty, TensorEmptyLike, empty, empty_like
from .eye import TensorEye, eye
from .from_dataframe import from_dataframe, from_index, from_series
from .from_dense import fromdense
from .from_sparse import fromsparse
from .full import full, full_like
from .identity import identity
from .indices import indices
from .linspace import TensorLinspace, linspace
from .meshgrid import meshgrid
from .ones import TensorOnes, TensorOnesLike, ones, ones_like
from .scalar import Scalar, scalar
from .tri_array import TensorTril, TensorTriu, tril, triu
from .zeros import TensorZeros, TensorZerosLike, zeros, zeros_like
