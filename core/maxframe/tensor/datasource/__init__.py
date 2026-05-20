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

from maxframe.tensor.datasource.arange import TensorArange, arange
from maxframe.tensor.datasource.array import (
    ArrayDataSource,
    CSRMatrixDataSource,
    array,
    asarray,
    ascontiguousarray,
    asfortranarray,
    tensor,
)
from maxframe.tensor.datasource.diag import TensorDiag, diag
from maxframe.tensor.datasource.diagflat import diagflat
from maxframe.tensor.datasource.empty import (
    TensorEmpty,
    TensorEmptyLike,
    empty,
    empty_like,
)
from maxframe.tensor.datasource.eye import TensorEye, eye
from maxframe.tensor.datasource.from_dataframe import (
    from_dataframe,
    from_index,
    from_series,
)
from maxframe.tensor.datasource.from_dense import fromdense
from maxframe.tensor.datasource.from_sparse import fromsparse
from maxframe.tensor.datasource.full import full, full_like
from maxframe.tensor.datasource.identity import identity
from maxframe.tensor.datasource.indices import indices
from maxframe.tensor.datasource.linspace import TensorLinspace, linspace
from maxframe.tensor.datasource.meshgrid import meshgrid
from maxframe.tensor.datasource.ones import TensorOnes, TensorOnesLike, ones, ones_like
from maxframe.tensor.datasource.scalar import Scalar, scalar
from maxframe.tensor.datasource.tri_array import TensorTril, TensorTriu, tril, triu
from maxframe.tensor.datasource.zeros import (
    TensorZeros,
    TensorZerosLike,
    zeros,
    zeros_like,
)
