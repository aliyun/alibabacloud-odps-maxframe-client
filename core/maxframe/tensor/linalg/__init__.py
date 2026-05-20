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

from maxframe.tensor.linalg.cholesky import TensorCholesky, cholesky
from maxframe.tensor.linalg.dot import TensorDot, dot
from maxframe.tensor.linalg.einsum import TensorEinsum, einsum
from maxframe.tensor.linalg.inner import inner, innerproduct
from maxframe.tensor.linalg.inv import TensorInv, inv
from maxframe.tensor.linalg.lstsq import lstsq
from maxframe.tensor.linalg.lu import TensorLU, lu
from maxframe.tensor.linalg.matmul import TensorMatmul, matmul
from maxframe.tensor.linalg.matrix_norm import matrix_norm
from maxframe.tensor.linalg.norm import TensorNorm, norm
from maxframe.tensor.linalg.qr import TensorQR, qr
from maxframe.tensor.linalg.solve import solve
from maxframe.tensor.linalg.solve_triangular import (
    TensorSolveTriangular,
    solve_triangular,
)
from maxframe.tensor.linalg.svd import TensorSVD, svd
from maxframe.tensor.linalg.tensordot import TensorTensorDot, tensordot
from maxframe.tensor.linalg.vdot import vdot
from maxframe.tensor.linalg.vector_norm import vector_norm


def _install():
    from maxframe.tensor.core import Tensor, TensorData

    for cls in (Tensor, TensorData):
        setattr(cls, "__matmul__", matmul)
        setattr(cls, "dot", dot)


_install()
del _install
