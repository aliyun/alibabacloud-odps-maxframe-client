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


def inv(m):
    from .matrix import inv_sparse_matrix

    return inv_sparse_matrix(m)


def lu(m):
    from .matrix import lu_sparse_matrix

    return lu_sparse_matrix(m)


def solve_triangular(a, b, lower=False, sparse=True):
    from .matrix import solve_triangular_sparse_matrix

    return solve_triangular_sparse_matrix(a, b, lower=lower, sparse=sparse)
