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

from maxframe import tensor as mt
from maxframe.codegen.spe.core import SPECodeGenerator
from maxframe.core.graph.builder.utils import build_graph
from maxframe.learn.model_selection import train_test_split


def test_train_test_split_without_shuffling():
    X, y = mt.arange(10).reshape((5, 2)), range(5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    graph = build_graph([X_train, X_test, y_train, y_test])
    codegen = SPECodeGenerator("test_1")
    code = codegen.generate_code(graph)

    local_vars = codegen._context.constants.copy()
    exec("\n".join(code), local_vars, local_vars)

    def _get_result(tileable):
        return local_vars[
            codegen._context.get_output_tileable_variable(tileable.op.outputs[0])
        ]

    assert np.array_equal(np.array([[0, 1], [2, 3], [4, 5]]), _get_result(X_train))
    assert np.array_equal(np.array([[6, 7], [8, 9]]), _get_result(X_test))
    assert np.array_equal(np.array([0, 1, 2]), _get_result(y_train))
    assert np.array_equal(np.array([3, 4]), _get_result(y_test))
