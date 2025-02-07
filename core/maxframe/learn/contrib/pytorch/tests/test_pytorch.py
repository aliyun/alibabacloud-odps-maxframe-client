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

from ..... import dataframe as md
from ..... import tensor as mt
from .....core import TILEABLE_TYPE
from ..run_function import run_pytorch_function


def test_run_function():
    def test_func(a, b):
        raise ValueError(a + b)

    df = md.DataFrame(mt.random.rand(10, 10))
    t = run_pytorch_function(test_func, args=(df, 1), execute=False)

    global_dict = {"__name__": "__main__"}
    global_dict.update(t.op.data)

    try:
        exec(t.op.code, global_dict)
    except ValueError as ex:
        result = ex.args[0]
        assert isinstance(result, TILEABLE_TYPE)
        assert result.op.lhs.key == df.key
        assert result.op.rhs == 1
    else:
        raise AssertionError("Error not raised")

    assert global_dict["torch_var_0"] is df
    assert "torch_var_1" not in global_dict
