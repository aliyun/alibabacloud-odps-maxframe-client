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

from io import StringIO

import pytest

from ...... import dataframe as md
from ......learn.contrib.pytorch import run_pytorch_script
from ....core import SPECodeContext
from ..pytorch import RunPyTorchAdapter


@pytest.fixture
def df1():
    return md.DataFrame(
        [
            [1, 10, 100, 1000],
            [2, 20, 200, 2000],
            [3, 30, 300, 3000],
            [7, 70, 700, 7000],
        ],
        index=md.Index([1, 2, 3, 7], name="test_idx"),
        columns=list("ABCD"),
    )


def test_run_pytorch_script(df1):
    res_t = run_pytorch_script(
        StringIO("print(var)"),
        n_workers=1,
        data={"var": df1},
        command_argv=["--run"],
        execute=False,
    )
    context = SPECodeContext()
    adapter = RunPyTorchAdapter()
    results = adapter.generate_code(res_t.op, context)
    assert results == [
        "var_1 = RunPyTorchAdapter._run_script("
        "b'print(var)', dict(var=var_0), [\"--run\"])"
    ]
