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

import base64
import hashlib
import textwrap

import pytest

from .... import dataframe as md
from .... import tensor as mt
from ....dataframe.utils import bind_func_args_from_pos
from ....lib import wrapped_pickle as pickle
from ..core import SPECodeGenerator


@pytest.fixture
def codegen():
    return SPECodeGenerator("session_1")


def test_simple_codegen(codegen):
    df = md.DataFrame(mt.random.rand(1000, 5), columns=list("ABCDE"))
    df["F"] = df["A"] + df["B"]
    df["G"] = df["C"] * 2 + df["D"]
    result = df.cumsum()
    dag = result.build_graph()
    generated = codegen.generate(dag)

    local_vars = generated.constants.copy()
    exec(generated.code, {}, local_vars)
    df_result = local_vars[generated.output_key_to_variables[result.key]]
    assert len(df_result.columns) == 7


def test_codegen_with_udf(codegen):
    df = md.DataFrame(mt.random.rand(1, 3), columns=list("ABC"))

    def f1(x) -> int:
        return x + 1

    def f2(x) -> int:
        return x - 1

    result = df.transform({"A": f1, "B": [f1, f2]})
    dag = result.build_graph()
    generated = codegen.generate(dag)

    udf_1_body = base64.b64encode(pickle.dumps(f1, protocol=pickle.DEFAULT_PROTOCOL))
    udf_2_body = base64.b64encode(pickle.dumps(f2, protocol=pickle.DEFAULT_PROTOCOL))
    udf_1_value = f"udf_main_entry = cloudpickle.loads(base64.b64decode({udf_1_body}), buffers=[])"
    udf_2_value = f"udf_main_entry = cloudpickle.loads(base64.b64decode({udf_2_body}), buffers=[])"
    udf_1 = f"user_udf_f1_{hashlib.md5(udf_1_value.encode('utf-8')).hexdigest()}"
    udf_2 = f"user_udf_f2_{hashlib.md5(udf_2_value.encode('utf-8')).hexdigest()}"

    expected_contents = f"""
    import numpy as np
    import pandas as pd
    import base64
    import cloudpickle
    {udf_1_value}
    {udf_1} = udf_main_entry
    {udf_2_value}
    {udf_2} = udf_main_entry
    if not running:
        raise RuntimeError('CANCELLED')
    var_0 = np.random.rand(1, 3)

    if not running:
        raise RuntimeError('CANCELLED')
    var_1 = pd.DataFrame(var_0, index=const_0, columns=['A', 'B', 'C'])

    del var_0
    if not running:
        raise RuntimeError('CANCELLED')
    var_2 = var_1.transform({{'A': {udf_1}, 'B': [{udf_1}, {udf_2}]}}, axis=0)

    del var_1
    """
    assert generated.code == textwrap.dedent(expected_contents).strip()


def test_codegen_with_udf_and_args(codegen):
    df = md.DataFrame(mt.random.rand(1, 3), columns=list("ABC"))

    def f1(x, y, a, b) -> int:
        return x + y + a + b

    result = df.transform({"A": f1}, 0, 2, a=3, b=4)
    dag = result.build_graph()
    generated = codegen.generate(dag)

    udf_1_body = base64.b64encode(
        pickle.dumps(
            bind_func_args_from_pos(f1, 1, 2, a=3, b=4),
            protocol=pickle.DEFAULT_PROTOCOL,
        )
    )
    udf_1_value = f"udf_main_entry = cloudpickle.loads(base64.b64decode({udf_1_body}), buffers=[])"
    udf_1 = f"user_udf_f1_{hashlib.md5(udf_1_value.encode('utf-8')).hexdigest()}"

    expected_contents = f"""
    import numpy as np
    import pandas as pd
    import base64
    import cloudpickle
    {udf_1_value}
    {udf_1} = udf_main_entry
    if not running:
        raise RuntimeError('CANCELLED')
    var_0 = np.random.rand(1, 3)

    if not running:
        raise RuntimeError('CANCELLED')
    var_1 = pd.DataFrame(var_0, index=const_0, columns=['A', 'B', 'C'])

    del var_0
    if not running:
        raise RuntimeError('CANCELLED')
    var_2 = var_1.transform({{'A': {udf_1}}}, axis=0)

    del var_1
    """
    assert generated.code == textwrap.dedent(expected_contents).strip()
