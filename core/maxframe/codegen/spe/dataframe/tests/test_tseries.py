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
from ...core import SPECodeContext
from ..tseries import DataFrameToDatetimeAdapter


def test_to_datetime():
    v0 = md.DataFrame([["2023-07-27 14:58:12.1234"]], columns=["dt"], chunk_size=1)
    v1 = md.to_datetime(v0.dt, format="%Y-%M-%d %h:%m:%s.%f")
    context = SPECodeContext()
    results = DataFrameToDatetimeAdapter().generate_code(v1.op, context)
    assert results[0] == (
        "var_1 = pd.to_datetime(var_0, errors='raise', dayfirst=False, "
        "yearfirst=False, format='%Y-%M-%d %h:%m:%s.%f', exact=True, "
        "infer_datetime_format=False, origin='unix', cache=True)"
    )
