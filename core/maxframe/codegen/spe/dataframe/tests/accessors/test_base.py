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

from ...... import dataframe as md
from ....core import SPECodeContext
from ...accessors.base import SeriesDatetimeMethodAdapter, SeriesStringMethodAdapter


def test_datetime_methods():
    v0 = md.Series([md.Timestamp("2023-10-11 22:11:32.123")])
    v1 = v0.dt.year
    context = SPECodeContext()
    results = SeriesDatetimeMethodAdapter().generate_code(v1.op, context)
    assert results[0] == "var_1 = var_0.dt.year"


def test_string_methods():
    v0 = md.Series(["hello, world"])
    v1 = v0.str.upper()
    context = SPECodeContext()
    results = SeriesStringMethodAdapter().generate_code(v1.op, context)
    assert results[0] == "var_1 = var_0.str.upper()"
