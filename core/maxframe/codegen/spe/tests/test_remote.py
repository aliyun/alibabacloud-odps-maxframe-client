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

from .... import remote as mr
from ..core import SPECodeContext
from ..remote import RemoteFunctionAdapter


def test_remote():
    def f(v1, v2):
        return v1 + v2

    v0 = mr.spawn(f, (2, 3))
    v1 = mr.spawn(f, (v0, 10))
    context = SPECodeContext()
    results = RemoteFunctionAdapter().generate_code(v1.op, context)
    assert callable(context.constants["const_0"])
    assert results[0] == "var_1 = const_0(var_0, 10)"
