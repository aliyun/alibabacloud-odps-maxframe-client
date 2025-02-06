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

import io
import pickle

import pytest

from ..wrapped_pickle import switch_unpickle


@pytest.mark.asyncio
async def test_wrapped_pickle():
    data = ["abcd", ["efgh"], {"ijk": "lm"}]
    pickled = pickle.dumps(data)
    assert pickle.loads(pickled) == data

    # patched pickle can be used normally
    assert pickle.load(io.BytesIO(pickled)) == data

    # when pickle prohibition is enabled, errors will be raised
    with pytest.raises(ValueError), switch_unpickle():
        pickle.loads(pickled)

    @switch_unpickle
    def limited_func():
        pickle.loads(pickled)

    with pytest.raises(ValueError):
        limited_func()

    @switch_unpickle
    async def limited_async_func():
        pickle.loads(pickled)

    with pytest.raises(ValueError):
        await limited_async_func()

    # patched pickle can be used normally when prohibition is eliminated
    assert pickle.load(io.BytesIO(pickled)) == data
