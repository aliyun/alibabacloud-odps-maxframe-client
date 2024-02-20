# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

import pandas as pd

from ..core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE
from ..initializer import read_pandas


def test_from_pandas():
    df_data = pd.DataFrame([["a", 1], ["b", 2]], columns=["a", "b"])
    assert isinstance(read_pandas(df_data), DATAFRAME_TYPE)

    s_data = pd.Series(["a", "b"])
    assert isinstance(read_pandas(s_data), SERIES_TYPE)

    idx_data = pd.Index(["a", "b"])
    assert isinstance(read_pandas(idx_data), INDEX_TYPE)
