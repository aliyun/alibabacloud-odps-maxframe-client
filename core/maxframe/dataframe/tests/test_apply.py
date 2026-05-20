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
import pandas as pd

import maxframe.dataframe as md


def test_dataframe_apply_axis1_with_dict_annotation():
    """Test df.apply with Series dict annotation"""
    df = md.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def process_row(row) -> pd.Series[{"x": int, "y": str}]:  # noqa: F821
        return pd.Series({"x": row["a"] * 2, "y": str(row["b"])})

    result = df.apply(process_row, axis=1)
    assert isinstance(result, md.DataFrame)
    assert list(result.dtypes.index) == ["x", "y"]
    assert result.dtypes["x"] == np.dtype(int)
