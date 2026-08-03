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

import pandas as pd

from maxframe.liteframe.datasource.utils import infer_dtypes_from_pandas
from maxframe.liteframe.operators.source import LiteFrameFromLocal
from maxframe.liteframe.utils import normalize_col_name, normalize_dtypes_index


def from_local_df(df, chunk_size=None, gpu=None, session=None, default_index_type=None):
    """Create a LiteFrame from a local pandas DataFrame or similar object."""
    # Lowercase all column names so internal storage is case-insensitive.
    # This must be done for both pandas and non-pandas inputs.
    lowered_cols = [normalize_col_name(c) for c in df.columns]
    # Check for duplicate column names after normalization
    seen = set()
    for c in lowered_cols:
        if c in seen:
            raise ValueError(f"Duplicate column name after case normalization: '{c}'")
        seen.add(c)
    df.columns = lowered_cols

    if isinstance(df, pd.DataFrame):
        dtypes = infer_dtypes_from_pandas(df)
    else:
        dtypes = pd.Series({col: dtype for col, dtype in zip(lowered_cols, df.dtypes)})
        dtypes = normalize_dtypes_index(dtypes)

    op = LiteFrameFromLocal(
        data=df,
        dtypes=dtypes,
        default_index_type=default_index_type,
        gpu=gpu,
    )
    shape = df.shape
    return op._new_liteframe_from_source(shape=shape, raw_chunk_size=chunk_size)
