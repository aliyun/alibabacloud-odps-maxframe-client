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

import pandas as pd

from ...udf import builtin_function


@builtin_function
def _tailor_unique(series):
    if not series.is_unique:
        return pd.Series([], name=series.name, dtype=series.dtype)
    return series


def is_unique(series):
    """
    Return boolean if values in the object are unique.

    Returns
    -------
    bool

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series([1, 2, 3])
    >>> s.is_unique.execute()
    True

    >>> s = md.Series([1, 2, 3, 1])
    >>> s.is_unique.execute()
    False
    """
    from ... import tensor as mt

    return mt.equal(
        series.mf.apply_chunk(_tailor_unique, dtype=series.dtype).nunique(),
        mt.shape(series)[0],
    )
