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

from ....lib.dtypes_extension import ArrowDtype


def struct_dtypes(series):
    """
    Return the dtype object of each child field of the struct.

    Returns
    -------
    pandas.Series
        The data type of each child field.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> import pandas as pd
    >>> import pyarrow as pa
    >>> s = md.Series(
    ...     [
    ...         {"version": 1, "project": "pandas"},
    ...         {"version": 2, "project": "pandas"},
    ...         {"version": 1, "project": "numpy"},
    ...     ],
    ...     dtype=pd.ArrowDtype(pa.struct(
    ...         [("version", pa.int64()), ("project", pa.string())]
    ...     ))
    ... )
    >>> s.struct.dtypes.execute()
    version     int64[pyarrow]
    project    string[pyarrow]
    dtype: object
    """
    pa_type = series.dtype.pyarrow_dtype
    fields = [pa_type[idx] for idx in range(pa_type.num_fields)]
    dtypes_list = [ArrowDtype(ft.type) for ft in fields]
    dt_name = [ft.name for ft in fields]
    return pd.Series(dtypes_list, index=dt_name)
