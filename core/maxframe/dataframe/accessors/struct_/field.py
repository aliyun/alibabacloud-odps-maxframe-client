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

from ....lib.dtypes_extension import ArrowDtype
from .core import SeriesStructMethod


def struct_field(series, name_or_index):
    """
    Extract a child field of a struct as a Series.

    Parameters
    ----------
    name_or_index : str | bytes | int | expression | list
        Name or index of the child field to extract.

        For list-like inputs, this will index into a nested
        struct.

    Returns
    -------
    pandas.Series
        The data corresponding to the selected child field.

    See Also
    --------
    Series.struct.explode : Return all child fields as a DataFrame.

    Notes
    -----
    The name of the resulting Series will be set using the following
    rules:

    - For string, bytes, or integer `name_or_index` (or a list of these, for
      a nested selection), the Series name is set to the selected
      field's name.
    - For a :class:`pyarrow.compute.Expression`, this is set to
      the string form of the expression.
    - For list-like `name_or_index`, the name will be set to the
      name of the final field selected.

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

    Extract by field name.

    >>> s.struct.field("project").execute()
    0    pandas
    1    pandas
    2     numpy
    Name: project, dtype: string[pyarrow]

    Extract by field index.

    >>> s.struct.field(0).execute()
    0    1
    1    2
    2    1
    Name: version, dtype: int64[pyarrow]

    For nested struct types, you can pass a list of values to index
    multiple levels:

    >>> version_type = pa.struct([
    ...     ("major", pa.int64()),
    ...     ("minor", pa.int64()),
    ... ])
    >>> s = md.Series(
    ...     [
    ...         {"version": {"major": 1, "minor": 5}, "project": "pandas"},
    ...         {"version": {"major": 2, "minor": 1}, "project": "pandas"},
    ...         {"version": {"major": 1, "minor": 26}, "project": "numpy"},
    ...     ],
    ...     dtype=pd.ArrowDtype(pa.struct(
    ...         [("version", version_type), ("project", pa.string())]
    ...     ))
    ... )
    >>> s.struct.field(["version", "minor"]).execute()
    0     5
    1     1
    2    26
    Name: minor, dtype: int64[pyarrow]
    >>> s.struct.field([0, 0]).execute()
    0    1
    1    2
    2    1
    Name: major, dtype: int64[pyarrow]
    """
    op = SeriesStructMethod(
        method="field",
        method_kwargs={"name_or_index": name_or_index},
    )
    names = name_or_index if isinstance(name_or_index, list) else [name_or_index]
    arrow_type = series.dtype.pyarrow_dtype
    arrow_name = None
    for n in names:
        arrow_name = arrow_type[n].name
        arrow_type = arrow_type[n].type
    return op(series, dtype=ArrowDtype(arrow_type), name=arrow_name)
