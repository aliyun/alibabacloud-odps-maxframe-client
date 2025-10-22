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

from typing import List

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import ListField
from ...serialization.serializables.field_type import FieldTypes
from ...utils import make_dtype, make_dtypes
from ..core import DataFrame
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class SeriesFlatJSONOperator(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.FLATJSON

    query_paths = ListField("query_paths", field_type=FieldTypes.string, default=None)

    def __call__(self, series, dtypes):
        if self._output_types[0] == OutputType.series:
            name, dtype = dtypes
            return self.new_series(
                [series],
                shape=series.shape,
                index_value=series.index_value,
                name=name,
                dtype=make_dtype(dtype),
            )
        dtypes = make_dtypes(dtypes)
        return self.new_dataframe(
            [series],
            shape=(series.shape[0], len(dtypes)),
            index_value=series.index_value,
            columns_value=parse_index(dtypes.index, store_data=True),
            dtypes=dtypes,
        )


def series_flatjson(
    series,
    query_paths: List[str],
    dtypes=None,
    dtype=None,
    name: str = None,
) -> DataFrame:
    """
    Flat JSON object in the series to a dataframe according to JSON query.

    Parameters
    ----------
    series : Series
        The series of json strings.

    query_paths: List[str] or str
        The JSON query paths for each generated column. The path format should follow
        [RFC9535](https://datatracker.ietf.org/doc/rfc9535/).

    dtypes : Series, default None
        Specify dtypes of returned DataFrame. Can't work with dtype.

    dtype : numpy.dtype, default None
        Specify dtype of returned Series. Can't work with dtypes.

    name : str, default None
        Specify name of the returned Series.

    Returns
    -------
    DataFrame or Series
        Result of DataFrame when dtypes specified, else Series.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> import pandas as pd
    >>> s = md.Series(
    ...     [
    ...         '{"age": 24, "gender": "male", "graduated": false}',
    ...         '{"age": 25, "gender": "female", "graduated": true}',
    ...     ]
    ... )
    >>> s.execute()
    0    {"age": 24, "gender": "male", "graduated": false}
    1    {"age": 25, "gender": "female", "graduated": true}
    dtype: object

    >>> df = s.mf.flatjson(
    ...    ["$.age", "$.gender", "$.graduated"],
    ...    dtypes=pd.Series(["int32", "object", "bool"], index=["age", "gender", "graduated"]),
    ... )
    >>> df.execute()
        age  gender  graduated
    0   24    male       True
    1   25  female       True

    >>> s2 = s.mf.flatjson("$.age", name="age", dtype="int32")
    >>> s2.execute()
    0    24
    1    25
    Name: age, dtype: int32
    """
    if isinstance(query_paths, str):
        query_paths = [query_paths]
    if dtypes is not None and dtype is not None:
        raise ValueError("Both dtypes and dtype cannot be specified at the same time.")
    if dtype is not None:
        if len(query_paths) != 1:
            raise ValueError("query_paths should have only one path if dtype is set")
        output_type = OutputType.series
    elif dtypes is not None:
        if len(dtypes) != len(query_paths):
            raise ValueError("query_paths and dtypes should have same length")
        output_type = OutputType.dataframe
    else:
        raise ValueError("dtypes or dtype should be specified")

    dtypes = (name, dtype) if dtype is not None else dtypes
    return SeriesFlatJSONOperator(query_paths=query_paths, _output_types=[output_type])(
        series, dtypes
    )
