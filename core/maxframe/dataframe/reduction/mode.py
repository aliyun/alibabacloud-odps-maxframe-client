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

import numpy as np

from ... import opcodes
from ...core import OutputType, get_output_types
from ...serialization.serializables import BoolField, Int32Field
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index, validate_axis


class DataFrameMode(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.MODE

    axis = Int32Field("axis", default=None)
    numeric_only = BoolField("numeric_only", default=False)
    dropna = BoolField("dropna", default=True)
    combine_size = Int32Field("combine_size", default=None)

    def __call__(self, in_obj):
        self._output_types = get_output_types(in_obj)
        params = in_obj.params
        shape = list(in_obj.shape)
        shape[self.axis] = np.nan
        params["shape"] = tuple(shape)

        if self.axis == 0:
            pd_idx = in_obj.index_value.to_pandas()[:0]
            params["index_value"] = parse_index(pd_idx)
        else:
            pd_idx = in_obj.columns_value.to_pandas()[:0]
            params["columns_value"] = parse_index(pd_idx)
            params["dtypes"] = None
        return self.new_tileable([in_obj], **params)


def mode_dataframe(df, axis=0, numeric_only=False, dropna=True, combine_size=None):
    """
    Get the mode(s) of each element along the selected axis.
    The mode of a set of values is the value that appears most often.
    It can be multiple values.
    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to iterate over while searching for the mode:
        * 0 or 'index' : get mode of each column
        * 1 or 'columns' : get mode of each row.
    numeric_only : bool, default False
        If True, only apply to numeric columns.
    dropna : bool, default True
        Don't consider counts of NaN/NaT.
    Returns
    -------
    DataFrame
        The modes of each column or row.
    See Also
    --------
    Series.mode : Return the highest frequency value in a Series.
    Series.value_counts : Return the counts of values in a Series.
    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([('bird', 2, 2),
    ...                    ('mammal', 4, mt.nan),
    ...                    ('arthropod', 8, 0),
    ...                    ('bird', 2, mt.nan)],
    ...                   index=('falcon', 'horse', 'spider', 'ostrich'),
    ...                   columns=('species', 'legs', 'wings'))
    >>> df.execute()
               species  legs  wings
    falcon        bird     2    2.0
    horse       mammal     4    NaN
    spider   arthropod     8    0.0
    ostrich       bird     2    NaN
    By default, missing values are not considered, and the mode of wings
    are both 0 and 2. Because the resulting DataFrame has two rows,
    the second row of ``species`` and ``legs`` contains ``NaN``.
    >>> df.mode().execute()
      species  legs  wings
    0    bird   2.0    0.0
    1     NaN   NaN    2.0
    Setting ``dropna=False`` ``NaN`` values are considered and they can be
    the mode (like for wings).
    >>> df.mode(dropna=False).execute()
      species  legs  wings
    0    bird     2    NaN
    Setting ``numeric_only=True``, only the mode of numeric columns is
    computed, and columns of other types are ignored.
    >>> df.mode(numeric_only=True).execute()
       legs  wings
    0   2.0    0.0
    1   NaN    2.0
    To compute the mode over columns and not rows, use the axis parameter:
    >>> df.mode(axis='columns', numeric_only=True).execute()
               0    1
    falcon   2.0  NaN
    horse    4.0  NaN
    spider   0.0  8.0
    ostrich  2.0  NaN
    """
    op = DataFrameMode(
        axis=validate_axis(axis),
        numeric_only=numeric_only,
        dropna=dropna,
        combine_size=combine_size,
        output_types=[OutputType.dataframe],
    )
    return op(df)


def mode_series(series, dropna=True, combine_size=None):
    """
    Return the mode(s) of the Series.
    The mode is the value that appears most often. There can be multiple modes.
    Always returns Series even if only one value is returned.
    Parameters
    ----------
    dropna : bool, default True
        Don't consider counts of NaN/NaT.
    Returns
    -------
    Series
        Modes of the Series in sorted order.
    """
    op = DataFrameMode(
        axis=0,
        dropna=dropna,
        combine_size=combine_size,
        output_types=[OutputType.series],
    )
    return op(series)
