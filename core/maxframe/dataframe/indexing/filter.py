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

import re

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import get_output_types
from ...serialization.serializables import Int32Field, ListField, StringField
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class DataFrameFilter(DataFrameOperatorMixin, DataFrameOperator):
    _op_type_ = opcodes.DATAFRAME_FILTER

    items = ListField("items", default=None)
    like = StringField("like", default=None)
    regex = StringField("regex", default=None)
    axis = Int32Field("axis", default=None)

    def __call__(self, df_or_series):
        self._output_types = get_output_types(df_or_series)

        # Get axis labels to filter
        if self.axis == 0:
            # Filter by index
            labels = df_or_series.index_value.to_pandas()
        else:
            # Filter by columns (DataFrame only)
            if not hasattr(df_or_series, "columns"):
                raise ValueError("axis=1 (columns) not valid for Series")
            labels = df_or_series.columns_value.to_pandas()

        # Apply filter criteria
        filtered_labels = self._apply_filter_criteria(labels)

        # Calculate output shape and metadata
        out_params = self._calculate_output_metadata(df_or_series, filtered_labels)
        return self.new_tileable([df_or_series], **out_params)

    def _apply_filter_criteria(self, labels):
        """Apply filter criteria to labels"""
        if self.items is not None:
            # Exact match filter
            return [label for label in labels if label in self.items]
        elif self.like is not None:
            # Substring match filter
            return [label for label in labels if self.like in str(label)]
        elif self.regex is not None:
            # Regex match filter
            pattern = re.compile(self.regex)
            return [label for label in labels if pattern.search(str(label))]
        else:
            return list(labels)

    def _calculate_output_metadata(self, input_tileable, filtered_labels):
        input_shape = input_tileable.shape

        out_params = input_tileable.params
        if self.axis == 0:
            out_params["shape"] = (len(filtered_labels) or np.nan,) + input_shape[1:]
            out_params["index_value"] = parse_index(
                pd.Index(filtered_labels), input_tileable.index_value
            )
        else:
            out_params["shape"] = (input_shape[0], len(filtered_labels))
            out_params["columns_value"] = parse_index(
                input_tileable.dtypes[filtered_labels].index, store_data=True
            )
        return out_params


def filter_dataframe(df_or_series, items=None, like=None, regex=None, axis=None):
    """
    Subset the dataframe rows or columns according to the specified index labels.

    Note that this routine does not filter a dataframe on its
    contents. The filter is applied to the labels of the index.

    Parameters
    ----------
    items : list-like
        Keep labels from axis which are in items.
    like : str
        Keep labels from axis for which "like in label == True".
    regex : str (regular expression)
        Keep labels from axis for which re.search(regex, label) == True.
    axis : {0 or 'index', 1 or 'columns', None}, default None
        The axis to filter on, expressed either as an index (int)
        or axis name (str). By default this is the info axis, 'columns' for
        DataFrame. For `Series` this parameter is unused and defaults to `None`.

    Returns
    -------
    same type as input object

    See Also
    --------
    DataFrame.loc : Access a group of rows and columns
        by label(s) or a boolean array.

    Notes
    -----
    The ``items``, ``like``, and ``regex`` parameters are
    enforced to be mutually exclusive.

    ``axis`` defaults to the info axis that is used when indexing
    with ``[]``.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame(mt.array(([1, 2, 3], [4, 5, 6])),
    ...                   index=['mouse', 'rabbit'],
    ...                   columns=['one', 'two', 'three'])
    >>> df.execute()
            one  two  three
    mouse     1    2      3
    rabbit    4    5      6

    >>> # select columns by name
    >>> df.filter(items=['one', 'three']).execute()
             one  three
    mouse     1      3
    rabbit    4      6

    >>> # select columns by regular expression
    >>> df.filter(regex='e$', axis=1).execute()
             one  three
    mouse     1      3
    rabbit    4      6

    >>> # select rows containing 'bbi'
    >>> df.filter(like='bbi', axis=0).execute()
             one  two  three
    rabbit    4    5      6
    """
    if axis is None:
        # For Series, axis is always 0 (index)
        # For DataFrame, default is 1 (columns)
        if hasattr(df_or_series, "columns"):
            axis = 1  # DataFrame - filter columns by default
        else:
            axis = 0  # Series - filter index

    param_count = sum(x is not None for x in [items, like, regex])
    if param_count == 0:
        raise TypeError("Must pass either `items`, `like`, or `regex`")
    if param_count > 1:
        raise TypeError(
            "keyword arguments `items`, `like`, `regex` are mutually exclusive"
        )
    op = DataFrameFilter(items=items, like=like, regex=regex, axis=axis)
    return op(df_or_series)
