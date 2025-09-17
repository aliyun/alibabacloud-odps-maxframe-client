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

from ... import opcodes
from ...core import get_output_types
from ...serialization.serializables import AnyField, Int32Field
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index, validate_axis


class DataFrameReorderLevels(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.REORDER_LEVELS

    order = AnyField("order")
    axis = Int32Field("axis", default=0)

    def __call__(self, df_or_series):
        # Determine output type
        self._output_types = get_output_types(df_or_series)

        if self.axis == 0:
            src_idx_value = df_or_series.index_value
        else:
            src_idx_value = df_or_series.columns_value

        # Create reordered index
        pd_index = src_idx_value.to_pandas()
        if not isinstance(pd_index, pd.MultiIndex):
            raise ValueError("reorder_levels can only be used with MultiIndex")
        pd_index = pd_index.reorder_levels(self.order)

        params = df_or_series.params
        if self.axis == 0:
            params["index_value"] = parse_index(pd_index)
        else:
            params["columns_value"] = parse_index(pd_index, store_data=True)
        return self.new_tileable([df_or_series], **params)


def _reorder_levels(df_or_series, order, axis=0):
    axis = validate_axis(axis, df_or_series)
    op = DataFrameReorderLevels(order=order, axis=axis)
    return op(df_or_series)


def df_reorder_levels(df, order, axis=0):
    """
    Rearrange index levels using input order. May not drop or duplicate levels.

    Parameters
    ----------
    order : list of int or list of str
        List representing new level order. Reference level by number
        (position) or by key (label).
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Where to reorder levels.

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> data = {
    ...     "class": ["Mammals", "Mammals", "Reptiles"],
    ...     "diet": ["Omnivore", "Carnivore", "Carnivore"],
    ...     "species": ["Humans", "Dogs", "Snakes"],
    ... }
    >>> df = md.DataFrame(data, columns=["class", "diet", "species"])
    >>> df = df.set_index(["class", "diet"])
    >>> df.execute()
                                      species
    class      diet
    Mammals    Omnivore                Humans
               Carnivore                 Dogs
    Reptiles   Carnivore               Snakes

    Let's reorder the levels of the index:

    >>> df.reorder_levels(["diet", "class"]).execute()
                                      species
    diet      class
    Omnivore  Mammals                  Humans
    Carnivore Mammals                    Dogs
              Reptiles                 Snakes
    """
    return _reorder_levels(df, order, axis=axis)


def series_reorder_levels(series, order):
    """
    Rearrange index levels using input order.

    May not drop or duplicate levels.

    Parameters
    ----------
    order : list of int representing new level order
        Reference level by number or key.

    Returns
    -------
    type of caller (new object)

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> arrays = [mt.array(["dog", "dog", "cat", "cat", "bird", "bird"]),
    ...           mt.array(["white", "black", "white", "black", "white", "black"])]
    >>> s = md.Series([1, 2, 3, 3, 5, 2], index=arrays)
    >>> s.execute()
    dog   white    1
          black    2
    cat   white    3
          black    3
    bird  white    5
          black    2
    dtype: int64
    >>> s.reorder_levels([1, 0]).execute()
    white  dog     1
    black  dog     2
    white  cat     3
    black  cat     3
    white  bird    5
    black  bird    2
    dtype: int64
    """
    return _reorder_levels(series, order)
