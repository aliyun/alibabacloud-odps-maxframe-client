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

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import EntityData
from ...serialization.serializables import AnyField, FieldTypes, KeyField, ListField
from ..core import SERIES_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_df, parse_index


class DataFrameDescribe(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.DESCRIBE

    input = KeyField("input", default=None)
    percentiles = ListField("percentiles", FieldTypes.float64, default=None)
    include = AnyField("include", default=None)
    exclude = AnyField("exclude", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _set_inputs(cls, op: "DataFrameDescribe", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.input = op._inputs[0]

    def __call__(self, df_or_series):
        if isinstance(df_or_series, SERIES_TYPE):
            test_series = pd.Series([], dtype=df_or_series.dtype).describe(
                percentiles=self.percentiles,
                include=self.include,
                exclude=self.exclude,
            )
            return self.new_series(
                [df_or_series],
                shape=(len(test_series),),
                dtype=test_series.dtype,
                index_value=parse_index(test_series.index, store_data=True),
            )
        else:
            test_inp_df = build_df(df_or_series)
            test_df = test_inp_df.describe(
                percentiles=self.percentiles,
                include=self.include,
                exclude=self.exclude,
            )
            if len(self.percentiles) == 0:
                # specify percentiles=False
                # Note: unlike pandas that False is illegal value for percentiles,
                # MaxFrame DataFrame allows user to specify percentiles=False
                # to skip computation about percentiles
                test_df.drop(["50%"], axis=0, inplace=True)
            return self.new_dataframe(
                [df_or_series],
                shape=test_df.shape,
                dtypes=test_df.dtypes,
                index_value=parse_index(test_df.index, store_data=True),
                columns_value=parse_index(test_df.columns, store_data=True),
            )


def describe(df_or_series, percentiles=None, include=None, exclude=None):
    """
    Generate descriptive statistics.

    Descriptive statistics include those that summarize the central
    tendency, dispersion and shape of a
    dataset's distribution, excluding ``NaN`` values.

    Analyzes both numeric and object series, as well
    as ``DataFrame`` column sets of mixed data types. The output
    will vary depending on what is provided. Refer to the notes
    below for more detail.

    Parameters
    ----------
    percentiles : list-like of numbers, optional
        The percentiles to include in the output. All should
        fall between 0 and 1. The default is
        ``[.25, .5, .75]``, which returns the 25th, 50th, and
        75th percentiles.
    include : 'all', list-like of dtypes or None (default), optional
        A white list of data types to include in the result. Ignored
        for ``Series``. Here are the options:

        - 'all' : All columns of the input will be included in the output.
        - A list-like of dtypes : Limits the results to the
          provided data types.
          To limit the result to numeric types submit
          ``numpy.number``. To limit it instead to object columns submit
          the ``numpy.object`` data type. Strings
          can also be used in the style of
          ``select_dtypes`` (e.g. ``df.describe(include=['O'])``).
        - None (default) : The result will include all numeric columns.
    exclude : list-like of dtypes or None (default), optional,
        A black list of data types to omit from the result. Ignored
        for ``Series``. Here are the options:

        - A list-like of dtypes : Excludes the provided data types
          from the result. To exclude numeric types submit
          ``numpy.number``. To exclude object columns submit the data
          type ``numpy.object``. Strings can also be used in the style of
          ``select_dtypes`` (e.g. ``df.describe(exclude=['O'])``).
        - None (default) : The result will exclude nothing.

    Returns
    -------
    Series or DataFrame
        Summary statistics of the Series or Dataframe provided.

    See Also
    --------
    DataFrame.count: Count number of non-NA/null observations.
    DataFrame.max: Maximum of the values in the object.
    DataFrame.min: Minimum of the values in the object.
    DataFrame.mean: Mean of the values.
    DataFrame.std: Standard deviation of the observations.
    DataFrame.select_dtypes: Subset of a DataFrame including/excluding
        columns based on their dtype.

    Notes
    -----
    For numeric data, the result's index will include ``count``,
    ``mean``, ``std``, ``min``, ``max`` as well as lower, ``50`` and
    upper percentiles. By default the lower percentile is ``25`` and the
    upper percentile is ``75``. The ``50`` percentile is the
    same as the median.

    For object data (e.g. strings or timestamps), the result's index
    will include ``count``, ``unique``, ``top``, and ``freq``. The ``top``
    is the most common value. The ``freq`` is the most common value's
    frequency. Timestamps also include the ``first`` and ``last`` items.

    If multiple object values have the highest count, then the
    ``count`` and ``top`` results will be arbitrarily chosen from
    among those with the highest count.

    For mixed data types provided via a ``DataFrame``, the default is to
    return only an analysis of numeric columns. If the dataframe consists
    only of object data without any numeric columns, the default is to
    return an analysis of object columns. If ``include='all'`` is provided
    as an option, the result will include a union of attributes of each type.

    The `include` and `exclude` parameters can be used to limit
    which columns in a ``DataFrame`` are analyzed for the output.
    The parameters are ignored when analyzing a ``Series``.

    Examples
    --------
    Describing a numeric ``Series``.

    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> s = md.Series([1, 2, 3])
    >>> s.describe().execute()
    count    3.0
    mean     2.0
    std      1.0
    min      1.0
    25%      1.5
    50%      2.0
    75%      2.5
    max      3.0
    dtype: float64

    Describing a ``DataFrame``. By default only numeric fields
    are returned.

    >>> df = md.DataFrame({'numeric': [1, 2, 3],
    ...                    'object': ['a', 'b', 'c']
    ...                    })
    >>> df.describe().execute()
           numeric
    count      3.0
    mean       2.0
    std        1.0
    min        1.0
    25%        1.5
    50%        2.0
    75%        2.5
    max        3.0

    Describing all columns of a ``DataFrame`` regardless of data type.

    >>> df.describe(include='all').execute()  # doctest: +SKIP.execute()
           numeric object
    count      3.0      3
    unique     NaN      3
    top        NaN      a
    freq       NaN      1
    mean       2.0    NaN
    std        1.0    NaN
    min        1.0    NaN
    25%        1.5    NaN
    50%        2.0    NaN
    75%        2.5    NaN
    max        3.0    NaN

    Describing a column from a ``DataFrame`` by accessing it as
    an attribute.

    >>> df.numeric.describe().execute()
    count    3.0
    mean     2.0
    std      1.0
    min      1.0
    25%      1.5
    50%      2.0
    75%      2.5
    max      3.0
    Name: numeric, dtype: float64

    Including only numeric columns in a ``DataFrame`` description.

    >>> df.describe(include=[mt.number]).execute()
           numeric
    count      3.0
    mean       2.0
    std        1.0
    min        1.0
    25%        1.5
    50%        2.0
    75%        2.5
    max        3.0

    Including only string columns in a ``DataFrame`` description.

    >>> df.describe(include=[object]).execute()  # doctest: +SKIP.execute()
           object
    count       3
    unique      3
    top         a
    freq        1
    """
    # fixme add support for categorical columns once implemented
    if percentiles is False:
        percentiles = []
    elif percentiles is None:
        percentiles = [0.25, 0.5, 0.75]
    else:
        percentiles = list(percentiles)
        if percentiles is not None:
            for p in percentiles:
                if p < 0 or p > 1:
                    raise ValueError(
                        "percentiles should all be in the interval [0, 1]. "
                        "Try [{0:.3f}] instead.".format(p / 100)
                    )
        # median should always be included
        if 0.5 not in percentiles:
            percentiles.append(0.5)
        percentiles = np.asarray(percentiles)

        # sort and check for duplicates
        unique_pcts = np.unique(percentiles)
        if len(unique_pcts) < len(percentiles):
            raise ValueError("percentiles cannot contain duplicates")
        percentiles = unique_pcts.tolist()

    op = DataFrameDescribe(percentiles=percentiles, include=include, exclude=exclude)
    return op(df_or_series)
