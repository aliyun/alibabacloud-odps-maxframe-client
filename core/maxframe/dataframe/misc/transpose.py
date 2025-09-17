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
from ...core import OutputType
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class DataFrameTranspose(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.TRANSPOSE

    def __init__(self, **kw):
        super().__init__(**kw)
        self.output_types = [OutputType.dataframe]

    def __call__(self, args):
        arg = args[0]
        new_shape = arg.shape[::-1]
        columns_value = arg.index_value
        index_value = parse_index(arg.dtypes.index)

        if not arg.index_value.has_value:
            dtypes = None
        else:
            from pandas.core.dtypes.cast import find_common_type

            dtype = find_common_type(list(arg.dtypes))
            pd_index = arg.index_value.to_pandas()
            dtypes = pd.Series([dtype] * len(pd_index), index=pd_index)

        return self.new_dataframe(
            [arg],
            shape=new_shape,
            dtypes=dtypes,
            columns_value=columns_value,
            index_value=index_value,
        )


def transpose(*args):
    """
    Transpose index and columns.

    Reflect the DataFrame over its main diagonal by writing rows as columns
    and vice-versa. The property :attr:`.T` is an accessor to the method
    :meth:`transpose`.

    Parameters
    ----------
    *args : tuple, optional
            Accepted for compatibility with NumPy.

    Returns
    -------
    DataFrame
        The transposed DataFrame.

    See Also
    --------
    numpy.transpose : Permute the dimensions of a given array.

    Notes
    -----
    Transposing a DataFrame with mixed dtypes will result in a homogeneous
    DataFrame with the `object` dtype.

    Examples
    --------
    **Square DataFrame with homogeneous dtype**

    >>> import maxframe.dataframe as md
    >>> d1 = {'col1': [1, 2], 'col2': [3, 4]}
    >>> df1 = md.DataFrame(data=d1).execute()
    >>> df1
        col1  col2
    0     1     3
    1     2     4

    >>> df1_transposed = df1.T.execute() # or df1.transpose().execute()
    >>> df1_transposed
          0  1
    col1  1  2
    col2  3  4

    When the dtype is homogeneous in the original DataFrame, we get a
    transposed DataFrame with the same dtype:

    >>> df1.dtypes
    col1    int64
    col2    int64
    dtype: object

    >>> df1_transposed.dtypes
    0    int64
    1    int64
    dtype: object

    **Non-square DataFrame with mixed dtypes**

    >>> d2 = {'name': ['Alice', 'Bob'],
    ...       'score': [9.5, 8],
    ...       'employed': [False, True],
    ...       'kids': [0, 0]}
    >>> df2 = md.DataFrame(data=d2).execute()
    >>> df2
        name  score  employed  kids
    0  Alice    9.5     False     0
    1    Bob    8.0      True     0

    >>> df2_transposed = df2.T.execute() # or df2.transpose().execute()
    >>> df2_transposed
                  0     1
    name      Alice   Bob
    score       9.5   8.0
    employed  False  True
    kids          0     0

    When the DataFrame has mixed dtypes, we get a transposed DataFrame with
    the `object` dtype:

    >>> df2.dtypes
    name         object
    score       float64
    employed       bool
    kids          int64
    dtype: object

    >>> df2_transposed.dtypes
    0    object
    1    object
    dtype: object
    """
    op = DataFrameTranspose()
    return op(args)
