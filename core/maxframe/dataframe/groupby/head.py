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
import pandas as pd

from ... import opcodes
from ...core import OutputType
from ...serialization import PickleContainer
from ...serialization.serializables import BoolField, DictField, Int64Field
from ...udf import BuiltinFunction
from ...utils import find_objects, pd_release_version
from ..core import IndexValue
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index

_pandas_enable_negative = pd_release_version >= (1, 4, 0)


class GroupByHead(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.GROUPBY_HEAD
    _op_module_ = "dataframe.groupby"

    row_count = Int64Field("row_count", default=5)
    groupby_params = DictField("groupby_params", default=dict())
    enable_negative = BoolField("enable_negative", default=_pandas_enable_negative)

    def has_custom_code(self) -> bool:
        callable_bys = find_objects(
            self.groupby_params.get("by"), types=PickleContainer, checker=callable
        )
        if not callable_bys:
            return False
        return any(not isinstance(fun, BuiltinFunction) for fun in callable_bys)

    def __call__(self, groupby):
        df = groupby
        while df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            df = df.inputs[0]

        selection = groupby.op.groupby_params.pop("selection", None)
        if df.ndim > 1 and selection:
            if isinstance(selection, tuple) and selection not in df.dtypes:
                selection = list(selection)

            result_df = df[selection]
        else:
            result_df = df

        self._output_types = (
            [OutputType.dataframe] if result_df.ndim == 2 else [OutputType.series]
        )

        params = result_df.params
        params["shape"] = (np.nan,) + result_df.shape[1:]
        if isinstance(df.index_value.value, IndexValue.RangeIndex):
            params["index_value"] = parse_index(pd.RangeIndex(-1), df.key)

        return self.new_tileable([df], **params)


def head(groupby, n=5):
    """
    Return first n rows of each group.

    Similar to ``.apply(lambda x: x.head(n))``, but it returns a subset of rows
    from the original Series or DataFrame with original index and order preserved
    (``as_index`` flag is ignored).

    Does not work for negative values of `n`.

    Returns
    -------
    Series or DataFrame

    See Also
    --------
    Series.groupby
    DataFrame.groupby

    Examples
    --------

    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([[1, 2], [1, 4], [5, 6]],
    ...                   columns=['A', 'B'])
    >>> df.groupby('A').head(1).execute()
       A  B
    0  1  2
    2  5  6
    >>> df.groupby('A').head(-1).execute()
    Empty DataFrame
    Columns: [A, B]
    Index: []
    """
    groupby_params = groupby.op.groupby_params.copy()
    groupby_params.pop("as_index", None)

    op = GroupByHead(
        row_count=n,
        groupby_params=groupby_params,
        enable_negative=_pandas_enable_negative,
    )
    return op(groupby)
