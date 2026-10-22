# Copyright 1999-2026 Alibaba Group Holding Ltd.
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

"""``LiteFrame.list.flatten`` implementation.

Builds on the shared ``_explode_list_row`` ``@builtin_function`` UDF from
:mod:`maxframe.liteframe.misc.explode` via :func:`LiteFrame.flat_apply`, so
the :class:`LiteFrameFunctional` operator does not treat it as custom code
(no isolation environment): a built-in function is reloaded by name in the
worker instead of being pickled.
"""

from maxframe.lib.dtypes_extension import is_list_dtype
from maxframe.liteframe.misc.explode import _explode_list_row, _list_value_dtype


def flatten_list_column(liteframe):
    """``LiteFrame.list.flatten``: flatten a single list column into rows.

    Mirrors ``pandas.Series.list.flatten``: each element of a list becomes its
    own row.  Empty lists and missing values are dropped (unlike ``explode``,
    which keeps them as NA rows).  The result keeps the single column name and
    uses the list's ``value_type``.
    """
    if len(liteframe.columns) != 1:
        raise ValueError("list.flatten requires a single-column LiteFrame")
    column = liteframe.columns[0]
    col_dtype = liteframe.dtypes.iloc[0]
    if not is_list_dtype(col_dtype):
        raise AttributeError("Can only use .list.flatten with list values")
    value_dtype = _list_value_dtype(col_dtype)
    return liteframe.flat_apply(
        _explode_list_row,
        dtypes={column: value_dtype},
        subset=[column],
        columns=[column],
        skip_empty=True,
    )
