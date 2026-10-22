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

from maxframe.lib.dtypes_extension import is_list_dtype
from maxframe.liteframe.accessors.list_.core import (
    ListMethodBaseHandler,
    LiteFrameListExpr,
    list_method_to_handlers,
)
from maxframe.liteframe.accessors.list_.flatten import flatten_list_column
from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _resolve_column_expr,
)

_method_docs = {
    "len": """
    Get the length of each list in the column.

    Returns
    -------
    LiteFrame
        Single-column LiteFrame with ``int64`` dtype; ``None`` lists
        yield ``None``.
    """,
    "get": """
    Get the value at an index of each list in the column.

    Out-of-range indices yield ``None`` rather than raising.

    Parameters
    ----------
    query_index : int
        The index to look up in each list.

    Returns
    -------
    LiteFrame
        Single-column LiteFrame with the list's value dtype.
    """,
    "__getitem__": """
    Get the value at an index of each list in the column.

    Raises ``IndexError`` for out-of-range indices.

    Parameters
    ----------
    query_index : int
        The index to look up in each list.

    Returns
    -------
    LiteFrame
        Single-column LiteFrame with the list's value dtype.
    """,
}


class ListAccessor:
    """
    Vectorized list functions for LiteFrame.

    Provides pandas-style .list accessor for single-column LiteFrame with
    Arrow list dtype. Multi-column LiteFrames raise ValueError on accessor
    access.

    Parameters
    ----------
    liteframe : LiteFrame
        Single-column LiteFrame with an Arrow list dtype to wrap.

    Examples
    --------
    >>> df = LiteFrame({"lst": [[1, 2], [3, 4]]})
    >>> df.list.len().execute()
       lst
    0    2
    1    2
    """

    def __init__(self, liteframe):
        if len(liteframe.columns) != 1:
            raise ValueError("Cannot apply .list accessor to multi-column LiteFrame")
        dtype = liteframe.dtypes.iloc[0]
        if not is_list_dtype(dtype):
            raise AttributeError("Can only use .list accessor with list values")
        self._liteframe = liteframe

    @classmethod
    def _gen_func(cls, method):
        """Generate method wrapper with dtype inference."""

        def _inner(self, *args, **kwargs):
            handler = list_method_to_handlers.get(method, ListMethodBaseHandler)
            dtype = handler.infer_dtype(method, self._liteframe.dtypes.iloc[0])

            col_name = self._liteframe.columns[0]
            col_dtype = self._liteframe.dtypes.iloc[0]
            col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)

            expr = LiteFrameListExpr(
                operand=col_expr,
                method=method,
                args=args,
                kwargs=kwargs,
                dtype=dtype,
            )

            named_expr = expr.rename(self._liteframe.columns[0])
            return _build_fused_projection(self._liteframe, [named_expr])

        _inner.__doc__ = _method_docs.get(method)
        return _inner

    def flatten(self):
        """Flatten a list column into one row per element.

        Mirrors ``pandas.Series.list.flatten``: each element of a list
        becomes its own row.  Empty lists and missing values are dropped.
        The result keeps the column name and uses the list's value type.

        Returns
        -------
        LiteFrame
            Single-column LiteFrame with one row per list element, using
            the list's value dtype and keeping the original column name.
        """
        return flatten_list_column(self._liteframe)

    @classmethod
    def _register(cls, method):
        """Register method to accessor class."""
        setattr(cls, method, cls._gen_func(method))
