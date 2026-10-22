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

from maxframe.lib.dtypes_extension import is_map_dtype
from maxframe.liteframe.accessors.dict_.core import (
    DictMethodBaseHandler,
    LiteFrameDictExpr,
    dict_method_to_handlers,
)
from maxframe.liteframe.arithmetic.core import (
    _build_fused_projection,
    _resolve_column_expr,
)

_method_docs = {
    "len": """
    Get the number of items in each dict in the column.

    Returns
    -------
    LiteFrame
        Single-column LiteFrame with ``int64`` dtype; ``None`` dicts
        yield ``None``.
    """,
    "get": """
    Get the value for a key in each dict in the column.

    Returns ``default_value`` when the key is absent.

    Parameters
    ----------
    query_key : Any
        The key to look up, matching the dict's key type.
    default_value : Any, optional
        Value to return when the key is absent, by default ``None``.

    Returns
    -------
    LiteFrame
        Single-column LiteFrame with the dict's value dtype.
    """,
    "__getitem__": """
    Get the value for a key in each dict in the column.

    Raises ``KeyError`` when the key is absent.

    Parameters
    ----------
    query_key : Any
        The key to look up, matching the dict's key type.

    Returns
    -------
    LiteFrame
        Single-column LiteFrame with the dict's value dtype.
    """,
    "remove": """
    Remove a key from each dict in the column.

    Parameters
    ----------
    query_key : Any
        The key to remove, matching the dict's key type.
    ignore_key_error : bool, optional, default False
        When ``True``, missing keys are silently skipped.  When
        ``False``, a missing key raises ``KeyError``.  ``None`` dicts
        are always skipped.

    Returns
    -------
    LiteFrame
        Single-column LiteFrame with the same map dtype.

    Raises
    ------
    KeyError
        If ``query_key`` is absent in a dict and ``ignore_key_error``
        is ``False``.
    """,
    "contains": """
    Check whether a key is present in each dict in the column.

    Parameters
    ----------
    query_key : Any
        The key to check, matching the dict's key type.

    Returns
    -------
    LiteFrame
        Single-column LiteFrame with ``bool`` dtype; ``None`` dicts
        yield ``None``.
    """,
}


class DictAccessor:
    """
    Vectorized dict functions for LiteFrame.

    Provides pandas-style .dict accessor for single-column LiteFrame with
    Arrow map dtype. Multi-column LiteFrames raise ValueError on accessor
    access.

    Parameters
    ----------
    liteframe : LiteFrame
        Single-column LiteFrame with an Arrow map dtype to wrap.

    Examples
    --------
    >>> df = LiteFrame({"d": [("k1", 1), ("k2", 2)]})
    >>> df.dict.len().execute()
       d
    0  2
    """

    def __init__(self, liteframe):
        if len(liteframe.columns) != 1:
            raise ValueError("Cannot apply .dict accessor to multi-column LiteFrame")
        dtype = liteframe.dtypes.iloc[0]
        if not is_map_dtype(dtype):
            raise AttributeError("Can only use .dict accessor with dict values")
        self._liteframe = liteframe

    @classmethod
    def _gen_func(cls, method):
        """Generate method wrapper with dtype inference."""

        def _inner(self, *args, **kwargs):
            handler = dict_method_to_handlers.get(method, DictMethodBaseHandler)
            dtype = handler.infer_dtype(method, self._liteframe.dtypes.iloc[0])

            col_name = self._liteframe.columns[0]
            col_dtype = self._liteframe.dtypes.iloc[0]
            col_expr = _resolve_column_expr(self._liteframe, col_name, col_dtype)

            expr = LiteFrameDictExpr(
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

    @classmethod
    def _register(cls, method):
        """Register method to accessor class."""
        setattr(cls, method, cls._gen_func(method))
