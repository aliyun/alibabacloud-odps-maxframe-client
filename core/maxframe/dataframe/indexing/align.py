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

from typing import Any, List, Optional, Union

import numpy as np

from ... import opcodes
from ...core import EntityData, OutputType, get_output_types
from ...serialization.serializables import (
    AnyField,
    Int16Field,
    Int64Field,
    KeyField,
    StringField,
)
from ...typing_ import TileableType
from ..core import IndexValue
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_empty_df, parse_index, validate_axis


class _NoNeedToAlign(Exception):
    pass


class DataFrameAlign(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.ALIGN

    lhs = KeyField("lhs")
    rhs = KeyField("rhs")
    join = StringField("join", default=None)
    axis = Int16Field("axis", default=None)
    level = AnyField("level", default=None)
    fill_value = AnyField("fill_value", default=None)
    method = StringField("method", default=None)
    limit = Int64Field("limit", default=None)
    fill_axis = Int16Field("fill_axis", default=None)
    broadcast_axis = Int16Field("broadcast_axis", default=None)

    @property
    def output_limit(self) -> int:
        return 2

    @classmethod
    def _set_inputs(cls, op: "DataFrameAlign", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.lhs = inputs[0]
        op.rhs = inputs[1]

    def __call__(self, lhs: TileableType, rhs: TileableType):
        if self.broadcast_axis != 1 or lhs.ndim == rhs.ndim:
            self._output_types = get_output_types(lhs, rhs)
        else:
            self._output_types = [OutputType.dataframe, OutputType.dataframe]

        if lhs.ndim == rhs.ndim:
            if lhs.ndim == 1:
                return self._call_series_series(lhs, rhs)
            else:
                return self._call_dataframe_dataframe(lhs, rhs)
        else:
            if lhs.ndim == 1:
                # join order need to be reversed if not symmetric
                asym_joins = {"left", "right"} - {self.join}
                if len(asym_joins) == 1:  # self.join in {"left", "right"}
                    self.join = asym_joins.pop()
                # need to put dataframe first
                self._output_types = get_output_types(rhs, lhs)
                return self._call_dataframe_series(rhs, lhs)[::-1]
            else:
                return self._call_dataframe_series(lhs, rhs)

    def _call_dataframe_dataframe(self, lhs: TileableType, rhs: TileableType):
        l_shape = list(lhs.shape)
        r_shape = list(rhs.shape)
        if self.axis is None or self.axis == 0:
            l_idx_val = r_idx_val = self._merge_index(
                lhs.index_value, rhs.index_value, how=self.join
            )
            l_shape[0] = r_shape[0] = np.nan
        else:
            l_idx_val, r_idx_val = lhs.index_value, rhs.index_value

        if self.axis is None or self.axis == 1:
            l_empty = build_empty_df(lhs.dtypes)
            r_empty = build_empty_df(rhs.dtypes)
            aligned, _ = l_empty.align(r_empty, axis=1)
            l_dtypes = r_dtypes = aligned.dtypes
            l_col_val = r_col_val = parse_index(aligned.columns, store_data=True)
            l_shape[1] = r_shape[1] = len(l_dtypes)
        else:
            l_dtypes, r_dtypes = lhs.dtypes, rhs.dtypes
            l_col_val, r_col_val = lhs.columns_value, rhs.columns_value

        l_kws = {
            "index_value": l_idx_val,
            "dtypes": l_dtypes,
            "shape": tuple(l_shape),
            "columns_value": l_col_val,
        }
        r_kws = {
            "index_value": r_idx_val,
            "dtypes": r_dtypes,
            "shape": tuple(r_shape),
            "columns_value": r_col_val,
        }
        return self.new_tileables([lhs, rhs], kws=[l_kws, r_kws])

    def _call_dataframe_series(self, lhs: TileableType, rhs: TileableType):
        l_shape = list(lhs.shape)
        if self.axis == 0 or self.broadcast_axis == 1:
            dtypes = lhs.dtypes
            col_val = lhs.columns_value
            l_idx_val = r_idx_val = self._merge_index(
                lhs.index_value, rhs.index_value, how=self.join
            )
            l_shape[0] = r_size = np.nan
        else:
            l_idx_val = lhs.index_value
            if not rhs.index_value.has_value():
                dtypes = None
                l_shape[1] = r_size = np.nan
                col_val = r_idx_val = self._merge_index(
                    lhs.columns_value, rhs.index_value, how=self.join
                )
            else:
                series_index = rhs.index_value.to_pandas()
                dtypes = lhs.dtypes.reindex(
                    lhs.dtypes.index.join(series_index, how=self.join)
                ).fillna(np.dtype(float))
                l_shape[1] = r_size = len(dtypes)
                col_val = r_idx_val = parse_index(dtypes.index, store_data=True)

        l_kws = {
            "index_value": l_idx_val,
            "dtypes": dtypes,
            "shape": tuple(l_shape),
            "columns_value": col_val,
        }
        if self.broadcast_axis == 1:
            r_kws = {
                "index_value": r_idx_val,
                "dtypes": dtypes,
                "shape": tuple(l_shape),
                "columns_value": col_val,
            }
        else:
            r_kws = {
                "index_value": r_idx_val,
                "shape": (r_size,),
                "dtype": rhs.dtype,
            }
        return self.new_tileables([lhs, rhs], kws=[l_kws, r_kws])

    def _call_series_series(self, lhs: TileableType, rhs: TileableType):
        idx = self._merge_index(lhs.index_value, rhs.index_value, how=self.join)
        kws = [
            {"index_value": idx, "shape": (np.nan,), "dtype": lhs.dtype},
            {"index_value": idx, "shape": (np.nan,), "dtype": rhs.dtype},
        ]
        return self.new_tileables([lhs, rhs], kws=kws)

    @staticmethod
    def _merge_index(
        left_index_value: IndexValue, right_index_value: IndexValue, how: str = "outer"
    ):
        left_pd = left_index_value.to_pandas()
        right_pd = right_index_value.to_pandas()

        if not left_index_value.has_value() or not right_index_value.has_value():
            left_pd = left_pd[:0]
            right_pd = right_pd[:0]
            store_data = False
        else:
            store_data = True

        joined = left_pd.join(right_pd, how=how)
        if store_data:
            return parse_index(joined, store_data=store_data)
        else:
            return parse_index(
                joined,
                {left_index_value.key, right_index_value.key},
                store_data=store_data,
            )


def align(
    df,
    other,
    join: str = "outer",
    axis: Union[int, str, None] = None,
    level: Union[int, str, None] = None,
    copy: bool = True,
    fill_value: Any = None,
    method: str = None,
    limit: Optional[int] = None,
    fill_axis: Union[int, str] = 0,
    broadcast_axis: Union[int, str] = None,
):
    """
    Align two objects on their axes with the specified join method.

    Join method is specified for each axis Index.

    Parameters
    ----------
    other : DataFrame or Series
    join : {'outer', 'inner', 'left', 'right'}, default 'outer'
    axis : allowed axis of the other object, default None
        Align on index (0), columns (1), or both (None).
    level : int or level name, default None
        Broadcast across a level, matching Index values on the
        passed MultiIndex level.
    copy : bool, default True
        Always returns new objects. If copy=False and no reindexing is
        required then original objects are returned.
    fill_value : scalar, default np.NaN
        Value to use for missing values. Defaults to NaN, but can be any
        "compatible" value.
    method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
        Method to use for filling holes in reindexed Series:

        - pad / ffill: propagate last valid observation forward to next valid.
        - backfill / bfill: use NEXT valid observation to fill gap.

    limit : int, default None
        If method is specified, this is the maximum number of consecutive
        NaN values to forward/backward fill. In other words, if there is
        a gap with more than this number of consecutive NaNs, it will only
        be partially filled. If method is not specified, this is the
        maximum number of entries along the entire axis where NaNs will be
        filled. Must be greater than 0 if not None.
    fill_axis : {0 or 'index', 1 or 'columns'}, default 0
        Filling axis, method and limit.
    broadcast_axis : {0 or 'index', 1 or 'columns'}, default None
        Broadcast values along this axis, if aligning two objects of
        different dimensions.

    Notes
    -----
    Currently argument `level` is not supported.

    Returns
    -------
    (left, right) : (DataFrame, type of other)
        Aligned objects.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame(
    ...     [[1, 2, 3, 4], [6, 7, 8, 9]], columns=["D", "B", "E", "A"], index=[1, 2]
    ... )
    >>> other = md.DataFrame(
    ...     [[10, 20, 30, 40], [60, 70, 80, 90], [600, 700, 800, 900]],
    ...     columns=["A", "B", "C", "D"],
    ...     index=[2, 3, 4],
    ... )
    >>> df.execute()
       D  B  E  A
    1  1  2  3  4
    2  6  7  8  9
    >>> other.execute()
        A    B    C    D
    2   10   20   30   40
    3   60   70   80   90
    4  600  700  800  900

    Align on columns:

    >>> left, right = df.align(other, join="outer", axis=1)
    >>> left.execute()
       A  B   C  D  E
    1  4  2 NaN  1  3
    2  9  7 NaN  6  8
    >>> right.execute()
        A    B    C    D   E
    2   10   20   30   40 NaN
    3   60   70   80   90 NaN
    4  600  700  800  900 NaN

    We can also align on the index:

    >>> left, right = df.align(other, join="outer", axis=0)
    >>> left.execute()
        D    B    E    A
    1  1.0  2.0  3.0  4.0
    2  6.0  7.0  8.0  9.0
    3  NaN  NaN  NaN  NaN
    4  NaN  NaN  NaN  NaN
    >>> right.execute()
        A      B      C      D
    1    NaN    NaN    NaN    NaN
    2   10.0   20.0   30.0   40.0
    3   60.0   70.0   80.0   90.0
    4  600.0  700.0  800.0  900.0

    Finally, the default `axis=None` will align on both index and columns:

    >>> left, right = df.align(other, join="outer", axis=None)
    >>> left.execute()
         A    B   C    D    E
    1  4.0  2.0 NaN  1.0  3.0
    2  9.0  7.0 NaN  6.0  8.0
    3  NaN  NaN NaN  NaN  NaN
    4  NaN  NaN NaN  NaN  NaN
    >>> right.execute()
           A      B      C      D   E
    1    NaN    NaN    NaN    NaN NaN
    2   10.0   20.0   30.0   40.0 NaN
    3   60.0   70.0   80.0   90.0 NaN
    4  600.0  700.0  800.0  900.0 NaN
    """
    axis = validate_axis(axis) if axis is not None else None
    fill_axis = validate_axis(fill_axis) if fill_axis is not None else None
    broadcast_axis = (
        validate_axis(broadcast_axis) if broadcast_axis is not None else None
    )

    if level is not None:
        raise NotImplementedError(f"Argument `level` not supported")
    if df.ndim != other.ndim and axis is None:
        raise ValueError("Must specify axis=0 or 1")

    op = DataFrameAlign(
        join=join,
        axis=axis,
        level=level,
        copy=copy,
        fill_value=fill_value,
        method=method,
        limit=limit,
        fill_axis=fill_axis,
        broadcast_axis=broadcast_axis,
    )
    return op(df, other)
