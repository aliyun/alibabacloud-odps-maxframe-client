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

from maxframe.liteframe.operators.sort import LiteFrameRank
from maxframe.liteframe.utils import is_numeric_dtype, normalize_col_name

_VALID_METHODS = ("average", "min", "max", "first", "dense")
_VALID_NA_OPTIONS = ("keep", "top", "bottom")


def rank(
    self,
    axis=0,
    method="average",
    numeric_only=False,
    na_option="keep",
    ascending=True,
    pct=False,
):
    """Compute numerical data ranks (1 through n) along axis 0.

    Equal values are assigned a rank that is the average of the ranks of
    those values (matching ``pandas.DataFrame.rank``).

    Parameters
    ----------
    axis : {0}, default 0
        Index to direct ranking. Only axis=0 is supported.
    method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        How to rank the group of records that have the same value (ties).
    numeric_only : bool, default False
        For DataFrame objects, rank only numeric columns if set to True.
    na_option : {'keep', 'top', 'bottom'}, default 'keep'
        How to rank NaN values. Only ``'keep'`` is supported in v1.
    ascending : bool, default True
        Whether the elements should be ranked in ascending order.
    pct : bool, default False
        Whether to display the returned rankings in percentile form.

    Returns
    -------
    LiteFrame
        LiteFrame with data ranks as values (Float64 columns).
    """
    # Normalize axis: accept only 0/"index" (row axis, the only supported
    # axis). Reject 1/"columns" as unsupported, and reject all other values
    # (including 0.5, "0", arbitrary strings) as invalid.  Strict type
    # checking: bool is rejected (bool is a subclass of int in Python), and
    # float must be an exact integer value (0.0 accepted, 0.5 rejected).
    if axis == "index":
        axis = 0
    elif axis == "columns":
        axis = 1
    elif isinstance(axis, bool):
        raise ValueError(f"Invalid axis {axis!r}, must be 0 or 'index' for rank.")
    elif isinstance(axis, int):
        pass  # accept int directly
    elif isinstance(axis, float):
        if not axis.is_integer():
            raise ValueError(f"Invalid axis {axis!r}, must be 0 or 'index' for rank.")
        axis = int(axis)
    else:
        raise ValueError(f"Invalid axis {axis!r}, must be 0 or 'index' for rank.")
    if axis == 1:
        raise NotImplementedError(
            "rank with axis=1 is not supported for LiteFrame; "
            "LiteFrame does not support column-axis operations."
        )
    if axis != 0:
        raise ValueError(f"Invalid axis {axis!r}, must be 0 or 'index' for rank.")
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Invalid method '{method}', must be one of {list(_VALID_METHODS)}"
        )
    if na_option not in _VALID_NA_OPTIONS:
        raise ValueError(
            f"Invalid na_option '{na_option}', must be one of {list(_VALID_NA_OPTIONS)}"
        )
    if na_option in ("top", "bottom"):
        raise NotImplementedError(
            f"na_option='{na_option}' is not supported in v1; only 'keep' is supported."
        )

    # Resolve columns: numeric_only drops non-numeric columns.
    in_dtypes = self._data.dtypes
    if numeric_only:
        cols = [c for c in in_dtypes.index if is_numeric_dtype(in_dtypes[c])]
    else:
        cols = list(in_dtypes.index)

    by = [normalize_col_name(c) for c in cols]

    op = LiteFrameRank(
        method=method,
        numeric_only=numeric_only,
        na_option=na_option,
        ascending=ascending,
        pct=pct,
    )
    # Carry the resolved target columns for the tiler/executor.
    op.extra_params["by"] = by
    return op(self)
