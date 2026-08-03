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

from maxframe.liteframe.groupby.core import LiteFrameGroupBy as LiteFrameGroupByAPI


def groupby(self, by, sort=False, dropna=True):
    """Group the LiteFrame by one or more columns.

    Parameters
    ----------
    by : str or list of str
        Column name(s) to group by.
    sort : bool, default False
        Whether to sort group keys in the result.

        .. note::
            ``sort=True`` is not yet implemented. Group keys are always
            returned in first-seen encounter order. If you need sorted
            keys, sort the result manually after aggregation.
    dropna : bool, default True
        If True, rows with null group keys are dropped before aggregation.
        If False, null-key rows are included as a separate group.

    Returns
    -------
    LiteFrameGroupBy
        An intermediate object supporting ``.agg()`` and shortcut methods
        like ``.sum()``, ``.mean()``, etc.

    Raises
    ------
    NotImplementedError
        If ``sort=True`` is passed.
    """
    return LiteFrameGroupByAPI(self, by=by, sort=sort, dropna=dropna)
