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

from maxframe.dataframe.utils import validate_axis
from maxframe.liteframe.operators.merge import LiteFrameConcat


def concat(objs, axis=0):
    """Concatenate LiteFrame objects along a particular axis.

    Stacks inputs vertically (``axis=0``) or side-by-side (``axis=1``).
    ``None`` entries are dropped; for ``axis=0`` the union of physical
    columns is produced (missing columns null-filled, like
    ``pandas.concat(..., join='outer')``); for ``axis=1`` all inputs must
    share identical row ``nsplits`` (LiteFrame rechunk is not yet
    supported), and a physical column name duplicated across inputs is
    rejected. Virtual range columns stay virtual when their ranges are
    contiguous (``axis=0``) or carried by a single input (``axis=1``);
    non-contiguous range columns (``axis=0``) are materialized, and a
    range column carried by more than one input (``axis=1``) is rejected.

    Parameters
    ----------
    objs : list of LiteFrame or None
        LiteFrame objects to concatenate. ``None`` entries are silently
        dropped.
    axis : {0, 1}, default 0
        Axis to concatenate along. ``0`` stacks rows vertically;
        ``1`` glues columns side-by-side.

    Returns
    -------
    LiteFrame
        A new lazy LiteFrame containing the concatenated data. Call
        ``.execute()`` to materialize the results.

    See Also
    --------
    merge : Merge LiteFrame with another LiteFrame using a database-style join.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf1 = mlf.LiteFrame({'a': [1, 2], 'b': [3, 4]})
    >>> lf2 = mlf.LiteFrame({'a': [5, 6], 'b': [7, 8]})

    Stack rows vertically (``axis=0``, default):

    >>> result = mlf.concat([lf1, lf2])
    >>> result.execute()
       a  b
    0  1  3
    1  2  4
    0  5  7
    1  6  8

    Glue columns side-by-side (``axis=1``):

    >>> lf3 = mlf.LiteFrame({'c': [5, 6], 'd': [7, 8]})
    >>> result = mlf.concat([lf1, lf3], axis=1)
    >>> result.execute()
       a  b  c  d
    0  1  3  5  7
    1  2  4  6  8
    """
    objs = [o for o in objs if o is not None]
    if not objs:
        raise ValueError("No LiteFrame objects to concatenate")
    axis = validate_axis(axis, objs[0])
    return LiteFrameConcat(axis=axis)(objs)
