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

from pandas.api.types import is_list_like

from ... import opcodes
from ...core import ENTITY_TYPE, get_output_types
from ...serialization.serializables import Int8Field, TupleField
from ...typing_ import EntityType
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import validate_axis


class DataFrameClip(DataFrameOperatorMixin, DataFrameOperator):
    _op_type_ = opcodes.CLIP

    bounds = TupleField("bounds", default=None)
    axis = Int8Field("axis", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _set_inputs(cls, op: "DataFrameClip", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(inputs[1:])

        bounds = list(op.bounds)
        if len(inputs) > 1:
            for idx in range(len(bounds)):
                if isinstance(bounds[idx], ENTITY_TYPE):
                    bounds[idx] = next(inputs_iter)
        op.bounds = tuple(bounds)

    def __call__(self, df):
        self._output_types = get_output_types(df)
        bound_inputs = [bd for bd in self.bounds if isinstance(bd, ENTITY_TYPE)]
        return self.new_tileable([df] + bound_inputs, **df.params)


def clip(df, lower=None, upper=None, *, axis=None, inplace=False):
    """
    Trim values at input threshold(s).

    Assigns values outside boundary to boundary values. Thresholds
    can be singular values or array like, and in the latter case
    the clipping is performed element-wise in the specified axis.

    Parameters
    ----------
    lower : float or array-like, default None
        Minimum threshold value. All values below this
        threshold will be set to it. If None, no lower clipping is performed.
    upper : float or array-like, default None
        Maximum threshold value. All values above this
        threshold will be set to it. If None, no upper clipping is performed.
    axis : int or str axis name, optional
        Align object with lower and upper along the given axis.
    inplace : bool, default False
        Whether to perform the operation in place on the data.
    *args, **kwargs
        Additional keywords have no effect but might be accepted
        for compatibility with numpy.

    Returns
    -------
    Series or DataFrame or None
        Same type as calling object with the values outside the
        clip boundaries replaced or None if ``inplace=True``.

    See Also
    --------
    Series.clip : Trim values at input threshold in series.
    DataFrame.clip : Trim values at input threshold in dataframe.
    numpy.clip : Clip (limit) the values in an array.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> data = {'col_0': [9, -3, 0, -1, 5], 'col_1': [-2, -7, 6, 8, -5]}
    >>> df = md.DataFrame(data)
    >>> df.execute()
       col_0  col_1
    0      9     -2
    1     -3     -7
    2      0      6
    3     -1      8
    4      5     -5

    Clips per column using lower and upper thresholds:

    >>> df.clip(lower=-4, upper=7).execute()
       col_0  col_1
    0      7     -2
    1     -3     -4
    2      0      6
    3     -1      7
    4      5     -4

    Clips using specific lower and upper thresholds per column element:

    >>> t = md.Series([2, -4, -1, 6, 3])
    >>> t.execute()
    0    2
    1   -4
    2   -1
    3    6
    4    3
    dtype: int64

    >>> df.clip(lower=t, upper=t).execute()
       col_0  col_1
    0      2      2
    1     -3     -4
    2      0     -1
    3     -1      6
    4      5      3
    """
    axis = validate_axis(axis, df) if axis is not None else None
    if axis is None and any(
        isinstance(x, ENTITY_TYPE) or is_list_like(x) for x in (lower, upper)
    ):
        if df.ndim == 1:
            axis = 0
        else:
            raise ValueError("Must specify axis=0 or 1")

    op = DataFrameClip(bounds=(lower, upper), axis=axis)
    out = op(df)
    if inplace:
        df.data = out.data
    return out
