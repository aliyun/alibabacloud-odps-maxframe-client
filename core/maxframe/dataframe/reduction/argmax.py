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

from ... import opcodes
from ...core import OutputType
from ..utils import validate_axis
from .core import DataFrameReduction, DataFrameReductionMixin, ReductionCallable


class DataFrameArgMax(DataFrameReduction, DataFrameReductionMixin):
    _op_type_ = opcodes.ARGMAX
    _func_name = "argmax"

    @property
    def is_atomic(self):
        return True

    def get_reduction_args(self, axis=None):
        args = dict(skipna=self.skipna)
        if self.inputs and self.inputs[0].ndim > 1:
            args["axis"] = axis
        return {k: v for k, v in args.items() if v is not None}

    @classmethod
    def get_reduction_callable(cls, op):
        func_name = getattr(op, "_func_name")
        kw = dict(skipna=op.skipna)
        kw = {k: v for k, v in kw.items() if v is not None}
        return ReductionCallable(func_name=func_name, kwargs=kw)


def argmax_series_index(series_or_index, axis=0, skipna=True, *args, **kwargs):
    """
    Return int position of the smallest value in the Series.

    If the maximum is achieved in multiple locations,
    the first row position is returned.

    Parameters
    ----------
    axis : {None}
        Unused. Parameter needed for compatibility with DataFrame.
    skipna : bool, default True
        Exclude NA/null values when showing the result.
    *args, **kwargs
        Additional arguments and keywords for compatibility with NumPy.

    Returns
    -------
    int
        Row position of the maximum value.

    See Also
    --------
    Series.argmin : Return position of the minimum value.
    Series.argmax : Return position of the maximum value.
    maxframe.tensor.argmax : Equivalent method for tensors.
    Series.idxmax : Return index label of the maximum values.
    Series.idxmin : Return index label of the minimum values.

    Examples
    --------
    Consider dataset containing cereal calories

    >>> import maxframe.dataframe as md
    >>> s = md.Series({'Corn Flakes': 100.0, 'Almond Delight': 110.0,
    ...                'Cinnamon Toast Crunch': 120.0, 'Cocoa Puff': 110.0})
    >>> s.execute()
    Corn Flakes              100.0
    Almond Delight           110.0
    Cinnamon Toast Crunch    120.0
    Cocoa Puff               110.0
    dtype: float64

    >>> s.argmax().execute()
    2
    >>> s.argmin().execute()
    0

    The maximum cereal calories is the third element and
    the minimum cereal calories is the first element,
    since series is zero-indexed.
    """
    # args not implemented, just ignore
    _ = args, kwargs

    validate_axis(axis, series_or_index)
    op = DataFrameArgMax(
        dropna=skipna,
        output_types=[OutputType.scalar],
    )
    return op(series_or_index)
