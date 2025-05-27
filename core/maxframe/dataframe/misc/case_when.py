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

import numpy as np
from pandas.core.dtypes.cast import find_common_type

from ... import opcodes
from ...core import TILEABLE_TYPE, EntityData
from ...serialization.serializables import FieldTypes, ListField
from ..core import SERIES_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import apply_if_callable


class DataFrameCaseWhen(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.CASE_WHEN

    conditions = ListField("conditions", FieldTypes.reference, default=None)
    replacements = ListField("replacements", FieldTypes.reference, default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _set_inputs(cls, op: "DataFrameCaseWhen", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        it = iter(inputs)
        next(it)
        op.conditions = [
            next(it) if isinstance(t, TILEABLE_TYPE) else t for t in op.conditions
        ]
        op.replacements = [
            next(it) if isinstance(t, TILEABLE_TYPE) else t for t in op.replacements
        ]

    def __call__(self, series):
        replacement_dtypes = [
            it.dtype if isinstance(it, SERIES_TYPE) else np.array(it).dtype
            for it in self.replacements
            if it is not None
        ]
        dtype = find_common_type([series.dtype] + replacement_dtypes)

        condition_tileables = [
            it for it in self.conditions if isinstance(it, TILEABLE_TYPE)
        ]
        replacement_tileables = [
            it for it in self.replacements if isinstance(it, TILEABLE_TYPE)
        ]
        inputs = [series] + condition_tileables + replacement_tileables

        params = series.params
        params["dtype"] = dtype
        return self.new_series(inputs, **params)


def case_when(series, caselist):
    """
    Replace values where the conditions are True.

    Parameters
    ----------
    caselist : A list of tuples of conditions and expected replacements
        Takes the form:  ``(condition0, replacement0)``,
        ``(condition1, replacement1)``, ... .
        ``condition`` should be a 1-D boolean array-like object
        or a callable. If ``condition`` is a callable,
        it is computed on the Series
        and should return a boolean Series or array.
        The callable must not change the input Series
        (though pandas doesn`t check it). ``replacement`` should be a
        1-D array-like object, a scalar or a callable.
        If ``replacement`` is a callable, it is computed on the Series
        and should return a scalar or Series. The callable
        must not change the input Series.

    Returns
    -------
    Series

    See Also
    --------
    Series.mask : Replace values where the condition is True.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> c = md.Series([6, 7, 8, 9], name='c')
    >>> a = md.Series([0, 0, 1, 2])
    >>> b = md.Series([0, 3, 4, 5])

    >>> c.case_when(caselist=[(a.gt(0), a),  # condition, replacement
    ...                       (b.gt(0), b)]).execute()
    0    6
    1    3
    2    1
    3    2
    Name: c, dtype: int64
    """
    if not isinstance(caselist, list):
        raise TypeError(
            f"The caselist argument should be a list; instead got {type(caselist)}"
        )

    if not caselist:
        raise ValueError(
            "provide at least one boolean condition, "
            "with a corresponding replacement."
        )

    for num, entry in enumerate(caselist):
        if not isinstance(entry, tuple):
            raise TypeError(
                f"Argument {num} must be a tuple; instead got {type(entry)}."
            )
        if len(entry) != 2:
            raise ValueError(
                f"Argument {num} must have length 2; "
                "a condition and replacement; "
                f"instead got length {len(entry)}."
            )
    caselist = [
        (
            apply_if_callable(condition, series),
            apply_if_callable(replacement, series),
        )
        for condition, replacement in caselist
    ]
    conditions = [case[0] for case in caselist]
    replacements = [case[1] for case in caselist]
    op = DataFrameCaseWhen(conditions=conditions, replacements=replacements)
    return op(series)
