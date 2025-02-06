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

import numpy as np

from ... import opcodes
from ...core import OutputType
from ...serialization.serializables import BoolField
from ..operators import DataFrameOperator, DataFrameOperatorMixin


class DataFrameCheckMonotonic(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.CHECK_MONOTONIC

    # 0 - increasing, 1 - decreasing
    decreasing = BoolField("decreasing", default=None)
    strict = BoolField("strict", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def __call__(self, df_obj):
        self._output_types = [OutputType.scalar]
        return self.new_tileable([df_obj], shape=(), dtype=np.dtype(bool))


def check_monotonic(series_or_index, decreasing=False, strict=False):
    """
    Check if values in the object are monotonic increasing
    or decreasing.

    Parameters
    ----------
    decreasing : bool
        If True, check if values are monotonic decreasing,
        otherwise check if values are monotonic increasing
    strict : bool
        If True, values need to be unique to get a positive
        result

    Returns
    -------
    Scalar
    """
    op = DataFrameCheckMonotonic(decreasing=decreasing, strict=strict)
    return op(series_or_index)


def is_monotonic(series_or_index):
    """
    Return boolean scalar if values in the object are
    monotonic_increasing.

    Returns
    -------
    Scalar
    """
    return check_monotonic(series_or_index, decreasing=False, strict=False)


is_monotonic_increasing = is_monotonic


def is_monotonic_decreasing(series_or_index):
    """
    Return boolean scalar if values in the object are
    monotonic_decreasing.

    Returns
    -------
    Scalar
    """
    return check_monotonic(series_or_index, decreasing=True, strict=False)
