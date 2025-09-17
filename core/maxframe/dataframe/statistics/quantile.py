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
import pandas as pd
from pandas.core.dtypes.cast import find_common_type

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData
from ...serialization.serializables import (
    AnyField,
    BoolField,
    DataTypeField,
    Int32Field,
    KeyField,
    StringField,
)
from ...tensor.core import TENSOR_TYPE
from ...tensor.datasource import empty
from ...tensor.datasource import from_series as tensor_from_series
from ...tensor.datasource import tensor as astensor
from ...tensor.statistics.quantile import quantile as tensor_quantile
from ..core import DATAFRAME_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_empty_df, parse_index, validate_axis


class DataFrameQuantile(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.QUANTILE

    input = KeyField("input", default=None)
    q = AnyField("q", default=None)
    axis = Int32Field("axis", default=None)
    numeric_only = BoolField("numeric_only", default=None)
    interpolation = StringField("interpolation", default=None)

    dtype = DataTypeField("dtype", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @classmethod
    def _set_inputs(cls, op: "DataFrameQuantile", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.input = op._inputs[0]
        if isinstance(op.q, TENSOR_TYPE):
            op.q = op._inputs[-1]

    def _calc_dtype_on_axis_1(self, a, dtypes):
        quantile_dtypes = []
        for name in dtypes.index:
            dt = tensor_quantile(
                tensor_from_series(a[name]),
                self.q,
                interpolation=self.interpolation,
                handle_non_numeric=not self.numeric_only,
            ).dtype
            quantile_dtypes.append(dt)
        return find_common_type(quantile_dtypes)

    def _call_dataframe(self, a, inputs):
        if self.numeric_only:
            empty_df = build_empty_df(a.dtypes)
            dtypes = empty_df._get_numeric_data().dtypes
        else:
            dtypes = a.dtypes
        if isinstance(self.q, TENSOR_TYPE):
            q_val = self.q
            pd_index = pd.Index([], dtype=q_val.dtype)
            name = None
            store_index_value = False
        else:
            q_val = np.asanyarray(self.q)
            if q_val.ndim == 0:
                pd_index = pd.Index(q_val.reshape(1))
            else:
                pd_index = pd.Index(q_val)
            name = self.q if q_val.size == 1 else None
            store_index_value = True
        tokenize_objects = (a, q_val, self.interpolation, type(self).__name__)

        if q_val.ndim == 0 and self.axis == 0:
            index_value = parse_index(dtypes.index, store_data=store_index_value)
            shape = (len(dtypes),)
            # calc dtype
            dtype = self._calc_dtype_on_axis_1(a, dtypes)
            return self.new_series(
                inputs,
                shape=shape,
                dtype=dtype,
                index_value=index_value,
                name=name or dtypes.index.name,
            )
        elif q_val.ndim == 0 and self.axis == 1:
            index_value = a.index_value
            shape = (len(a),)
            # calc dtype
            dt = tensor_quantile(
                empty(a.shape[1], dtype=find_common_type(list(dtypes))),
                self.q,
                interpolation=self.interpolation,
                handle_non_numeric=not self.numeric_only,
            ).dtype
            return self.new_series(
                inputs,
                shape=shape,
                dtype=dt,
                index_value=index_value,
                name=name or index_value.name,
            )
        elif q_val.ndim == 1 and self.axis == 0:
            shape = (len(q_val), len(dtypes))
            index_value = parse_index(
                pd_index, *tokenize_objects, store_data=store_index_value
            )
            dtype_list = []
            for name in dtypes.index:
                dtype_list.append(
                    tensor_quantile(
                        tensor_from_series(a[name]),
                        self.q,
                        interpolation=self.interpolation,
                        handle_non_numeric=not self.numeric_only,
                    ).dtype
                )
            dtypes = pd.Series(dtype_list, index=dtypes.index)
            return self.new_dataframe(
                inputs,
                shape=shape,
                dtypes=dtypes,
                index_value=index_value,
                columns_value=parse_index(dtypes.index, store_data=True),
            )
        else:
            assert q_val.ndim == 1 and self.axis == 1
            shape = (len(q_val), a.shape[0])
            index_value = parse_index(
                pd_index, *tokenize_objects, store_data=store_index_value
            )
            pd_columns = a.index_value.to_pandas()
            dtype_list = np.full(len(pd_columns), self._calc_dtype_on_axis_1(a, dtypes))
            dtypes = pd.Series(dtype_list, index=pd_columns)
            return self.new_dataframe(
                inputs,
                shape=shape,
                dtypes=dtypes,
                index_value=index_value,
                columns_value=parse_index(
                    dtypes.index, store_data=True, key=a.index_value.key
                ),
            )

    def _call_series(self, a, inputs):
        if isinstance(self.q, TENSOR_TYPE):
            q_val = self.q
            index_val = pd.Index([], dtype=q_val.dtype)
            store_index_value = False
        else:
            q_val = np.asanyarray(self.q)
            if q_val.ndim == 0:
                index_val = pd.Index(q_val.reshape(1))
            else:
                index_val = pd.Index(q_val)
            store_index_value = True

        # get dtype by tensor
        a_t = astensor(a)
        self._dtype = dtype = tensor_quantile(
            a_t,
            self.q,
            interpolation=self.interpolation,
            handle_non_numeric=not self.numeric_only,
        ).dtype

        if q_val.ndim == 0:
            return self.new_scalar(inputs, dtype=dtype)
        else:
            return self.new_series(
                inputs,
                shape=q_val.shape,
                dtype=dtype,
                index_value=parse_index(
                    index_val,
                    a,
                    q_val,
                    self.interpolation,
                    type(self).__name__,
                    store_data=store_index_value,
                ),
                name=a.name,
            )

    def __call__(self, a, q_input=None):
        inputs = [a]
        if q_input is not None:
            inputs.append(q_input)
        if isinstance(a, DATAFRAME_TYPE):
            return self._call_dataframe(a, inputs)
        else:
            return self._call_series(a, inputs)


def series_quantile(series, q=0.5, interpolation="linear"):
    """
    Return value at the given quantile.

    Parameters
    ----------
    q : float or array-like, default 0.5 (50% quantile)
        0 <= q <= 1, the quantile(s) to compute.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}

        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:

            * linear: `i + (j - i) * fraction`, where `fraction` is the
              fractional part of the index surrounded by `i` and `j`.
            * lower: `i`.
            * higher: `j`.
            * nearest: `i` or `j` whichever is nearest.
            * midpoint: (`i` + `j`) / 2.

    Returns
    -------
    float or Series
        If ``q`` is an array or a tensor, a Series will be returned where the
        index is ``q`` and the values are the quantiles, otherwise
        a float will be returned.

    See Also
    --------
    core.window.Rolling.quantile
    numpy.percentile

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> s = md.Series([1, 2, 3, 4])
    >>> s.quantile(.5).execute()
    2.5
    >>> s.quantile([.25, .5, .75]).execute()
    0.25    1.75
    0.50    2.50
    0.75    3.25
    dtype: float64
    """

    if isinstance(q, ENTITY_TYPE):
        q = astensor(q)
        q_input = q
    else:
        q_input = None

    op = DataFrameQuantile(q=q, interpolation=interpolation, gpu=series.op.gpu)
    return op(series, q_input=q_input)


def dataframe_quantile(df, q=0.5, axis=0, numeric_only=True, interpolation="linear"):
    # FIXME: Timedelta not support. Data invalid: ODPS-0010000:InvalidArgument:duration[ns] is not equal to string
    """
    Return values at the given quantile over requested axis.

    Parameters
    ----------
    q : float or array-like, default 0.5 (50% quantile)
        Value between 0 <= q <= 1, the quantile(s) to compute.
    axis : {0, 1, 'index', 'columns'} (default 0)
        Equals 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
    numeric_only : bool, default True
        If False, the quantile of datetime and timedelta data will be
        computed as well.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:
        * linear: `i + (j - i) * fraction`, where `fraction` is the
          fractional part of the index surrounded by `i` and `j`.
        * lower: `i`.
        * higher: `j`.
        * nearest: `i` or `j` whichever is nearest.
        * midpoint: (`i` + `j`) / 2.

    Returns
    -------
    Series or DataFrame
        If ``q`` is an array or a tensor, a DataFrame will be returned where the
          index is ``q``, the columns are the columns of self, and the
          values are the quantiles.
        If ``q`` is a float, a Series will be returned where the
          index is the columns of self and the values are the quantiles.

    See Also
    --------
    core.window.Rolling.quantile: Rolling quantile.
    numpy.percentile: Numpy function to compute the percentile.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]),
    ...                   columns=['a', 'b'])
    >>> df.quantile(.1).execute()
    a    1.3
    b    3.7
    Name: 0.1, dtype: float64

    >>> df.quantile([.1, .5]).execute()
           a     b
    0.1  1.3   3.7
    0.5  2.5  55.0
    """
    if isinstance(q, ENTITY_TYPE):
        q = astensor(q)
        q_input = q
    else:
        q_input = None
    axis = validate_axis(axis, df)

    op = DataFrameQuantile(
        q=q,
        interpolation=interpolation,
        axis=axis,
        numeric_only=numeric_only,
        gpu=df.op.gpu,
    )
    return op(df, q_input=q_input)
