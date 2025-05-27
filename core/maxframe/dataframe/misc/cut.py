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

from numbers import Integral
from typing import List

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData, ExecutableTuple, OutputType
from ...serialization.serializables import (
    AnyField,
    BoolField,
    Int32Field,
    KeyField,
    StringField,
)
from ...tensor import tensor as astensor
from ...tensor.core import TensorOrder
from ..core import INDEX_TYPE, SERIES_TYPE
from ..initializer import Series as asseries
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class DataFrameCut(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.CUT

    _input = KeyField("input")
    bins = AnyField("bins", default=None)
    right = BoolField("right", default=None)
    labels = AnyField("labels", default=None)
    retbins = BoolField("retbins", default=None)
    precision = Int32Field("precision", default=None)
    include_lowest = BoolField("include_lowest", default=None)
    duplicates = StringField("duplicates", default=None)
    ordered = BoolField("ordered", default=None)

    @property
    def input(self):
        return self._input

    @property
    def output_limit(self):
        return 1 if not self.retbins else 2

    @classmethod
    def _set_inputs(cls, op: "DataFrameCut", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs)
        op._input = next(inputs_iter)
        if isinstance(op.bins, ENTITY_TYPE):
            op.bins = next(inputs_iter)
        if isinstance(op.labels, ENTITY_TYPE):
            op.labels = next(inputs_iter)

    def __call__(self, x):
        if isinstance(x, pd.Series):
            x = asseries(x)
        elif not isinstance(x, ENTITY_TYPE):
            x = astensor(x)
        if x.ndim != 1:
            raise ValueError("Input array must be 1 dimensional")
        if x.size == 0:
            raise ValueError("Cannot cut empty array")

        inputs = [x]
        if self.labels is not None and not isinstance(self.labels, (bool, ENTITY_TYPE)):
            self.labels = np.asarray(self.labels)
        # infer dtype
        x_empty = (
            pd.Series([1], dtype=x.dtype)
            if isinstance(x, SERIES_TYPE)
            else np.asarray([1], dtype=x.dtype)
        )
        if isinstance(self.bins, INDEX_TYPE):
            bins = self.bins.index_value.to_pandas()
            inputs.append(self.bins)
            bins_unknown = True
        elif isinstance(self.bins, ENTITY_TYPE):
            bins = np.asarray([2], dtype=self.bins.dtype)
            inputs.append(self.bins)
            bins_unknown = True
        else:
            bins = self.bins
            bins_unknown = isinstance(self.bins, Integral)
        if isinstance(self.labels, ENTITY_TYPE):
            bins_unknown = True
            labels = None
            inputs.append(self.labels)
        else:
            if self.labels is False or not bins_unknown:
                labels = self.labels
            else:
                labels = None
        ret = pd.cut(
            x_empty,
            bins,
            right=self.right,
            labels=labels,
            retbins=True,
            include_lowest=self.include_lowest,
            duplicates=self.duplicates,
        )

        kws = []
        output_types = []
        if bins_unknown and isinstance(ret[0].dtype, pd.CategoricalDtype):
            # inaccurate dtype, just create an empty one
            out_dtype = pd.CategoricalDtype()
        else:
            out_dtype = ret[0].dtype
        if isinstance(ret[0], pd.Series):
            output_types.append(OutputType.series)
            kws.append(
                {
                    "dtype": out_dtype,
                    "shape": x.shape,
                    "index_value": x.index_value,
                    "name": x.name,
                }
            )
        elif isinstance(ret[0], np.ndarray):
            output_types.append(OutputType.tensor)
            kws.append(
                {"dtype": out_dtype, "shape": x.shape, "order": TensorOrder.C_ORDER}
            )
        else:
            assert isinstance(ret[0], pd.Categorical)
            output_types.append(OutputType.categorical)
            kws.append(
                {
                    "dtype": out_dtype,
                    "shape": x.shape,
                    "categories_value": parse_index(
                        out_dtype.categories, store_data=True
                    ),
                }
            )

        if self.retbins:
            if isinstance(self.bins, (pd.IntervalIndex, INDEX_TYPE)):
                output_types.append(OutputType.index)
                kws.append(
                    {
                        "dtype": self.bins.dtype,
                        "shape": self.bins.shape,
                        "index_value": (
                            self.bins.index_value
                            if isinstance(self.bins, INDEX_TYPE)
                            else parse_index(self.bins, store_data=False)
                        ),
                        "name": self.bins.name,
                    }
                )
            else:
                output_types.append(OutputType.tensor)
                kws.append(
                    {
                        "dtype": ret[1].dtype,
                        "shape": ret[1].shape if ret[1].size > 0 else (np.nan,),
                        "order": TensorOrder.C_ORDER,
                    }
                )

        self.output_types = output_types
        return ExecutableTuple(self.new_tileables(inputs, kws=kws))


def cut(
    x,
    bins,
    right: bool = True,
    labels=None,
    retbins: bool = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
):
    """
    Bin values into discrete intervals.

    Use `cut` when you need to segment and sort data values into bins. This
    function is also useful for going from a continuous variable to a
    categorical variable. For example, `cut` could convert ages to groups of
    age ranges. Supports binning into an equal number of bins, or a
    pre-specified array of bins.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    bins : int, sequence of scalars, or IntervalIndex
        The criteria to bin by.

        * int : Defines the number of equal-width bins in the range of `x`. The
          range of `x` is extended by .1% on each side to include the minimum
          and maximum values of `x`.
        * sequence of scalars : Defines the bin edges allowing for non-uniform
          width. No extension of the range of `x` is done.
        * IntervalIndex : Defines the exact bins to be used. Note that
          IntervalIndex for `bins` must be non-overlapping.

    right : bool, default True
        Indicates whether `bins` includes the rightmost edge or not. If
        ``right == True`` (the default), then the `bins` ``[1, 2, 3, 4]``
        indicate (1,2], (2,3], (3,4]. This argument is ignored when
        `bins` is an IntervalIndex.
    labels : array or False, default None
        Specifies the labels for the returned bins. Must be the same length as
        the resulting bins. If False, returns only integer indicators of the
        bins. This affects the type of the output container (see below).
        This argument is ignored when `bins` is an IntervalIndex. If True,
        raises an error.
    retbins : bool, default False
        Whether to return the bins or not. Useful when bins is provided
        as a scalar.
    precision : int, default 3
        The precision at which to store and display the bins labels.
    include_lowest : bool, default False
        Whether the first interval should be left-inclusive or not.
    duplicates : {default 'raise', 'drop'}, optional
        If bin edges are not unique, raise ValueError or drop non-uniques.
    ordered : bool, default True
        Whether the labels are ordered or not. Applies to returned types
        Categorical and Series (with Categorical dtype). If True, the resulting
        categorical will be ordered. If False, the resulting categorical will be
        unordered (labels must be provided).

    Returns
    -------
    out : Categorical, Series, or Tensor
        An array-like object representing the respective bin for each value
        of `x`. The type depends on the value of `labels`.

        * True (default) : returns a Series for Series `x` or a
          Categorical for all other inputs. The values stored within
          are Interval dtype.

        * sequence of scalars : returns a Series for Series `x` or a
          Categorical for all other inputs. The values stored within
          are whatever the type in the sequence is.

        * False : returns a tensor of integers.

    bins : Tensor or IntervalIndex.
        The computed or specified bins. Only returned when `retbins=True`.
        For scalar or sequence `bins`, this is a tensor with the computed
        bins. If set `duplicates=drop`, `bins` will drop non-unique bin. For
        an IntervalIndex `bins`, this is equal to `bins`.

    See Also
    --------
    qcut : Discretize variable into equal-sized buckets based on rank
        or based on sample quantiles.
    Categorical : Array type for storing data that come from a
        fixed set of values.
    Series : One-dimensional array with axis labels (including time series).
    IntervalIndex : Immutable Index implementing an ordered, sliceable set.

    Notes
    -----
    Any NA values will be NA in the result. Out of bounds values will be NA in
    the resulting Series or Categorical object.

    Examples
    --------
    Discretize into three equal-sized bins.

    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md

    >>> md.cut(mt.array([1, 7, 5, 4, 6, 3]), 3).execute()
    ... # doctest: +ELLIPSIS
    [(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...
    Categories (3, interval[float64]): [(0.994, 3.0] < (3.0, 5.0] ...

    >>> md.cut(mt.array([1, 7, 5, 4, 6, 3]), 3, retbins=True).execute()
    ... # doctest: +ELLIPSIS
    ([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...
    Categories (3, interval[float64]): [(0.994, 3.0] < (3.0, 5.0] ...
    array([0.994, 3.   , 5.   , 7.   ]))

    Discovers the same bins, but assign them specific labels. Notice that
    the returned Categorical's categories are `labels` and is ordered.

    >>> md.cut(mt.array([1, 7, 5, 4, 6, 3]),
    ...        3, labels=["bad", "medium", "good"]).execute()
    [bad, good, medium, medium, good, bad]
    Categories (3, object): [bad < medium < good]

    ordered=False will result in unordered categories when labels are passed. This parameter
    can be used to allow non-unique labels:

    >>> md.cut(np.array([1, 7, 5, 4, 6, 3]), 3,
    ...        labels=["B", "A", "B"], ordered=False).execute()
    ['B', 'B', 'A', 'A', 'B', 'B']
    Categories (2, object): ['A', 'B']

    ``labels=False`` implies you just want the bins back.

    >>> md.cut([0, 1, 1, 2], bins=4, labels=False).execute()
    array([0, 1, 1, 3])

    Passing a Series as an input returns a Series with categorical dtype:

    >>> s = md.Series(mt.array([2, 4, 6, 8, 10]),
    ...               index=['a', 'b', 'c', 'd', 'e'])
    >>> md.cut(s, 3).execute()
    ... # doctest: +ELLIPSIS
    a    (1.992, 4.667]
    b    (1.992, 4.667]
    c    (4.667, 7.333]
    d     (7.333, 10.0]
    e     (7.333, 10.0]
    dtype: category
    Categories (3, interval[float64]): [(1.992, 4.667] < (4.667, ...

    Passing a Series as an input returns a Series with mapping value.
    It is used to map numerically to intervals based on bins.

    >>> s = md.Series(mt.array([2, 4, 6, 8, 10]),
    ...               index=['a', 'b', 'c', 'd', 'e'])
    >>> md.cut(s, [0, 2, 4, 6, 8, 10], labels=False, retbins=True, right=False).execute()
    ... # doctest: +ELLIPSIS
    (a    0.0
     b    1.0
     c    2.0
     d    3.0
     e    NaN
     dtype: float64, array([0, 2, 4, 6, 8, 10]))

    Use `drop` optional when bins is not unique

    >>> md.cut(s, [0, 2, 4, 6, 10, 10], labels=False, retbins=True,
    ...        right=False, duplicates='drop').execute()
    ... # doctest: +ELLIPSIS
    (a    0.0
     b    1.0
     c    2.0
     d    3.0
     e    NaN
     dtype: float64, array([0, 2, 4, 6, 10]))

    Passing an IntervalIndex for `bins` results in those categories exactly.
    Notice that values not covered by the IntervalIndex are set to NaN. 0
    is to the left of the first bin (which is closed on the right), and 1.5
    falls between two bins.

    >>> bins = md.Index(pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)]))
    >>> md.cut([0, 0.5, 1.5, 2.5, 4.5], bins).execute()
    [NaN, (0, 1], NaN, (2, 3], (4, 5]]
    Categories (3, interval[int64]): [(0, 1] < (2, 3] < (4, 5]]
    """

    if isinstance(bins, Integral) and bins < 1:
        raise ValueError("`bins` should be a positive integer")

    op = DataFrameCut(
        bins=bins,
        right=right,
        labels=labels,
        retbins=retbins,
        precision=precision,
        include_lowest=include_lowest,
        duplicates=duplicates,
        ordered=ordered,
    )
    ret = op(x)
    if not retbins:
        return ret[0]
    else:
        return ret
