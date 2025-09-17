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

import operator
import warnings
from typing import List

import numpy as np

from ... import opcodes
from ...core import ENTITY_TYPE, ExecutableTuple
from ...serialization.serializables import AnyField, BoolField, KeyField, TupleField
from ...typing_ import EntityType
from ..core import TENSOR_TYPE, TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin
from ..utils import is_asc_sorted

_hist_bin_selector_names = {
    "stone",
    "auto",
    "doane",
    "fd",
    "rice",
    "scott",
    "sqrt",
    "sturges",
}


def _ravel_and_check_weights(a, weights):
    """Check a and weights have matching shapes, and ravel both"""
    a = astensor(a)

    # Ensure that the array is a "subtractable" dtype
    if a.dtype == np.bool_:
        warnings.warn(
            f"Converting input from {a.dtype} to {np.uint8} for compatibility.",
            RuntimeWarning,
            stacklevel=3,
        )
        a = a.astype(np.uint8)

    if weights is not None:
        weights = astensor(weights)
        if weights.shape != a.shape:
            raise ValueError("weights should have the same shape as a.")
        weights = weights.ravel()
    a = a.ravel()
    return a, weights


def _check_range(range):
    first_edge, last_edge = range
    if first_edge > last_edge:
        raise ValueError("max must be larger than min in range parameter.")
    if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
        raise ValueError(f"supplied range of [{first_edge}, {last_edge}] is not finite")
    return first_edge, last_edge


def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = _check_range(range)
    else:
        assert a.size == 0
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


class TensorHistogramBinEdges(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.HISTOGRAM_BIN_EDGES

    bins = AnyField("bins", default=None)
    range = TupleField("range", default=None)
    weights = KeyField("weights", default=None)
    uniform_bins = TupleField("uniform_bins", default=None)

    @classmethod
    def _set_inputs(cls, op: "TensorHistogramBinEdges", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(inputs)
        next(inputs_iter)
        if isinstance(op.bins, ENTITY_TYPE):
            op.bins = next(inputs_iter)
        if op.weights is not None:
            op.weights = next(inputs_iter)

    def __call__(self, a, bins, range, weights):
        from ... import tensor as mt

        if range is not None:
            _check_range(range)
        if isinstance(bins, str):
            # string, 'auto', 'stone', ...
            # shape is unknown
            bin_name = bins
            # if `bins` is a string for an automatic method,
            # this will replace it with the number of bins calculated
            if bin_name not in _hist_bin_selector_names:
                raise ValueError(f"{bin_name!r} is not a valid estimator for `bins`")
            if weights is not None:
                raise TypeError(
                    "Automated estimation of the number of "
                    "bins is not supported for weighted data"
                )
            if isinstance(range, tuple) and len(range) == 2:
                # if `bins` is a string, e.g. 'auto', 'stone'...,
                # and `range` provided as well,
                # `a` should be trimmed first
                first_edge, last_edge = _get_outer_edges(a, range)
                a = a[(a >= first_edge) & (a <= last_edge)]
            shape = (np.nan,)
        elif mt.ndim(bins) == 0:
            try:
                n_equal_bins = operator.index(bins)
            except TypeError:  # pragma: no cover
                raise TypeError("`bins` must be an integer, a string, or an array")
            if n_equal_bins < 1:
                raise ValueError("`bins` must be positive, when an integer")
            shape = (bins + 1,)
        elif mt.ndim(bins) == 1:
            if not isinstance(bins, TENSOR_TYPE):
                bins = np.asarray(bins)
                if not is_asc_sorted(bins):
                    raise ValueError(
                        "`bins` must increase monotonically, when an array"
                    )
            shape = astensor(bins).shape
        else:
            raise ValueError("`bins` must be 1d, when an array")

        inputs = [a]
        if isinstance(bins, TENSOR_TYPE):
            inputs.append(bins)
        if weights is not None:
            inputs.append(weights)

        return self.new_tensor(inputs, shape=shape, order=TensorOrder.C_ORDER)


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    r"""
    Function to calculate only the edges of the bins used by the `histogram`
    function.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened tensor.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines the bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.

        If `bins` is a string from the list below, `histogram_bin_edges` will use
        the method chosen to calculate the optimal bin width and
        consequently the number of bins (see `Notes` for more detail on
        the estimators) from the data that falls within the requested
        range. While the bin width will be optimal for the actual data
        in the range, the number of bins will be computed to fill the
        entire range, including the empty portions. For visualisation,
        using the 'auto' option is suggested. Weighted data is not
        supported for automated bin size selection.

        'auto'
            Maximum of the 'sturges' and 'fd' estimators. Provides good
            all around performance.

        'fd' (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into
            account data variability and data size.

        'doane'
            An improved version of Sturges' estimator that works better
            with non-normal datasets.

        'scott'
            Less robust estimator that that takes into account data
            variability and data size.

        'stone'
            Estimator based on leave-one-out cross-validation estimate of
            the integrated squared error. Can be regarded as a generalization
            of Scott's rule.

        'rice'
            Estimator does not take variability into account, only data
            size. Commonly overestimates number of bins required.

        'sturges'
            R's default method, only accounts for data size. Only
            optimal for gaussian data and underestimates number of bins
            for large non-gaussian datasets.

        'sqrt'
            Square root (of data size) estimator, used by Excel and
            other programs for its speed and simplicity.

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.

    weights : array_like, optional
        A tensor of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). This is currently not used by any of the bin estimators,
        but may be in the future.

    Returns
    -------
    bin_edges : tensor of dtype float
        The edges to pass into `histogram`

    See Also
    --------
    histogram

    Notes
    -----
    The methods to estimate the optimal number of bins are well founded
    in literature, and are inspired by the choices R provides for
    histogram visualisation. Note that having the number of bins
    proportional to :math:`n^{1/3}` is asymptotically optimal, which is
    why it appears in most estimators. These are simply plug-in methods
    that give good starting points for number of bins. In the equations
    below, :math:`h` is the binwidth and :math:`n_h` is the number of
    bins. All estimators that compute bin counts are recast to bin width
    using the `ptp` of the data. The final bin count is obtained from
    ``np.round(np.ceil(range / h))``.

    'auto' (maximum of the 'sturges' and 'fd' estimators)
        A compromise to get a good value. For small datasets the Sturges
        value will usually be chosen, while larger datasets will usually
        default to FD.  Avoids the overly conservative behaviour of FD
        and Sturges for small and large datasets respectively.
        Switchover point is usually :math:`a.size \approx 1000`.

    'fd' (Freedman Diaconis Estimator)
        .. math:: h = 2 \frac{IQR}{n^{1/3}}

        The binwidth is proportional to the interquartile range (IQR)
        and inversely proportional to cube root of a.size. Can be too
        conservative for small datasets, but is quite good for large
        datasets. The IQR is very robust to outliers.

    'scott'
        .. math:: h = \sigma \sqrt[3]{\frac{24 * \sqrt{\pi}}{n}}

        The binwidth is proportional to the standard deviation of the
        data and inversely proportional to cube root of ``x.size``. Can
        be too conservative for small datasets, but is quite good for
        large datasets. The standard deviation is not very robust to
        outliers. Values are very similar to the Freedman-Diaconis
        estimator in the absence of outliers.

    'rice'
        .. math:: n_h = 2n^{1/3}

        The number of bins is only proportional to cube root of
        ``a.size``. It tends to overestimate the number of bins and it
        does not take into account data variability.

    'sturges'
        .. math:: n_h = \log _{2}n+1

        The number of bins is the base 2 log of ``a.size``.  This
        estimator assumes normality of data and is too conservative for
        larger, non-normal datasets. This is the default method in R's
        ``hist`` method.

    'doane'
        .. math:: n_h = 1 + \log_{2}(n) +
                        \log_{2}(1 + \frac{|g_1|}{\sigma_{g_1}})

            g_1 = mean[(\frac{x - \mu}{\sigma})^3]

            \sigma_{g_1} = \sqrt{\frac{6(n - 2)}{(n + 1)(n + 3)}}

        An improved version of Sturges' formula that produces better
        estimates for non-normal datasets. This estimator attempts to
        account for the skew of the data.

    'sqrt'
        .. math:: n_h = \sqrt n

        The simplest and fastest estimator. Only takes into account the
        data size.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> arr = mt.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
    >>> mt.histogram_bin_edges(arr, bins='auto', range=(0, 1)).execute()
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> mt.histogram_bin_edges(arr, bins=2).execute()
    array([0. , 2.5, 5. ])

    For consistency with histogram, a tensor of pre-computed bins is
    passed through unmodified:

    >>> mt.histogram_bin_edges(arr, [1, 2]).execute()
    array([1, 2])

    This function allows one set of bins to be computed, and reused across
    multiple histograms:

    >>> shared_bins = mt.histogram_bin_edges(arr, bins='auto')
    >>> shared_bins.execute()
    array([0., 1., 2., 3., 4., 5.])

    >>> group_id = mt.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
    >>> a = arr[group_id == 0]
    >>> a.execute()
    array([0, 1, 3])
    >>> hist_0, _ = mt.histogram(a, bins=shared_bins).execute()
    >>> b = arr[group_id == 1]
    >>> b.execute()
    array([0, 0, 2, 3, 4, 5])
    >>> hist_1, _ = mt.histogram(b, bins=shared_bins).execute()

    >>> hist_0; hist_1
    array([1, 1, 0, 1, 0])
    array([2, 0, 1, 1, 2])

    Which gives more easily comparable results than using separate bins for
    each histogram:

    >>> hist_0, bins_0 = mt.histogram(a, bins='auto').execute()
    >>> hist_1, bins_1 = mt.histogram(b, bins='auto').execute()
    >>> hist_0; hist_1
    array([1, 1, 1])
    array([2, 1, 1, 2])
    >>> bins_0; bins_1
    array([0., 1., 2., 3.])
    array([0.  , 1.25, 2.5 , 3.75, 5.  ])

    """
    a, weights = _ravel_and_check_weights(a, weights)
    op = TensorHistogramBinEdges(bins=bins, range=range, weights=weights, dtype=a.dtype)
    return op(a, bins, range, weights)


class TensorHistogram(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.HISTOGRAM

    bins = AnyField("bins", default=None)
    range = TupleField("range", default=None)
    weights = KeyField("weights", default=None)
    density = BoolField("density", default=None)
    ret_bins = BoolField("ret_bins", default=None)

    @property
    def output_limit(self):
        return 1 if not self.ret_bins else 2

    @classmethod
    def _set_inputs(cls, op: "TensorHistogram", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(inputs)
        next(inputs_iter)
        if isinstance(op.bins, ENTITY_TYPE):
            op.bins = next(inputs_iter)
        if op.weights is not None:
            op.weights = next(inputs_iter)

    def __call__(self, a, bins, range, weights):
        a, weights = _ravel_and_check_weights(a, weights)
        histogram_bin_edges_op = TensorHistogramBinEdges(
            bins=bins, range=range, weights=weights, dtype=np.dtype(np.float64)
        )
        bins = self.bins = histogram_bin_edges_op(a, bins, range, weights)

        inputs = [histogram_bin_edges_op.inputs[0]]
        if isinstance(bins, TENSOR_TYPE):
            inputs.append(bins)
        # Histogram is an integer or a float array depending on the weights.
        if weights is None:
            dtype = np.dtype(np.intp)
        else:
            inputs.append(weights)
            dtype = weights.dtype
        self.dtype = dtype

        hist = self.new_tensor(
            inputs, shape=(bins.size - 1,), order=TensorOrder.C_ORDER
        )
        return ExecutableTuple([hist, bins])


def histogram(a, bins=10, range=None, weights=None, density=None):
    r"""
    Compute the histogram of a set of data.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened tensor.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing tensor of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.

        If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by `histogram_bin_edges`.

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.

    weights : array_like, optional
        A tensor of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). If `density` is True, the weights are
        normalized, so that the integral of the density over the range
        remains 1.
    density : bool, optional
        If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.

        Overrides the ``normed`` keyword if given.

    Returns
    -------
    hist : tensor
        The values of the histogram. See `density` and `weights` for a
        description of the possible semantics.
    bin_edges : tensor of dtype float
        Return the bin edges ``(length(hist)+1)``.


    See Also
    --------
    histogramdd, bincount, searchsorted, digitize, histogram_bin_edges

    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words,
    if `bins` is::

      [1, 2, 3, 4]

    then the first bin is ``[1, 2)`` (including 1, but excluding 2) and
    the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which
    *includes* 4.


    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> mt.histogram([1, 2, 1], bins=[0, 1, 2, 3]).execute()
    (array([0, 2, 1]), array([0, 1, 2, 3]))
    >>> mt.histogram(mt.arange(4), bins=mt.arange(5), density=True).execute()
    (array([0.25, 0.25, 0.25, 0.25]), array([0, 1, 2, 3, 4]))
    >>> mt.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3]).execute()
    (array([1, 4, 1]), array([0, 1, 2, 3]))

    >>> a = mt.arange(5)
    >>> hist, bin_edges = mt.histogram(a, density=True)
    >>> hist.execute()
    array([0.5, 0. , 0.5, 0. , 0. , 0.5, 0. , 0.5, 0. , 0.5])
    >>> hist.sum().execute()
    2.4999999999999996
    >>> mt.sum(hist * mt.diff(bin_edges)).execute()
    1.0

    Automated Bin Selection Methods example, using 2 peak random data
    with 2000 points:

    >>> import matplotlib.pyplot as plt
    >>> rng = mt.random.RandomState(10)  # deterministic random data
    >>> a = mt.hstack((rng.normal(size=1000),
    ...                rng.normal(loc=5, scale=2, size=1000)))
    >>> _ = plt.hist(np.asarray(a), bins='auto')  # arguments are passed to np.histogram
    >>> plt.title("Histogram with 'auto' bins")
    Text(0.5, 1.0, "Histogram with 'auto' bins")
    >>> plt.show()

    """
    a, weights = _ravel_and_check_weights(a, weights)
    op = TensorHistogram(bins=bins, range=range, weights=weights, density=density)
    return op(a, bins, range, weights)
