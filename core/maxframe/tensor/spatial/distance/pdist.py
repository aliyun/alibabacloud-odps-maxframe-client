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

from typing import List, Tuple

import numpy as np

from .... import opcodes
from ....core import EntityData
from ....core.operator import OperatorStage
from ....serialization import PickleContainer
from ....serialization.serializables import (
    AnyField,
    FieldTypes,
    Float16Field,
    Int32Field,
    KeyField,
    TupleField,
)
from ....udf import BuiltinFunction
from ...core import TensorOrder
from ...datasource.array import tensor as astensor
from ...operators import TensorMapReduceOperator, TensorOperatorMixin


class TensorPDist(TensorMapReduceOperator, TensorOperatorMixin):
    _op_type_ = opcodes.PDIST

    metric = AnyField("metric", default=None)
    p = Float16Field(
        "p", on_serialize=lambda x: float(x) if x is not None else x, default=None
    )
    w = KeyField("w", default=None)
    v = KeyField("V", default=None)
    vi = KeyField("VI", default=None)
    aggregate_size = Int32Field("aggregate_size", default=None)

    a = KeyField("a", default=None)
    a_offset = Int32Field("a_offset", default=None)
    b = KeyField("b", default=None)
    b_offset = Int32Field("b_offset", default=None)
    out_sizes = TupleField("out_sizes", FieldTypes.int32, default=None)
    n = Int32Field("n", default=None)

    @classmethod
    def _set_inputs(cls, op: "TensorPDist", inputs: List[EntityData]) -> None:
        super()._set_inputs(op, inputs)
        inputs_iter = iter(inputs)

        if op.stage == OperatorStage.map:
            op.a = next(inputs_iter)
            if op.b is not None:
                op.b = next(inputs_iter)
        else:
            next(inputs_iter)

        if op.w is not None:
            op.w = next(inputs_iter)
        if op.v is not None:
            op.v = next(inputs_iter)
        if op.vi is not None:
            op.vi = next(inputs_iter)

    def __call__(self, x, shape: Tuple):
        inputs = [x]
        for val in [self.w, self.v, self.vi]:
            if val is not None:
                inputs.append(val)
        return self.new_tensor(inputs, shape=shape, order=TensorOrder.C_ORDER)

    def has_custom_code(self) -> bool:
        return (
            callable(self.metric) and not isinstance(self.metric, BuiltinFunction)
        ) or isinstance(self.metric, PickleContainer)


def pdist(X, metric="euclidean", **kwargs):
    """
    Pairwise distances between observations in n-dimensional space.

    See Notes for common calling conventions.

    Parameters
    ----------
    X : Tensor
        An m by n tensor of m original observations in an
        n-dimensional space.
    metric : str or function, optional
        The distance metric to use. The distance function can
        be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching',
        'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
    **kwargs : dict, optional
        Extra arguments to `metric`: refer to each metric documentation for a
        list of all possible arguments.

        Some possible arguments:

        p : scalar
        The p-norm to apply for Minkowski, weighted and unweighted.
        Default: 2.

        w : Tensor
        The weight vector for metrics that support weights (e.g., Minkowski).

        V : Tensor
        The variance vector for standardized Euclidean.
        Default: var(X, axis=0, ddof=1)

        VI : Tensor
        The inverse of the covariance matrix for Mahalanobis.
        Default: inv(cov(X.T)).T

        out : Tensor.
        The output tensor
        If not None, condensed distance matrix Y is stored in this tensor.
        Note: metric independent, it will become a regular keyword arg in a
        future scipy version

    Returns
    -------
    Y : Tensor
        Returns a condensed distance matrix Y.  For
        each :math:`i` and :math:`j` (where :math:`i<j<m`),where m is the number
        of original observations. The metric ``dist(u=X[i], v=X[j])``
        is computed and stored in entry ``ij``.

    See Also
    --------
    squareform : converts between condensed distance matrices and
                 square distance matrices.

    Notes
    -----
    See ``squareform`` for information on how to calculate the index of
    this entry or to convert the condensed distance matrix to a
    redundant square matrix.

    The following are common calling conventions.

    1. ``Y = pdist(X, 'euclidean')``

       Computes the distance between m points using Euclidean distance
       (2-norm) as the distance metric between the points. The points
       are arranged as m n-dimensional row vectors in the matrix X.

    2. ``Y = pdist(X, 'minkowski', p=2.)``

       Computes the distances using the Minkowski distance
       :math:`||u-v||_p` (p-norm) where :math:`p \\geq 1`.

    3. ``Y = pdist(X, 'cityblock')``

       Computes the city block or Manhattan distance between the
       points.

    4. ``Y = pdist(X, 'seuclidean', V=None)``

       Computes the standardized Euclidean distance. The standardized
       Euclidean distance between two n-vectors ``u`` and ``v`` is

       .. math::

          \\sqrt{\\sum {(u_i-v_i)^2 / V[x_i]}}


       V is the variance vector; V[i] is the variance computed over all
       the i'th components of the points.  If not passed, it is
       automatically computed.

    5. ``Y = pdist(X, 'sqeuclidean')``

       Computes the squared Euclidean distance :math:`||u-v||_2^2` between
       the vectors.

    6. ``Y = pdist(X, 'cosine')``

       Computes the cosine distance between vectors u and v,

       .. math::

          1 - \\frac{u \\cdot v}
                   {{||u||}_2 {||v||}_2}

       where :math:`||*||_2` is the 2-norm of its argument ``*``, and
       :math:`u \\cdot v` is the dot product of ``u`` and ``v``.

    7. ``Y = pdist(X, 'correlation')``

       Computes the correlation distance between vectors u and v. This is

       .. math::

          1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                   {{||(u - \\bar{u})||}_2 {||(v - \\bar{v})||}_2}

       where :math:`\\bar{v}` is the mean of the elements of vector v,
       and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.

    8. ``Y = pdist(X, 'hamming')``

       Computes the normalized Hamming distance, or the proportion of
       those vector elements between two n-vectors ``u`` and ``v``
       which disagree. To save memory, the matrix ``X`` can be of type
       boolean.

    9. ``Y = pdist(X, 'jaccard')``

       Computes the Jaccard distance between the points. Given two
       vectors, ``u`` and ``v``, the Jaccard distance is the
       proportion of those elements ``u[i]`` and ``v[i]`` that
       disagree.

    10. ``Y = pdist(X, 'chebyshev')``

       Computes the Chebyshev distance between the points. The
       Chebyshev distance between two n-vectors ``u`` and ``v`` is the
       maximum norm-1 distance between their respective elements. More
       precisely, the distance is given by

       .. math::

          d(u,v) = \\max_i {|u_i-v_i|}

    11. ``Y = pdist(X, 'canberra')``

       Computes the Canberra distance between the points. The
       Canberra distance between two points ``u`` and ``v`` is

       .. math::

         d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                              {|u_i|+|v_i|}


    12. ``Y = pdist(X, 'braycurtis')``

       Computes the Bray-Curtis distance between the points. The
       Bray-Curtis distance between two points ``u`` and ``v`` is


       .. math::

            d(u,v) = \\frac{\\sum_i {|u_i-v_i|}}
                           {\\sum_i {|u_i+v_i|}}

    13. ``Y = pdist(X, 'mahalanobis', VI=None)``

       Computes the Mahalanobis distance between the points. The
       Mahalanobis distance between two points ``u`` and ``v`` is
       :math:`\\sqrt{(u-v)(1/V)(u-v)^T}` where :math:`(1/V)` (the ``VI``
       variable) is the inverse covariance. If ``VI`` is not None,
       ``VI`` will be used as the inverse covariance matrix.

    14. ``Y = pdist(X, 'yule')``

       Computes the Yule distance between each pair of boolean
       vectors. (see yule function documentation)

    15. ``Y = pdist(X, 'matching')``

       Synonym for 'hamming'.

    16. ``Y = pdist(X, 'dice')``

       Computes the Dice distance between each pair of boolean
       vectors. (see dice function documentation)

    17. ``Y = pdist(X, 'kulsinski')``

       Computes the Kulsinski distance between each pair of
       boolean vectors. (see kulsinski function documentation)

    18. ``Y = pdist(X, 'rogerstanimoto')``

       Computes the Rogers-Tanimoto distance between each pair of
       boolean vectors. (see rogerstanimoto function documentation)

    19. ``Y = pdist(X, 'russellrao')``

       Computes the Russell-Rao distance between each pair of
       boolean vectors. (see russellrao function documentation)

    20. ``Y = pdist(X, 'sokalmichener')``

       Computes the Sokal-Michener distance between each pair of
       boolean vectors. (see sokalmichener function documentation)

    21. ``Y = pdist(X, 'sokalsneath')``

       Computes the Sokal-Sneath distance between each pair of
       boolean vectors. (see sokalsneath function documentation)

    22. ``Y = pdist(X, 'wminkowski', p=2, w=w)``

       Computes the weighted Minkowski distance between each pair of
       vectors. (see wminkowski function documentation)

    23. ``Y = pdist(X, f)``

       Computes the distance between all pairs of vectors in X
       using the user supplied 2-arity function f. For example,
       Euclidean distance between the vectors could be computed
       as follows::

         dm = pdist(X, lambda u, v: np.sqrt(((u-v)**2).sum()))

       Note that you should avoid passing a reference to one of
       the distance functions defined in this library. For example,::

         dm = pdist(X, sokalsneath)

       would calculate the pair-wise distances between the vectors in
       X using the Python function sokalsneath. This would result in
       sokalsneath being called :math:`{n \\choose 2}` times, which
       is inefficient. Instead, the optimized C version is more
       efficient, and we call it using the following syntax.::

         dm = pdist(X, 'sokalsneath')

    """

    X = astensor(X, order="C")

    if X.issparse():
        raise ValueError("Sparse tensors are not supported by this function.")

    s = X.shape
    if len(s) != 2:
        raise ValueError("A 2-dimensional tensor must be passed.")

    m = s[0]
    out = kwargs.pop("out", None)
    if out is not None:
        if not hasattr(out, "shape"):
            raise TypeError("return arrays must be a tensor")
        if out.shape != (m * (m - 1) // 2,):
            raise ValueError("output tensor has incorrect shape.")
        if out.dtype != np.double:
            raise ValueError("Output tensor must be double type.")

    if not callable(metric) and not isinstance(metric, str):
        raise TypeError(
            "2nd argument metric must be a string identifier or a function."
        )

    # scipy remove "wminkowski" since v1.8.0, use "minkowski" with `w=`
    # keyword-argument for the given weight.
    if metric == "wminkowski":
        metric = "minkowski"

    p = kwargs.pop("p", None)
    w = kwargs.pop("w", None)
    if w is not None:
        w = astensor(w)
    v = kwargs.pop("V", None)
    if v is not None:
        v = astensor(v)
    vi = kwargs.pop("VI", None)
    if vi is not None:
        vi = astensor(vi)
    aggregate_size = kwargs.pop("aggregate_size", None)

    if len(kwargs) > 0:
        raise TypeError(
            f"`pdist` got an unexpected keyword argument '{next(iter(kwargs))}'"
        )

    op = TensorPDist(
        metric=metric,
        p=p,
        w=w,
        v=v,
        vi=vi,
        aggregate_size=aggregate_size,
        dtype=np.dtype(float),
    )
    shape = (m * (m - 1) // 2,)
    ret = op(X, shape)

    if out is None:
        return ret
    else:
        out.data = ret.data
        return out
