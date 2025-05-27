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

from .... import opcodes
from ....core import ExecutableTuple
from ....serialization.serializables import BoolField, Int32Field, KeyField, StringField
from ....tensor.core import TensorOrder
from ....tensor.operators import TensorOperator, TensorOperatorMixin
from ...utils.validation import check_array


class TensorNormalize(TensorOperator, TensorOperatorMixin):
    _op_module_ = "learn"
    _op_type_ = opcodes.NORMALIZE

    input = KeyField("input")
    norm = StringField("norm", default=None)
    axis = Int32Field("axis", default=None)
    return_norm = BoolField("return_norm", default=None)
    # for test purpose
    use_sklearn = BoolField("use_sklearn", default=None)

    def __init__(self, **kw):
        super().__init__(**kw)
        if self.use_sklearn is None:
            # force to use sklearn if not specified
            self.use_sklearn = True

    @classmethod
    def _set_inputs(cls, op, inputs):
        super()._set_inputs(op, inputs)
        op.input = op.inputs[0]

    @property
    def output_limit(self):
        return 2 if self.return_norm else 1

    def __call__(self, x, copy=True):
        x = check_array(
            x,
            accept_sparse=True,
            estimator="the normalize function",
            dtype=(np.float64, np.float32, np.float16),
        )

        normed = None
        if not self.return_norm:
            res = self.new_tensor([x], shape=x.shape, order=x.order)
        else:
            kws = [
                {"shape": x.shape, "order": x.order},
                {
                    "shape": (x.shape[0] if self.axis == 1 else x.shape[1],),
                    "order": TensorOrder.C_ORDER,
                },
            ]
            res, normed = self.new_tensors([x], kws=kws, output_limit=2)

        if not copy and self.axis == 1:
            # follow the behaviour of sklearn
            x.data = res.data

        if normed is None:
            return res
        return ExecutableTuple([res, normed])


def normalize(X, norm="l2", axis=1, copy=True, return_norm=False):
    """
    Scale input vectors individually to unit norm (vector length).

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.

    norm : 'l1', 'l2', or 'max', optional ('l2' by default)
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).

    axis : 0 or 1, optional (1 by default)
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.

    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a tensor and if axis is 1).

    return_norm : boolean, default False
        whether to return the computed norms

    Returns
    -------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Normalized input X.

    norms : Tensor, shape [n_samples] if axis=1 else [n_features]
        A tensor of norms along given axis for X.
        When X is sparse, a NotImplementedError will be raised
        for norm 'l1' or 'l2'.

    See also
    --------
    Normalizer: Performs normalization using the ``Transformer`` API
        (e.g. as part of a preprocessing :class:`maxframe.learn.pipeline.Pipeline`).
    """
    if norm not in ("l1", "l2", "max"):
        raise ValueError(f"'{norm}' is not a supported norm")
    if axis not in (0, 1):
        raise ValueError(f"'{axis}' is not a supported axis")
    op = TensorNormalize(norm=norm, axis=axis, return_norm=return_norm, dtype=X.dtype)
    return op(X, copy=copy)
