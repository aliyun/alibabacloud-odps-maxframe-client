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

from typing import List

import numpy as np

from maxframe import tensor as mt
from maxframe.core import ENTITY_TYPE, EntityData, OutputType
from maxframe.core.operator import Operator
from maxframe.learn.core import LearnOperatorMixin
from maxframe.serialization.serializables import (
    BoolField,
    Float64Field,
    Int32Field,
    KeyField,
    StringField,
)
from maxframe.tensor.random import RandomStateField


class GLMFitOp(Operator, LearnOperatorMixin):
    """Abstract base for regularized linear-model fit operands.

    Fields below are the intersection shared across the Logistic / SGD /
    Ridge / Lasso / ElasticNet family. Estimator-specific hyperparameters
    (e.g. ``C``, ``solver``, ``loss``, ``learning_rate``, ``positive``) live
    on the concrete subclass. This base defines no ``_op_type_`` and is never
    tiled directly.
    """

    # universal inputs
    X = KeyField("X")
    y = KeyField("y")
    sample_weight = KeyField("sample_weight", None)
    coef = KeyField("coef", None)

    # universal hyperparameters (intersection of the family)
    fit_intercept = BoolField("fit_intercept", default=True)
    alpha = Float64Field("alpha", default=0.0001)
    l1_ratio = Float64Field("l1_ratio", default=0.15)
    penalty = StringField("penalty", default="l2")
    max_iter = Int32Field("max_iter", default=1000)
    tol = Float64Field("tol", default=1e-3)
    random_state = RandomStateField("random_state", default=None)
    warm_start = BoolField("warm_start", default=False)

    def __init__(self, random_state=None, **kw):
        # Normalize to a numpy RandomState so RandomStateField can serialize
        # its state (get_state()) across the tiler boundary. Accepts None,
        # int seed, or an existing RandomState.
        if random_state is None or isinstance(random_state, (int, np.integer)):
            random_state = np.random.RandomState(random_state)
        elif not isinstance(random_state, np.random.RandomState):
            # maxframe/other RandomState — seed a numpy one from its state
            random_state = np.random.RandomState(
                np.random.RandomState().randint(0, 2**31 - 1)
            )
        super().__init__(random_state=random_state, **kw)

    @property
    def output_limit(self):
        return 3  # coef, n_iter, warn_msg

    @classmethod
    def _set_inputs(cls, op: "GLMFitOp", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        input_iter = iter(inputs)
        for field in ("X", "y", "sample_weight", "coef"):
            value = getattr(op, field)
            if isinstance(value, ENTITY_TYPE):
                setattr(op, field, next(input_iter))

    def __call__(self, X, y, sample_weight=None, coef=None):
        """Single-target output contract.

        Produces three tileables: ``coef`` of shape
        ``(n_features + int(fit_intercept),)``, a scalar ``n_iter``, and a
        scalar ``warn_msg`` object. Classifier subclasses that need a
        ``(n_features + intercept, n_classes)`` shape override this method.
        """
        X = mt.asarray(X)
        y = mt.asarray(y)
        sample_weight = mt.asarray(sample_weight) if sample_weight is not None else None
        coef = mt.asarray(coef) if coef is not None else None

        inputs = [X, y]
        if sample_weight is not None:
            inputs.append(sample_weight)
        if coef is not None:
            inputs.append(coef)

        self._output_types = [OutputType.tensor, OutputType.scalar, OutputType.object]

        n_features = X.shape[1]
        shape = (n_features + int(self.fit_intercept),)
        kws = [
            {"shape": shape, "dtype": np.dtype(float)},
            {"shape": (), "dtype": np.dtype(int)},  # n_iter
            {"shape": ()},  # warn_msg
        ]
        return self.new_tileables(inputs, kws=kws)
