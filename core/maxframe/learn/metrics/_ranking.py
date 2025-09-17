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

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData, ExecutableTuple, OutputType
from ...core.operator import Operator
from ...serialization.serializables import (
    AnyField,
    BoolField,
    Float64Field,
    KeyField,
    StringField,
)
from ...tensor import tensor as astensor
from ...utils import cache_tileables
from ..core import LearnOperatorMixin
from ..utils.validation import check_consistent_length, column_or_1d


class AucOp(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.AUC

    x = KeyField("x", default=None)
    y = KeyField("y", default=None)

    @classmethod
    def _set_inputs(cls, op: "AucOp", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        it = iter(inputs)
        for attr in ("x", "y"):
            if isinstance(getattr(op, attr, None), ENTITY_TYPE):
                setattr(op, attr, next(it))

    def __call__(self, x, y):
        self._output_types = [OutputType.scalar]
        inputs = [x, y]
        inputs = [v for v in inputs if isinstance(v, ENTITY_TYPE)]
        return self.new_tileable(inputs, shape=(), dtype=np.float64)


class RocAucScore(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.ROC_AUC_SCORE

    y_true = KeyField("y_true", default=None)
    y_score = KeyField("y_score", default=None)
    average = StringField("average", default="macro")
    sample_weight = KeyField("sample_weight", default=None)
    max_fpr = Float64Field("max_fpr", default=None)
    multi_class = StringField("multi_class", default="ovr")
    labels = AnyField("labels", default=None)

    @classmethod
    def _set_inputs(cls, op: "RocAucScore", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        it = iter(inputs)
        for attr in ("y_true", "y_score", "sample_weight", "labels"):
            if isinstance(getattr(op, attr, None), ENTITY_TYPE):
                setattr(op, attr, next(it))

    def __call__(self, y_true, y_score, sample_weight=None, labels=None):
        inputs = [y_true, y_score, sample_weight, labels]
        inputs = [x for x in inputs if isinstance(x, ENTITY_TYPE)]
        self._output_types = [OutputType.tensor]
        shape = () if self.average is not None else (np.nan,)
        return self.new_tileable(inputs, shape=shape, dtype=np.float64)


class RocCurve(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.ROC_CURVE

    y_true = KeyField("y_true", default=None)
    y_score = KeyField("y_score", default=None)
    pos_label = AnyField("pos_label", default=None)
    sample_weight = KeyField("sample_weight", default=None)
    drop_intermediate = BoolField("drop_intermediate", default=True)

    @property
    def output_limit(self) -> int:
        return 3

    @classmethod
    def _set_inputs(cls, op: "RocAucScore", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        it = iter(inputs)
        for attr in ("y_true", "y_score", "sample_weight"):
            if isinstance(getattr(op, attr, None), ENTITY_TYPE):
                setattr(op, attr, next(it))

    def __call__(self, y_true, y_score, sample_weight=None):
        inputs = [y_true, y_score, sample_weight]
        inputs = [x for x in inputs if isinstance(x, ENTITY_TYPE)]
        self._output_types = [OutputType.tensor] * 3
        kws = [
            {"shape": (np.nan,), "dtype": np.dtype(float)},
            {"shape": (np.nan,), "dtype": np.dtype(float)},
            {"shape": (np.nan,), "dtype": np.dtype(float)},
        ]
        return self.new_tileables(inputs, kws=kws)


def auc(x, y, execute=False, session=None, run_kwargs=None):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule

    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.  For an alternative
    way to summarize a precision-recall curve, see
    :func:`average_precision_score`.

    Parameters
    ----------
    x : tensor, shape = [n]
        x coordinates. These must be either monotonic increasing or monotonic
        decreasing.
    y : tensor, shape = [n]
        y coordinates.

    Returns
    -------
    auc : tensor, with float value

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> from maxframe.learn import metrics
    >>> y = mt.array([1, 1, 2, 2])
    >>> pred = mt.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr).execute()
    0.75

    See also
    --------
    roc_auc_score : Compute the area under the ROC curve
    average_precision_score : Compute average precision from prediction scores
    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds
    """
    x, y = check_consistent_length(x, y)
    x = column_or_1d(x)
    y = column_or_1d(y)

    op = AucOp(x=x, y=y)
    ret = op(x, y)
    if execute:
        return ret.execute(session=session, **(run_kwargs or dict()))
    return ret


def roc_auc_score(
    y_true,
    y_score,
    *,
    average="macro",
    sample_weight=None,
    max_fpr=None,
    multi_class="raise",
    labels=None,
    execute=False,
    session=None,
    run_kwargs=None,
):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.

    Note: this implementation can be used with binary, multiclass and
    multilabel classification, but some restrictions apply (see Parameters).

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        True labels or binary label indicators. The binary and multiclass cases
        expect labels with shape (n_samples,) while the multilabel case expects
        binary label indicators with shape (n_samples, n_classes).

    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores.

        * In the binary case, it corresponds to an array of shape
          `(n_samples,)`. Both probability estimates and non-thresholded
          decision values can be provided. The probability estimates correspond
          to the **probability of the class with the greater label**,
          i.e. `estimator.classes_[1]` and thus
          `estimator.predict_proba(X, y)[:, 1]`. The decision values
          corresponds to the output of `estimator.decision_function(X, y)`.
          See more information in the :ref:`User guide <roc_auc_binary>`;
        * In the multiclass case, it corresponds to an array of shape
          `(n_samples, n_classes)` of probability estimates provided by the
          `predict_proba` method. The probability estimates **must**
          sum to 1 across the possible classes. In addition, the order of the
          class scores must correspond to the order of ``labels``,
          if provided, or else to the numerical or lexicographical order of
          the labels in ``y_true``. See more information in the
          :ref:`User guide <roc_auc_multiclass>`;
        * In the multilabel case, it corresponds to an array of shape
          `(n_samples, n_classes)`. Probability estimates are provided by the
          `predict_proba` method and the non-thresholded decision values by
          the `decision_function` method. The probability estimates correspond
          to the **probability of the class with the greater label for each
          output** of the classifier. See more information in the
          :ref:`User guide <roc_auc_multilabel>`.

    average : {'micro', 'macro', 'samples', 'weighted'} or None, \
            default='macro'
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:
        Note: multiclass ROC AUC currently only handles the 'macro' and
        'weighted' averages.

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    max_fpr : float > 0 and <= 1, default=None
        If not ``None``, the standardized partial AUC [2]_ over the range
        [0, max_fpr] is returned. For the multiclass case, ``max_fpr``,
        should be either equal to ``None`` or ``1.0`` as AUC ROC partial
        computation currently is not supported for multiclass.

    multi_class : {'raise', 'ovr', 'ovo'}, default='raise'
        Only used for multiclass targets. Determines the type of configuration
        to use. The default value raises an error, so either
        ``'ovr'`` or ``'ovo'`` must be passed explicitly.

        ``'ovr'``:
            Stands for One-vs-rest. Computes the AUC of each class
            against the rest [3]_ [4]_. This
            treats the multiclass case in the same way as the multilabel case.
            Sensitive to class imbalance even when ``average == 'macro'``,
            because class imbalance affects the composition of each of the
            'rest' groupings.
        ``'ovo'``:
            Stands for One-vs-one. Computes the average AUC of all
            possible pairwise combinations of classes [5]_.
            Insensitive to class imbalance when
            ``average == 'macro'``.

    labels : array-like of shape (n_classes,), default=None
        Only used for multiclass targets. List of labels that index the
        classes in ``y_score``. If ``None``, the numerical or lexicographical
        order of the labels in ``y_true`` is used.

    Returns
    -------
    auc : float

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    .. [2] `Analyzing a portion of the ROC curve. McClish, 1989
            <https://www.ncbi.nlm.nih.gov/pubmed/2668680>`_

    .. [3] Provost, F., Domingos, P. (2000). Well-trained PETs: Improving
           probability estimation trees (Section 6.2), CeDER Working Paper
           #IS-00-04, Stern School of Business, New York University.

    .. [4] `Fawcett, T. (2006). An introduction to ROC analysis. Pattern
            Recognition Letters, 27(8), 861-874.
            <https://www.sciencedirect.com/science/article/pii/S016786550500303X>`_

    .. [5] `Hand, D.J., Till, R.J. (2001). A Simple Generalisation of the Area
            Under the ROC Curve for Multiple Class Classification Problems.
            Machine Learning, 45(2), 171-186.
            <http://link.springer.com/article/10.1023/A:1010920819831>`_

    See Also
    --------
    average_precision_score : Area under the precision-recall curve.
    roc_curve : Compute Receiver operating characteristic (ROC) curve.
    RocCurveDisplay.from_estimator : Plot Receiver Operating Characteristic
        (ROC) curve given an estimator and some data.
    RocCurveDisplay.from_predictions : Plot Receiver Operating Characteristic
        (ROC) curve given the true and predicted values.

    Examples
    --------
    Binary case:

    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from maxframe.learn.metrics import roc_auc_score
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    >>> roc_auc_score(y, clf.predict_proba(X)[:, 1]).execute()
    0.99...
    >>> roc_auc_score(y, clf.decision_function(X)).execute()
    0.99...

    Multiclass case:

    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(solver="liblinear").fit(X, y)
    >>> roc_auc_score(y, clf.predict_proba(X), multi_class='ovr').execute()
    0.99...

    Multilabel case:

    >>> import numpy as np
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> X, y = make_multilabel_classification(random_state=0)
    >>> clf = MultiOutputClassifier(clf).fit(X, y)
    >>> # get a list of n_output containing probability arrays of shape
    >>> # (n_samples, n_classes)
    >>> y_pred = clf.predict_proba(X)
    >>> # extract the positive columns for each output
    >>> y_pred = np.transpose([pred[:, 1] for pred in y_pred])
    >>> roc_auc_score(y, y_pred, average=None).execute()
    array([0.82..., 0.86..., 0.94..., 0.85... , 0.94...])
    >>> from sklearn.linear_model import RidgeClassifierCV
    >>> clf = RidgeClassifierCV().fit(X, y)
    >>> roc_auc_score(y, clf.decision_function(X), average=None).execute()
    array([0.81..., 0.84... , 0.93..., 0.87..., 0.94...])
    """
    cache_tileables(y_true, y_score)

    y_true = astensor(y_true)
    y_score = astensor(y_score)
    sample_weight = column_or_1d(sample_weight) if sample_weight is not None else None

    op = RocAucScore(
        y_true=y_true,
        y_score=y_score,
        average=average,
        sample_weight=sample_weight,
        max_fpr=max_fpr,
        multi_class=multi_class,
        labels=labels,
    )
    ret = op(y_true, y_score, sample_weight=sample_weight, labels=labels)
    if execute:
        ret = ret.execute(session=session, **(run_kwargs or dict()))
    return ret


def roc_curve(
    y_true,
    y_score,
    pos_label=None,
    sample_weight=None,
    drop_intermediate=True,
    execute=False,
    session=None,
    run_kwargs=None,
):
    """Compute Receiver operating characteristic (ROC)

    Note: this implementation is restricted to the binary classification task.

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------

    y_true : tensor, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : tensor, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.

    Returns
    -------
    fpr : tensor, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].

    tpr : tensor, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].

    thresholds : tensor, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.

    See also
    --------
    roc_auc_score : Compute the area under the ROC curve

    Notes
    -----
    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    .. [2] Fawcett T. An introduction to ROC analysis[J]. Pattern Recognition
           Letters, 2006, 27(8):861-874.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> from maxframe.learn import metrics
    >>> y = mt.array([1, 1, 2, 2])
    >>> scores = mt.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> tpr
    array([0. , 0.5, 0.5, 1. , 1. ])
    >>> thresholds
    array([1.8 , 0.8 , 0.4 , 0.35, 0.1 ])

    """
    cache_tileables(y_true, y_score)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    sample_weight = column_or_1d(sample_weight) if sample_weight is not None else None

    op = RocCurve(
        y_true=y_true,
        y_score=y_score,
        pos_label=pos_label,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate,
    )
    ret = op(y_true, y_score, sample_weight)
    if execute:
        ret = (
            ExecutableTuple(ret)
            .execute(session=session, **(run_kwargs or dict()))
            .fetch(session=session, **(run_kwargs or dict()))
        )
    return ExecutableTuple(ret)
