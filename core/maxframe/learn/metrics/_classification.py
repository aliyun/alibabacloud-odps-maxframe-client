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
from ... import tensor as mt
from ...core import ENTITY_TYPE, ExecutableTuple, OutputType
from ...core.operator import Operator
from ...serialization.serializables import (
    AnyField,
    BoolField,
    FieldTypes,
    Float64Field,
    Int64Field,
    KeyField,
    ListField,
    StringField,
)
from ...tensor.core import TensorOrder
from ...typing_ import EntityType
from ..core import LearnOperatorMixin
from ..utils import check_array, check_consistent_length
from ._check_targets import _check_targets


def _weighted_sum(sample_score, sample_weight, normalize=False):
    if normalize:
        return mt.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return mt.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()


class AccuracyScore(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.ACCURACY_SCORE

    y_true = AnyField("y_true", default=None)
    y_pred = AnyField("y_pred", default=None)
    normalize = BoolField("normalize", default=None)
    sample_weight = AnyField("sample_weight", default=None)
    type_true = KeyField("type_true", default=None)

    @classmethod
    def _set_inputs(cls, op: "AccuracyScore", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op.inputs)
        if op.y_true is not None:
            op.y_true = next(inputs_iter)
        if op.y_pred is not None:
            op.y_pred = next(inputs_iter)
        if op.type_true is not None:
            op.type_true = next(inputs_iter)
        if isinstance(op.sample_weight, ENTITY_TYPE):
            op.sample_weight = next(inputs_iter)

    def __call__(self, y_true, y_pred):
        self._output_types = [OutputType.tensor]
        type_true, y_true, y_pred = _check_targets(y_true, y_pred)
        self.type_true = type_true
        inputs = [y_true, y_pred, type_true]
        if isinstance(self.sample_weight, ENTITY_TYPE):
            inputs.append(self.sample_weight)

        dtype = (
            np.dtype(float)
            if self.normalize
            else np.result_type(y_true.dtype, y_pred.dtype)
        )
        return self.new_tileable(
            inputs, dtype=dtype, shape=(), order=TensorOrder.C_ORDER
        )


def accuracy_score(
    y_true,
    y_pred,
    normalize=True,
    sample_weight=None,
    execute=False,
    session=None,
    run_kwargs=None,
):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Read more in the :ref:`User Guide <accuracy_score>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator tensor / sparse tensor
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator tensor / sparse tensor
        Predicted labels, as returned by a classifier.

    normalize : bool, optional (default=True)
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    score : float
        If ``normalize == True``, return the fraction of correctly
        classified samples (float), else returns the number of correctly
        classified samples (int).

        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.

    See also
    --------
    jaccard_score, hamming_loss, zero_one_loss

    Notes
    -----
    In binary and multiclass classification, this function is equal
    to the ``jaccard_score`` function.

    Examples
    --------
    >>> from maxframe.learn.metrics import accuracy_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> accuracy_score(y_true, y_pred).execute()
    0.5
    >>> accuracy_score(y_true, y_pred, normalize=False).execute()
    2

    In the multilabel case with binary label indicators:

    >>> import maxframe.tensor as mt
    >>> accuracy_score(mt.array([[0, 1], [1, 1]]), mt.ones((2, 2))).execute()
    0.5
    """

    # Compute accuracy for each possible representation
    op = AccuracyScore(
        y_true=y_true, y_pred=y_pred, normalize=normalize, sample_weight=sample_weight
    )
    score = op(y_true, y_pred)
    if not execute:
        return score
    return score.execute(session=session, **(run_kwargs or dict()))


class LogLoss(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.LOG_LOSS

    y_true = AnyField("y_true")
    y_pred = AnyField("y_pred")
    eps = Float64Field("eps", default=1e-15)
    normalize = BoolField("normalize", default=True)
    sample_weight = AnyField("sample_weight", default=None)
    labels = AnyField("labels", default=None)

    @classmethod
    def _set_inputs(cls, op: "LogLoss", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op.inputs)
        op.y_true = next(inputs_iter)
        op.y_pred = next(inputs_iter)
        if isinstance(op.sample_weight, ENTITY_TYPE):
            op.sample_weight = next(inputs_iter)
        if isinstance(op.labels, ENTITY_TYPE):
            op.labels = next(inputs_iter)

    def __call__(self, y_true, y_pred, sample_weight=None, labels=None):
        self._output_types = [OutputType.tensor]
        self.sample_weight = sample_weight
        self.labels = labels
        inputs = [y_true, y_pred]
        if isinstance(self.sample_weight, ENTITY_TYPE):
            inputs.append(self.sample_weight)
        if isinstance(self.labels, ENTITY_TYPE):
            inputs.append(self.labels)

        dtype = (
            np.dtype(float)
            if self.normalize
            else np.result_type(y_true.dtype, y_pred.dtype)
        )
        return self.new_tileable(
            inputs, dtype=dtype, shape=(), order=TensorOrder.C_ORDER
        )


def log_loss(
    y_true,
    y_pred,
    *,
    eps=1e-15,
    normalize=True,
    sample_weight=None,
    labels=None,
    execute=False,
    session=None,
    run_kwargs=None,
):
    r"""Log loss, aka logistic loss or cross-entropy loss.

    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of a logistic model that returns ``y_pred`` probabilities
    for its training data ``y_true``.
    The log loss is only defined for two or more labels.
    For a single sample with true label :math:`y \in \{0,1\}` and
    and a probability estimate :math:`p = \operatorname{Pr}(y = 1)`, the log
    loss is:

    .. math::
        L_{\log}(y, p) = -(y \log (p) + (1 - y) \log (1 - p))

    Read more in the :ref:`User Guide <log_loss>`.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.

    y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method. If ``y_pred.shape = (n_samples,)``
        the probabilities provided are assumed to be that of the
        positive class. The labels in ``y_pred`` are assumed to be
        ordered alphabetically, as done by
        :class:`preprocessing.LabelBinarizer`.

    eps : float, default=1e-15
        Log loss is undefined for p=0 or p=1, so probabilities are
        clipped to max(eps, min(1 - eps, p)).

    normalize : bool, default=True
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    labels : array-like, default=None
        If not provided, labels will be inferred from y_true. If ``labels``
        is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
        assumed to be binary and are inferred from ``y_true``.

    Returns
    -------
    loss : float

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).

    Examples
    --------
    >>> from maxframe.learn.metrics import log_loss
    >>> log_loss(["spam", "ham", "ham", "spam"],
    ...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
    0.21616...

    References
    ----------
    C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
    p. 209.
    """
    if not isinstance(y_true, (ENTITY_TYPE, np.ndarray)):
        y_true = mt.array(y_true)
    if not isinstance(y_pred, (ENTITY_TYPE, np.ndarray)):
        y_pred = mt.array(y_pred)
    if sample_weight is not None and not isinstance(y_pred, (ENTITY_TYPE, np.ndarray)):
        sample_weight = mt.array(sample_weight)
    if labels is not None and not isinstance(labels, (ENTITY_TYPE, np.ndarray)):
        labels = mt.array(labels)

    y_pred = check_array(y_pred, ensure_2d=False)
    y_pred, y_true, sample_weight = check_consistent_length(
        y_pred, y_true, sample_weight
    )

    op = LogLoss(eps=eps, normalize=normalize)
    res = op(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        labels=labels,
    )
    if execute:
        return res.execute(session=session, **(run_kwargs or {}))
    return res


class MultiLabelConfusionMatrix(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.MULTILABEL_CONFUSION_MATRIX

    y_true = KeyField("y_true", default=None)
    y_pred = KeyField("y_pred", default=None)
    sample_weight = AnyField("sample_weight", default=None)
    labels = AnyField("labels", default=None)
    samplewise = BoolField("samplewise", default=False)

    @classmethod
    def _set_inputs(cls, op: "MultiLabelConfusionMatrix", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op.inputs)
        if isinstance(op.y_true, ENTITY_TYPE):
            op.y_true = next(inputs_iter)
        if isinstance(op.y_pred, ENTITY_TYPE):
            op.y_pred = next(inputs_iter)
        if isinstance(op.sample_weight, ENTITY_TYPE):
            op.sample_weight = next(inputs_iter)
        if isinstance(op.labels, ENTITY_TYPE):
            op.labels = next(inputs_iter)

    def __call__(self, y_true, y_pred, sample_weight=None, labels=None):
        self._output_types = [OutputType.tensor]

        self.y_true = y_true
        self.y_pred = y_pred
        self.sample_weight = sample_weight
        self.labels = labels

        if not self.samplewise:
            tensor_size = np.nan
        else:
            tensor_size = y_true.shape[0]

        inputs = [y_true, y_pred, sample_weight, labels]
        inputs = [t for t in inputs if isinstance(t, ENTITY_TYPE)]
        return self.new_tileable(inputs, shape=(tensor_size, 2, 2), dtype=np.dtype(int))


def multilabel_confusion_matrix(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    labels=None,
    samplewise=False,
    execute=False,
    session=None,
    run_kwargs=None,
):
    """
    Compute a confusion matrix for each class or sample.

    Compute class-wise (default) or sample-wise (samplewise=True) multilabel
    confusion matrix to evaluate the accuracy of a classification, and output
    confusion matrices for each class or sample.

    In multilabel confusion matrix :math:`MCM`, the count of true negatives
    is :math:`MCM_{:,0,0}`, false negatives is :math:`MCM_{:,1,0}`,
    true positives is :math:`MCM_{:,1,1}` and false positives is
    :math:`MCM_{:,0,1}`.

    Multiclass data will be treated as if binarized under a one-vs-rest
    transformation. Returned confusion matrices will be in the order of
    sorted unique labels in the union of (y_true, y_pred).

    Read more in the :ref:`User Guide <multilabel_confusion_matrix>`.

    Parameters
    ----------
    y_true : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
            (n_samples,)
        Ground truth (correct) target values.

    y_pred : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
            (n_samples,)
        Estimated targets as returned by a classifier.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    labels : array-like of shape (n_classes,), default=None
        A list of classes or column indices to select some (or to force
        inclusion of classes absent from the data).

    samplewise : bool, default=False
        In the multilabel case, this calculates a confusion matrix per sample.

    Returns
    -------
    multi_confusion : ndarray of shape (n_outputs, 2, 2)
        A 2x2 confusion matrix corresponding to each output in the input.
        When calculating class-wise multi_confusion (default), then
        n_outputs = n_labels; when calculating sample-wise multi_confusion
        (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
        the results will be returned in the order specified in ``labels``,
        otherwise the results will be returned in sorted order by default.

    See Also
    --------
    confusion_matrix : Compute confusion matrix to evaluate the accuracy of a
        classifier.

    Notes
    -----
    The `multilabel_confusion_matrix` calculates class-wise or sample-wise
    multilabel confusion matrices, and in multiclass tasks, labels are
    binarized under a one-vs-rest way; while
    :func:`~sklearn.metrics.confusion_matrix` calculates one confusion matrix
    for confusion between every two classes.

    Examples
    --------
    Multiclass case:

    >>> import maxframe.tensor as mt
    >>> from maxframe.learn.metrics import multilabel_confusion_matrix
    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> multilabel_confusion_matrix(y_true, y_pred,
    ...                             labels=["ant", "bird", "cat"]).execute()
    array([[[3, 1],
            [0, 2]],
    <BLANKLINE>
           [[5, 0],
            [1, 0]],
    <BLANKLINE>
           [[2, 1],
            [1, 2]]])

    Multilabel-indicator case not implemented yet.
    """
    if not isinstance(y_true, (ENTITY_TYPE, np.ndarray)):
        y_true = mt.array(y_true)
    if not isinstance(y_pred, (ENTITY_TYPE, np.ndarray)):
        y_pred = mt.array(y_pred)
    if sample_weight is not None and not isinstance(y_pred, (ENTITY_TYPE, np.ndarray)):
        sample_weight = mt.array(sample_weight)
    if labels is not None and not isinstance(labels, (ENTITY_TYPE, np.ndarray)):
        labels = mt.array(labels)

    op = MultiLabelConfusionMatrix(samplewise=samplewise)
    res = op(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        labels=labels,
    )
    if execute:
        return res.execute(session=session, **(run_kwargs or {}))
    return res


def _check_zero_division(zero_division):  # pragma: no cover
    if isinstance(zero_division, str) and zero_division == "warn":
        return
    elif isinstance(zero_division, (int, float)) and zero_division in [0, 1]:
        return
    raise ValueError(
        "Got zero_division={0}." ' Must be one of ["warn", 0, 1]'.format(zero_division)
    )


class PrecisionRecallFScoreSupport(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.PRECISION_RECALL_F_SCORE_SUPPORT

    y_true = KeyField("y_true", default=None)
    y_pred = KeyField("y_pred", default=None)
    beta = Float64Field("beta", default=1.0)
    labels = AnyField("labels", default=None)
    pos_label = Int64Field("pos_label", default=1)
    average = StringField("average", default=None)
    warn_for = ListField("warn_for", FieldTypes.string, default=None)
    sample_weight = KeyField("sample_weight", default=None)
    zero_division = AnyField("zero_division", default=None)

    @property
    def output_limit(self) -> int:
        return 3 if self.average else 4

    @classmethod
    def _set_inputs(cls, op: "PrecisionRecallFScoreSupport", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op.inputs)
        if op.y_true is not None:
            op.y_true = next(inputs_iter)
        if op.y_pred is not None:
            op.y_pred = next(inputs_iter)
        if isinstance(op.sample_weight, ENTITY_TYPE):
            op.sample_weight = next(inputs_iter)
        if isinstance(op.labels, ENTITY_TYPE):
            op.labels = next(inputs_iter)

    def __call__(self, y_true, y_pred, sample_weight=None, labels=None):
        self._output_types = [OutputType.tensor] * 4

        self.y_true = y_true
        self.y_pred = y_pred
        self.sample_weight = sample_weight
        self.labels = labels

        inputs = [y_true, y_pred, sample_weight, labels]
        inputs = [t for t in inputs if isinstance(t, ENTITY_TYPE)]

        if self.average:
            if self.average == "samples":
                tensor_shape = (y_true.shape[0],)
            else:
                tensor_shape = ()
        else:
            tensor_shape = (np.nan,)

        kws = [{"dtype": np.dtype(float), "shape": tensor_shape}] * 3
        if not self.average:
            kws.append({"dtype": np.dtype(int), "shape": tensor_shape})
        return self.new_tileables(inputs, kws=kws)


def precision_recall_fscore_support(
    y_true,
    y_pred,
    *,
    beta=1.0,
    labels=None,
    pos_label=1,
    average=None,
    warn_for=("precision", "recall", "f-score"),
    sample_weight=None,
    zero_division="warn",
    execute=False,
    session=None,
    run_kwargs=None,
):
    """Compute precision, recall, F-measure and support for each class

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.

    The support is the number of occurrences of each class in ``y_true``.

    If ``pos_label is None`` and in binary classification, this function
    returns the average precision, recall and F-measure if ``average``
    is one of ``'micro'``, ``'macro'``, ``'weighted'`` or ``'samples'``.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    beta : float, 1.0 by default
        The strength of recall versus precision in the F-score.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, 1 by default
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : string, [None (default), 'binary', 'micro', 'macro', 'samples', \
                       'weighted']
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division:
           - recall: when there are no positive labels
           - precision: when there are no positive predictions
           - f-score: both

        If set to "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    recall : float (if average is not None) or array of float, , shape =\
        [n_unique_labels]

    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    support : None (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``y_true``.

    References
    ----------
    .. [1] `Wikipedia entry for the Precision and recall
           <https://en.wikipedia.org/wiki/Precision_and_recall>`_

    .. [2] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_

    .. [3] `Discriminative Methods for Multi-labeled Classification Advances
           in Knowledge Discovery and Data Mining (2004), pp. 22-30 by Shantanu
           Godbole, Sunita Sarawagi
           <http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf>`_

    Examples
    --------
    >>> import numpy as np
    >>> from maxframe.learn.metrics import precision_recall_fscore_support
    >>> y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    >>> y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
    >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
    (0.22..., 0.33..., 0.26..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
    (0.33..., 0.33..., 0.33..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
    (0.22..., 0.33..., 0.26..., None)

    It is possible to compute per-label precisions, recalls, F1-scores and
    supports instead of averaging:

    >>> precision_recall_fscore_support(y_true, y_pred, average=None,
    ... labels=['pig', 'dog', 'cat'])
    (array([0.        , 0.        , 0.66...]),
     array([0., 0., 1.]), array([0. , 0. , 0.8]),
     array([2, 2, 2]))

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined;
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score,
    and ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.
    """
    _check_zero_division(zero_division)
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")

    if not isinstance(y_true, (ENTITY_TYPE, np.ndarray)):
        y_true = mt.array(y_true)
    if not isinstance(y_pred, (ENTITY_TYPE, np.ndarray)):
        y_pred = mt.array(y_pred)
    if sample_weight is not None and not isinstance(y_pred, (ENTITY_TYPE, np.ndarray)):
        sample_weight = mt.array(sample_weight)
    if labels is not None and not isinstance(labels, (ENTITY_TYPE, np.ndarray)):
        labels = mt.array(labels)

    op = PrecisionRecallFScoreSupport(
        beta=beta,
        pos_label=pos_label,
        average=average,
        warn_for=warn_for,
        zero_division=zero_division,
    )
    res = ExecutableTuple(
        op(y_true, y_pred, sample_weight=sample_weight, labels=labels)
    )
    if execute:
        res = res.execute(session=session, **(run_kwargs or {}))
    return res if op.output_limit == 4 else res + (None,)


def precision_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
    execute=False,
    session=None,
    run_kwargs=None,
):
    """Compute the precision

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, 1 by default
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Precision of the positive class in binary classification or weighted
        average of the precision of each class for the multiclass task.

    See also
    --------
    precision_recall_fscore_support, multilabel_confusion_matrix

    Examples
    --------
    >>> from maxframe.learn.metrics import precision_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> precision_score(y_true, y_pred, average='macro')
    0.22...
    >>> precision_score(y_true, y_pred, average='micro')
    0.33...
    >>> precision_score(y_true, y_pred, average='weighted')
    0.22...
    >>> precision_score(y_true, y_pred, average=None)
    array([0.66..., 0.        , 0.        ])
    >>> y_pred = [0, 0, 0, 0, 0, 0]
    >>> precision_score(y_true, y_pred, average=None)
    array([0.33..., 0.        , 0.        ])
    >>> precision_score(y_true, y_pred, average=None, zero_division=1)
    array([0.33..., 1.        , 1.        ])

    Notes
    -----
    When ``true positive + false positive == 0``, precision returns 0 and
    raises ``UndefinedMetricWarning``. This behavior can be
    modified with ``zero_division``.

    """
    p, _, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("precision",),
        sample_weight=sample_weight,
        zero_division=zero_division,
        execute=execute,
        session=session,
        run_kwargs=run_kwargs,
    )
    return p


def recall_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
    execute=False,
    session=None,
    run_kwargs=None,
):
    """Compute the recall

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, 1 by default
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Recall of the positive class in binary classification or weighted
        average of the recall of each class for the multiclass task.

    See also
    --------
    precision_recall_fscore_support, balanced_accuracy_score,
    multilabel_confusion_matrix

    Examples
    --------
    >>> from maxframe.learn.metrics import recall_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> recall_score(y_true, y_pred, average='macro')
    0.33...
    >>> recall_score(y_true, y_pred, average='micro')
    0.33...
    >>> recall_score(y_true, y_pred, average='weighted')
    0.33...
    >>> recall_score(y_true, y_pred, average=None)
    array([1., 0., 0.])
    >>> y_true = [0, 0, 0, 0, 0, 0]
    >>> recall_score(y_true, y_pred, average=None)
    array([0.5, 0. , 0. ])
    >>> recall_score(y_true, y_pred, average=None, zero_division=1)
    array([0.5, 1. , 1. ])

    Notes
    -----
    When ``true positive + false negative == 0``, recall returns 0 and raises
    ``UndefinedMetricWarning``. This behavior can be modified with
    ``zero_division``.
    """
    _, r, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("recall",),
        sample_weight=sample_weight,
        zero_division=zero_division,
        execute=execute,
        session=session,
        run_kwargs=run_kwargs,
    )
    return r


def f1_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
    execute=False,
    session=None,
    run_kwargs=None,
):
    """Compute the F1 score, also known as balanced F-score or F-measure

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    In the multi-class and multi-label case, this is the average of
    the F1 score of each class with weighting depending on the ``average``
    parameter.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, 1 by default
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative. If set to "warn", this acts as 0,
        but warnings are also raised.

    Returns
    -------
    f1_score : float or array of float, shape = [n_unique_labels]
        F1 score of the positive class in binary classification or weighted
        average of the F1 scores of each class for the multiclass task.

    See also
    --------
    fbeta_score, precision_recall_fscore_support, jaccard_score,
    multilabel_confusion_matrix

    References
    ----------
    .. [1] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_

    Examples
    --------
    >>> from maxframe.learn.metrics import f1_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> f1_score(y_true, y_pred, average='macro')
    0.26...
    >>> f1_score(y_true, y_pred, average='micro')
    0.33...
    >>> f1_score(y_true, y_pred, average='weighted')
    0.26...
    >>> f1_score(y_true, y_pred, average=None)
    array([0.8, 0. , 0. ])
    >>> y_true = [0, 0, 0, 0, 0, 0]
    >>> y_pred = [0, 0, 0, 0, 0, 0]
    >>> f1_score(y_true, y_pred, zero_division=1)
    1.0...

    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined;
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score,
    and ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.
    """
    return fbeta_score(
        y_true,
        y_pred,
        beta=1,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
        execute=execute,
        session=session,
        run_kwargs=run_kwargs,
    )


def fbeta_score(
    y_true,
    y_pred,
    *,
    beta,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
    execute=False,
    session=None,
    run_kwargs=None,
):
    """Compute the F-beta score

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.

    The `beta` parameter determines the weight of recall in the combined
    score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
    favors recall (``beta -> 0`` considers only precision, ``beta -> +inf``
    only recall).

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    beta : float
        Determines the weight of recall in the combined score.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, 1 by default
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : string, [None, 'binary' (default), 'micro', 'macro', 'samples', \
                       'weighted']
        This parameter is required for multiclass/multilabel targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative. If set to "warn", this acts as 0,
        but warnings are also raised.

    Returns
    -------
    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        F-beta score of the positive class in binary classification or weighted
        average of the F-beta score of each class for the multiclass task.

    See also
    --------
    precision_recall_fscore_support, multilabel_confusion_matrix

    References
    ----------
    .. [1] R. Baeza-Yates and B. Ribeiro-Neto (2011).
           Modern Information Retrieval. Addison Wesley, pp. 327-328.

    .. [2] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_

    Examples
    --------
    >>> from maxframe.learn.metrics import fbeta_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> fbeta_score(y_true, y_pred, average='macro', beta=0.5)
    0.23...
    >>> fbeta_score(y_true, y_pred, average='micro', beta=0.5)
    0.33...
    >>> fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    0.23...
    >>> fbeta_score(y_true, y_pred, average=None, beta=0.5)
    array([0.71..., 0.        , 0.        ])

    Notes
    -----
    When ``true positive + false positive == 0`` or
    ``true positive + false negative == 0``, f-score returns 0 and raises
    ``UndefinedMetricWarning``. This behavior can be
    modified with ``zero_division``.
    """

    _, _, f, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        beta=beta,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=("f-score",),
        sample_weight=sample_weight,
        zero_division=zero_division,
        execute=execute,
        session=session,
        run_kwargs=run_kwargs,
    )
    return f
