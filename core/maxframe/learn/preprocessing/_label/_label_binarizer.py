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
import scipy.sparse as sps

from ....udf import builtin_function

try:
    from sklearn.utils.multiclass import type_of_target as sk_type_of_target
except ImportError:
    sk_type_of_target = lambda _: None

from .... import opcodes
from .... import tensor as mt
from ....core import ENTITY_TYPE, OutputType
from ....core.operator import Operator
from ....serialization.serializables import AnyField, BoolField, Int32Field
from ....tensor.core import TensorOrder
from ....typing_ import EntityType, TileableType
from ...core import BaseEstimator, LearnOperatorMixin, TransformerMixin
from ...utils.multiclass import type_of_target, unique_labels
from ...utils.validation import _num_samples, check_array, check_is_fitted


class LabelBinarizer(TransformerMixin, BaseEstimator):
    """Binarize labels in a one-vs-all fashion.

    Several regression and binary classification algorithms are
    available in scikit-learn. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    At learning time, this simply consists in learning one regressor
    or binary classifier per class. In doing so, one needs to convert
    multi-class labels to binary labels (belong or does not belong
    to the class). LabelBinarizer makes this process easy with the
    transform method.

    At prediction time, one assigns the class for which the corresponding
    model gave the greatest confidence. LabelBinarizer makes this easy
    with the inverse_transform method.

    Read more in the :ref:`User Guide <preprocessing_targets>`.

    Parameters
    ----------

    neg_label : int, default=0
        Value with which negative labels must be encoded.

    pos_label : int, default=1
        Value with which positive labels must be encoded.

    sparse_output : bool, default=False
        True if the returned array from transform is desired to be in sparse
        CSR format.

    Attributes
    ----------

    classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.

    y_type_ : str
        Represents the type of the target data as evaluated by
        utils.multiclass.type_of_target. Possible type are 'continuous',
        'continuous-multioutput', 'binary', 'multiclass',
        'multiclass-multioutput', 'multilabel-indicator', and 'unknown'.

    sparse_input_ : bool
        True if the input data to transform is given as a sparse matrix, False
        otherwise.

    Examples
    --------
    >>> from maxframe.learn import preprocessing
    >>> lb = preprocessing.LabelBinarizer()
    >>> lb.fit([1, 2, 6, 4, 2])
    LabelBinarizer()
    >>> lb.classes_.execute()
    array([1, 2, 4, 6])
    >>> lb.transform([1, 6]).execute()
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    Binary targets transform to a column vector

    >>> lb = preprocessing.LabelBinarizer()
    >>> lb.fit_transform(['yes', 'no', 'no', 'yes']).execute()
    array([[1],
           [0],
           [0],
           [1]])

    Passing a 2D matrix for multilabel classification

    >>> import numpy as np
    >>> lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))
    LabelBinarizer()
    >>> lb.classes_.execute()
    array([0, 1, 2])
    >>> lb.transform([0, 1, 2, 1]).execute()
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [0, 1, 0]])

    See Also
    --------
    label_binarize : Function to perform the transform operation of
        LabelBinarizer with fixed classes.
    OneHotEncoder : Encode categorical features using a one-hot aka one-of-K
        scheme.
    """

    def __init__(self, *, neg_label=0, pos_label=1, sparse_output=False):
        if neg_label >= pos_label:
            raise ValueError(
                "neg_label={0} must be strictly less than "
                "pos_label={1}.".format(neg_label, pos_label)
            )

        if sparse_output and (pos_label == 0 or neg_label != 0):
            raise ValueError(
                "Sparse binarization is only supported with non "
                "zero pos_label and zero neg_label, got "
                "pos_label={0} and neg_label={1}"
                "".format(pos_label, neg_label)
            )

        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output

    @staticmethod
    @builtin_function
    def _check_y_type(value, y):
        if isinstance(y, np.ndarray):
            y = y.item()
        if "multioutput" in y:
            raise ValueError(
                "Multioutput target data is not supported with label binarization"
            )
        return value

    def fit(self, y, check=True, execute=False, session=None, run_kwargs=None):
        """Fit label binarizer.

        Parameters
        ----------
        y : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification.

        Returns
        -------
        self : returns an instance of self.
        """
        y = mt.tensor(y)
        self.y_type_ = type_of_target(y)

        if _num_samples(y) == 0:  # pragma: no cover
            raise ValueError("y has 0 samples: %r" % y)

        if check:
            y = y.mf.apply_chunk(
                self._check_y_type,
                args=(self.y_type_,),
                shape=y.shape,
                dtype=y.dtype,
                sparse=y.issparse(),
            )

        self.sparse_input_ = mt.tensor(y).issparse()
        self.classes_ = unique_labels(y)
        if execute:
            self.execute(session=session, **(run_kwargs or {}))
        return self

    def fit_transform(
        self, y, check=True, execute=False, session=None, run_kwargs=None
    ):
        """Fit label binarizer and transform multi-class labels to binary
        labels.

        The output of transform is sometimes referred to as
        the 1-of-K coding scheme.

        Parameters
        ----------
        y : {ndarray, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification. Sparse matrix can be
            CSR, CSC, COO, DOK, or LIL.

        Returns
        -------
        Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            Shape will be (n_samples, 1) for binary problems. Sparse matrix
            will be of CSR format.
        """
        return self.fit(
            y, check=check, execute=execute, session=session, run_kwargs=run_kwargs
        ).transform(
            y, check=check, execute=execute, session=session, run_kwargs=run_kwargs
        )

    @staticmethod
    @builtin_function
    def _check_y_type_src_target(val, yt_src, yt_target):
        if isinstance(yt_src, np.ndarray):
            yt_src = yt_src.item()
        if isinstance(yt_target, np.ndarray):
            yt_target = yt_target.item()
        y_is_multilabel = yt_target.startswith("multilabel")
        if y_is_multilabel and not yt_src.startswith("multilabel"):
            raise ValueError("The object was not fitted with multilabel input.")
        return val

    def transform(self, y, check=True, execute=False, session=None, run_kwargs=None):
        """Transform multi-class labels to binary labels.

        The output of transform is sometimes referred to by some authors as
        the 1-of-K coding scheme.

        Parameters
        ----------
        y : {array, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification. Sparse matrix can be
            CSR, CSC, COO, DOK, or LIL.

        Returns
        -------
        Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            Shape will be (n_samples, 1) for binary problems. Sparse matrix
            will be of CSR format.
        """
        check_is_fitted(self)

        y = mt.tensor(y)
        target = type_of_target(y)

        if check:
            y = y.mf.apply_chunk(
                self._check_y_type_src_target,
                args=(self.y_type_, target),
                shape=y.shape,
                dtype=y.dtype,
            )

        return label_binarize(
            y,
            classes=self.classes_,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
            sparse_output=self.sparse_output,
            execute=execute,
            session=session,
            run_kwargs=run_kwargs,
        )

    def inverse_transform(self, Y, threshold=None):
        """Transform binary labels back to multi-class labels.

        Parameters
        ----------
        Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            Target values. All sparse matrices are converted to CSR before
            inverse transformation.

        threshold : float, default=None
            Threshold used in the binary and multi-label cases.

            Use 0 when ``Y`` contains the output of decision_function
            (classifier).
            Use 0.5 when ``Y`` contains the output of predict_proba.

            If None, the threshold is assumed to be half way between
            neg_label and pos_label.

        Returns
        -------
        y : {ndarray, sparse matrix} of shape (n_samples,)
            Target values. Sparse matrix will be of CSR format.

        Notes
        -----
        In the case when the binary labels are fractional
        (probabilistic), inverse_transform chooses the class with the
        greatest value. Typically, this allows to use the output of a
        linear model's decision_function method directly as the input
        of inverse_transform.
        """
        check_is_fitted(self)

        if threshold is None:
            threshold = (self.pos_label + self.neg_label) / 2.0

        Y = mt.tensor(Y)
        shape = (Y.shape[0],) if self.y_type_ != "multilabel-indicator" else Y.shape
        y_inv = Y.mf.apply_chunk(
            _inverse_binarize,
            args=(self.y_type_, self.classes_, threshold),
            dtype=self.classes_.dtype,
            shape=shape,
            sparse=False,
        )
        if self.sparse_input_:
            y_inv = y_inv.tosparse()
        elif y_inv.issparse():
            y_inv = y_inv.todense()

        return y_inv


class LabelBinarizeOp(Operator, LearnOperatorMixin):
    _op_type_ = opcodes.LABEL_BINARIZE

    y = AnyField("y")
    classes = AnyField("classes", default=None)
    n_classes = Int32Field("n_classes", default=None)
    neg_label = Int32Field("neg_label", default=None)
    pos_label = Int32Field("pos_label", default=None)
    sparse_output = BoolField("sparse_output", default=None)
    y_type = AnyField("y_type", default=None)
    # for chunk
    pos_switch = BoolField("pos_switch", default=None)

    def __call__(
        self,
        y: TileableType,
        classes: TileableType,
        y_type: TileableType = None,
        check: bool = True,
    ):
        inputs = []
        if check and not isinstance(y, list):
            # XXX Workaround that will be removed when list of list format is
            # dropped
            self.y = y = check_array(y, accept_sparse=True, ensure_2d=False, dtype=None)

        for arg in (y, classes, y_type):
            if isinstance(arg, ENTITY_TYPE):
                inputs.append(arg)

        self.sparse = self.sparse_output
        self.output_types = [OutputType.tensor]
        n_classes = len(classes) if classes is not None else self.n_classes
        if n_classes == 2:
            n_dim1 = 1
        else:
            n_dim1 = n_classes
        return self.new_tileable(
            inputs,
            shape=(mt.asarray(y).shape[0], n_dim1),
            dtype=np.dtype(int),
            order=TensorOrder.C_ORDER,
        )

    @classmethod
    def _set_inputs(cls, op: "LabelBinarizeOp", inputs: List[EntityType]):
        super()._set_inputs(op, inputs)
        input_it = iter(op.inputs)
        if isinstance(op.y, ENTITY_TYPE):
            op.y = next(input_it)
        if isinstance(op.classes, ENTITY_TYPE):
            op.classes = next(input_it)
        if isinstance(op.y_type, ENTITY_TYPE):
            op.y_type = next(input_it)


def _label_binarize(
    y,
    *,
    classes=None,
    n_classes=None,
    neg_label=0,
    pos_label=1,
    sparse_output=False,
    y_type=None,
):
    """
    Internal label binarize function with additional arguments.

    Special Parameters
    ------------------
    n_classes : int
        Number of classes. If provided, will use 0..n_classes - 1 as classes.
    y_type : str
        Type of label. If provided, the OP will use the value directly to reduce
        potential execution.
    """
    op = LabelBinarizeOp(
        y=y,
        classes=classes,
        n_classes=n_classes,
        neg_label=neg_label,
        pos_label=pos_label,
        sparse_output=sparse_output,
        y_type=y_type,
    )
    return op(y, classes)


def label_binarize(
    y,
    *,
    classes,
    neg_label=0,
    pos_label=1,
    sparse_output=False,
    execute=False,
    session=None,
    run_kwargs=None,
):
    """Binarize labels in a one-vs-all fashion.

    Several regression and binary classification algorithms are
    available in scikit-learn. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    This function makes it possible to compute this transformation for a
    fixed set of class labels known ahead of time.

    Parameters
    ----------
    y : array-like
        Sequence of integer labels or multilabel data to encode.

    classes : array-like of shape (n_classes,)
        Uniquely holds the label for each class.

    neg_label : int, default=0
        Value with which negative labels must be encoded.

    pos_label : int, default=1
        Value with which positive labels must be encoded.

    sparse_output : bool, default=False,
        Set to true if output binary array is desired in CSR sparse format.

    Returns
    -------
    Y : {tensor, sparse tensor} of shape (n_samples, n_classes)
        Shape will be (n_samples, 1) for binary problems.

    Examples
    --------
    >>> from maxframe.learn.preprocessing import label_binarize
    >>> label_binarize([1, 6], classes=[1, 2, 4, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    The class ordering is preserved:

    >>> label_binarize([1, 6], classes=[1, 6, 4, 2])
    array([[1, 0, 0, 0],
           [0, 1, 0, 0]])

    Binary targets transform to a column vector

    >>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
    array([[1],
           [0],
           [0],
           [1]])

    See Also
    --------
    LabelBinarizer : Class used to wrap the functionality of label_binarize and
        allow for fitting to classes independently of the transform operation.
    """
    result = _label_binarize(
        y,
        classes=classes,
        neg_label=neg_label,
        pos_label=pos_label,
        sparse_output=sparse_output,
    )
    if not execute:
        return result
    return result.execute(session=session, **(run_kwargs or {}))


@builtin_function
def _inverse_binarize(y, output_type, classes, threshold):
    if output_type == "multiclass":
        return _inverse_binarize_multiclass(y, classes)
    else:
        return _inverse_binarize_thresholding(y, output_type, classes, threshold)


def _inverse_binarize_multiclass(y, classes):  # pragma: no cover
    """Inverse label binarization transformation for multiclass.

    Multiclass uses the maximal score instead of a threshold.
    """
    from sklearn.utils.sparsefuncs import min_max_axis

    classes = np.asarray(classes)

    if sps.issparse(y):
        # Find the argmax for each row in y where y is a CSR matrix

        y = y.tocsr()
        n_samples, n_outputs = y.shape
        outputs = np.arange(n_outputs)
        row_max = min_max_axis(y, 1)[1]
        row_nnz = np.diff(y.indptr)

        y_data_repeated_max = np.repeat(row_max, row_nnz)
        # picks out all indices obtaining the maximum per row
        y_i_all_argmax = np.flatnonzero(y_data_repeated_max == y.data)

        # For corner case where last row has a max of 0
        if row_max[-1] == 0:
            y_i_all_argmax = np.append(y_i_all_argmax, [len(y.data)])

        # Gets the index of the first argmax in each row from y_i_all_argmax
        index_first_argmax = np.searchsorted(y_i_all_argmax, y.indptr[:-1])
        # first argmax of each row
        y_ind_ext = np.append(y.indices, [0])
        y_i_argmax = y_ind_ext[y_i_all_argmax[index_first_argmax]]
        # Handle rows of all 0
        y_i_argmax[np.where(row_nnz == 0)[0]] = 0

        # Handles rows with max of 0 that contain negative numbers
        samples = np.arange(n_samples)[(row_nnz > 0) & (row_max.ravel() == 0)]
        for i in samples:
            ind = y.indices[y.indptr[i] : y.indptr[i + 1]]
            y_i_argmax[i] = classes[np.setdiff1d(outputs, ind)][0]

        return classes[y_i_argmax]
    else:
        return classes.take(y.argmax(axis=1), mode="clip")


def _inverse_binarize_thresholding(
    y, output_type, classes, threshold
):  # pragma: no cover
    """Inverse label binarization transformation using thresholding."""

    if output_type == "binary" and y.ndim == 2 and y.shape[1] > 2:
        raise ValueError("output_type='binary', but y.shape = {0}".format(y.shape))

    if output_type != "binary" and y.shape[1] != len(classes):
        raise ValueError(
            "The number of class is not equal to the number of dimension of y."
        )

    classes = np.asarray(classes)

    # Perform thresholding
    if sps.issparse(y):
        if threshold > 0:
            if y.format not in ("csr", "csc"):
                y = y.tocsr()
            y.data = np.array(y.data > threshold, dtype=int)
            y.eliminate_zeros()
        else:
            y = np.array(y.toarray() > threshold, dtype=int)
    else:
        y = np.array(y > threshold, dtype=int)

    # Inverse transform data
    if output_type == "binary":
        if sps.issparse(y):
            y = y.toarray()
        if y.ndim == 2 and y.shape[1] == 2:
            return classes[y[:, 1]]
        else:
            if len(classes) == 1:
                return np.repeat(classes[0], len(y))
            else:
                return classes[y.ravel()]

    elif output_type == "multilabel-indicator":
        return y

    else:
        raise ValueError("{0} format is not supported".format(output_type))
