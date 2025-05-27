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

from .... import tensor as mt
from ....udf import builtin_function
from ...core import BaseEstimator, TransformerMixin
from ...utils._encode import _encode, _unique
from ...utils.validation import _num_samples, column_or_1d

try:
    from sklearn.utils.validation import check_is_fitted
except ImportError:
    check_is_fitted = lambda *_: None


class LabelEncoder(TransformerMixin, BaseEstimator):
    """Encode target labels with value between 0 and n_classes-1.

    This transformer should be used to encode target values, *i.e.* `y`, and
    not the input `X`.

    Read more in the :ref:`User Guide <preprocessing_targets>`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.

    See Also
    --------
    OrdinalEncoder : Encode categorical features using an ordinal encoding
        scheme.
    OneHotEncoder : Encode categorical features as a one-hot numeric array.

    Examples
    --------
    `LabelEncoder` can be used to normalize labels.

    >>> from maxframe.learn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6]).execute()
    LabelEncoder()
    >>> le.classes_.to_numpy()
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6]).to_numpy()
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2]).to_numpy()
    array([1, 1, 2, 6])

    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.

    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"]).execute()
    LabelEncoder()
    >>> list(le.classes_.to_numpy())
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"]).to_numpy()
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]).to_numpy())
    ['tokyo', 'tokyo', 'paris']
    """

    def fit(self, y, execute=False, session=None, run_kwargs=None):
        """Fit label encoder.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
            Fitted label encoder.
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = _unique(y)
        if execute:
            self.execute(session=session, **(run_kwargs or dict()))
        return self

    def fit_transform(self, y, execute=False, session=None, run_kwargs=None):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Encoded labels.
        """
        y = column_or_1d(y, warn=True)
        self.classes_, y = _unique(y, return_inverse=True)
        if execute:
            self.execute(session=session, extra_tileables=y, **(run_kwargs or dict()))
        return y

    def transform(self, y, execute=False, session=None, run_kwargs=None):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Labels as normalized encodings.
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return mt.array([])

        t = _encode(y, uniques=self.classes_)
        if execute:
            t = t.execute(session=session, **(run_kwargs or dict()))
        return t

    @staticmethod
    @builtin_function
    def _class_checker(chunk_data, classes_data):
        diff = np.setdiff1d(chunk_data, np.arange(len(classes_data)))
        if len(diff):
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        return classes_data[chunk_data]

    def inverse_transform(self, y, execute=False, session=None, run_kwargs=None):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Original encoding.
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # inverse transform of empty array is empty array
        if _num_samples(y) == 0:
            return mt.array([])

        labels = mt.asarray(y).mf.apply_chunk(
            self._class_checker, args=(self.classes_,), dtype=self.classes_.dtype
        )
        if execute:
            labels.execute(session=session, **(run_kwargs or dict()))
        return labels

    def _more_tags(self):
        return {"X_types": ["1dlabels"]}
