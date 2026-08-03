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

import numpy as np

from maxframe import tensor as mt
from maxframe.learn.core import BaseEstimator, TransformerMixin
from maxframe.learn.utils._encode import _encode, _unique
from maxframe.learn.utils.validation import check_array, check_is_fitted
from maxframe.session import execute as execute_tileable
from maxframe.session import fetch as fetch_tileable
from maxframe.utils import check_unexpected_kwargs


class OneHotEncoder(TransformerMixin, BaseEstimator):
    """Encode categorical features as a one-hot numeric array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse_output``
    parameter).

    By default, the input is expected to be an array of integers or strings.
    If categorical features are represented as numeric data, you can use
    ``categories='auto'`` to infer the categories from the training data.

    Parameters
    ----------
    categories : 'auto' or list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

    drop : {'first', 'if_binary'} or an array-like of shape (n_features,), \
            default=None
        Specifies a methodology to use to drop one of the categories per
        feature. This is useful in situations where perfectly collinear
        features cause problems, such as when feeding the resulting data
        into an unregularized linear regression model.

        However, dropping one category breaks the symmetry of the original
        representation and can therefore induce a bias in downstream models,
        for instance for penalized linear classification or regression models.

        - None : retain all features (the default).
        - 'first' : drop the first category in each feature.
        - 'if_binary' : drop the first category in each feature with two
          categories. Features with 1 or more than 2 categories are
          left intact.
        - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
          should be dropped.

    sparse_output : bool, default=True
        When True, returns a sparse matrix. When False, returns a dense array.

    dtype : number type, default=np.float64
        Desired dtype of output.

    handle_unknown : {'error', 'ignore'}, default='error'
        Specifies the way unknown categories are handled during :meth:`transform`.

        - 'error' : Raise an error if an unknown category is encountered.
        - 'ignore' : When an unknown category is encountered, the resulting
          one-hot encoded columns for this feature will be all zeros.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``). This includes the category specified in ``drop``
        (if any).

    drop_idx_ : array of shape (n_features,)
        - ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category
          to be dropped for each feature.
        - ``drop_idx_[i] = None`` if no category is to be dropped from the
          feature with index ``i``, e.g. when `drop='if_binary'` and the
          feature isn't binary.
        - ``drop_idx_ = None`` if all the transformed features will be
          retained.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    See Also
    --------
    LabelEncoder : Encode target labels with value between 0 and n_classes-1.
    LabelBinarizer : Binarize labels in a one-vs-all fashion.

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.

    >>> from maxframe.learn.preprocessing import OneHotEncoder
    >>> enc = OneHotEncoder(sparse_output=False)
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OneHotEncoder(sparse_output=False)
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
    array([[1., 0., 1., 0., 0.],
           [0., 1., 0., 0., 0.]])
    """

    def __init__(
        self,
        *,
        categories="auto",
        drop=None,
        sparse_output=True,
        dtype=np.float64,
        handle_unknown="error",
        **kwargs,
    ):
        self.categories = categories
        self.drop = drop
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        if "sparse" in kwargs:
            self.sparse_output = kwargs.pop("sparse")
        else:
            self.sparse_output = sparse_output
        check_unexpected_kwargs(kwargs)

    def _validate_keywords(self):
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError(
                f"handle_unknown should be either 'error' or 'ignore', "
                f"got {self.handle_unknown}."
            )
        if self.drop is not None and self.drop not in ("first", "if_binary"):
            if not isinstance(self.drop, (list, np.ndarray)):
                raise ValueError(
                    f"drop should be 'first', 'if_binary', None, or an array-like, "
                    f"got {self.drop}."
                )

    def _compute_drop_idx(self):
        """Compute the drop indices for each feature."""
        if self.drop is None:
            self.drop_idx_ = None
            return

        n_features = len(self.categories_)

        if self.drop == "first":
            self.drop_idx_ = np.zeros(n_features, dtype=object)
        elif self.drop == "if_binary":
            self.drop_idx_ = np.array(
                [0 if len(cats) == 2 else None for cats in self._categories_np],
                dtype=object,
            )
        else:
            # drop is array-like
            drop_array = np.asarray(self.drop)
            if drop_array.shape[0] != n_features:
                raise ValueError(
                    f"drop should have length equal to the number of features "
                    f"({n_features}), got {drop_array.shape[0]}."
                )
            self.drop_idx_ = np.empty(n_features, dtype=object)
            for i, (drop_val, cats) in enumerate(zip(drop_array, self._categories_np)):
                if drop_val is None:
                    self.drop_idx_[i] = None
                else:
                    idx = np.where(cats == drop_val)[0]
                    if len(idx) == 0:
                        raise ValueError(
                            f"The following categories were not found in "
                            f"feature {i}: [{drop_val}]."
                        )
                    self.drop_idx_[i] = idx[0]

    def fit(self, X, y=None, execute=False, session=None, run_kwargs=None):
        """Fit OneHotEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        execute : bool, default=False
            Whether to execute the computation immediately.

        session : MaxFrame session, optional
            The session to use for execution.

        run_kwargs : dict, optional
            Additional keyword arguments for execution.

        Returns
        -------
        self : OneHotEncoder
            Fitted encoder.
        """
        self._validate_keywords()

        X = check_array(X, dtype=None)
        X = mt.asarray(X)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.categories == "auto":
            # Use _unique() per column - returns MaxFrame tensors (lazy)
            self.categories_ = []
            for i in range(n_features):
                col = X[:, i]
                cats = _unique(col)  # Returns MaxFrame tensor, NOT executed
                self.categories_.append(cats)

            # Only execute if requested - like LabelEncoder pattern
            if execute:
                self.categories_ = execute_tileable(
                    self.categories_, session=session, **(run_kwargs or {})
                )
        else:
            # User-provided categories - convert to MaxFrame tensors
            if len(self.categories) != n_features:
                raise ValueError(
                    f"Shape mismatch: if categories is not 'auto', it must have "
                    f"length equal to the number of features ({n_features})."
                )
            self.categories_ = [mt.asarray(c) for c in self.categories]
            if execute:
                self.categories_ = execute_tileable(
                    self.categories_, session=session, **(run_kwargs or {})
                )

        # Compute drop_idx_ - needs materialized categories
        if execute:
            self._categories_np = [
                c if isinstance(c, np.ndarray) else fetch_tileable(c)
                for c in self.categories_
            ]
            self._compute_drop_idx()

        if execute and hasattr(self, "categories_") and self.categories_:
            # Execute the fitted categories if they exist
            execute_tileable(self.categories_, session=session, **(run_kwargs or {}))
        return self

    def transform(self, X, execute=False, session=None, run_kwargs=None):
        """Transform X using one-hot encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        execute : bool, default=False
            Whether to execute the computation immediately.

        session : MaxFrame session, optional
            The session to use for execution.

        run_kwargs : dict, optional
            Additional keyword arguments for execution.

        Returns
        -------
        X_out : {ndarray, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            Transformed input. If `sparse_output=True`, a sparse matrix will
            be returned.
        """
        check_is_fitted(self)

        X = check_array(X, dtype=None)
        X = mt.asarray(X)

        n_samples, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but OneHotEncoder is expecting "
                f"{self.n_features_in_} features as input."
            )

        # Build entire computation graph lazily
        encoded_cols = []
        for i, cats in enumerate(self.categories_):
            col = X[:, i]
            n_cats = len(self._categories_np[i])
            cats_np = self._categories_np[i]

            if self.handle_unknown == "ignore":
                # For handle_unknown='ignore', use direct comparison
                # This avoids _encode which raises errors for unknowns
                # For each category, create a column that's 1 where col == category
                cat_cols = []
                for cat_val in cats_np:
                    # col == cat_val gives a boolean tensor
                    cat_cols.append((col == cat_val).astype(self.dtype))
                # Stack columns to create 2D one-hot matrix
                onehot = mt.stack(cat_cols, axis=1)
            else:
                # Use efficient eye matrix indexing for error mode
                # Get integer indices using _encode() - returns MaxFrame tensor (lazy)
                indices = _encode(col, uniques=cats, check_unknown=True)

                # Convert indices to one-hot using eye matrix indexing (lazy)
                # Create identity matrix of size n_categories
                eye = mt.eye(n_cats, dtype=self.dtype)
                onehot = eye[indices]

            encoded_cols.append(onehot)

        # Concatenate all columns horizontally (lazy)
        result = mt.concatenate(encoded_cols, axis=1)

        # Handle drop parameter (lazy slicing)
        if self.drop_idx_ is not None:
            result = self._apply_drop(result)

        # Convert to sparse if requested (lazy)
        if self.sparse_output:
            result = result.tosparse()

        # Only execute at the very end if requested
        if execute:
            result = result.execute(session=session, **(run_kwargs or {}))
        return result

    def _apply_drop(self, X):
        """Apply dropping of categories to the encoded result."""
        if self.drop_idx_ is None:
            return X

        # Calculate which columns to keep
        keep_cols = []
        col_offset = 0
        for i, cats in enumerate(self.categories_):
            n_cats = len(self._categories_np[i])
            drop_idx = self.drop_idx_[i]

            for j in range(n_cats):
                if drop_idx is None or j != drop_idx:
                    keep_cols.append(col_offset + j)
            col_offset += n_cats

        # Select columns to keep
        return X[:, keep_cols]

    def fit_transform(self, X, y=None, execute=False, session=None, run_kwargs=None):
        """Fit OneHotEncoder to X, then transform X.

        Equivalent to fit(X).transform(X) but more efficient.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        execute : bool, default=False
            Whether to execute the computation immediately.

        session : MaxFrame session, optional
            The session to use for execution.

        run_kwargs : dict, optional
            Additional keyword arguments for execution.

        Returns
        -------
        X_out : {ndarray, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            Transformed input.
        """
        # Fit with execution (need categories for transform)
        self.fit(X, y, session=session, run_kwargs=run_kwargs, execute=execute)

        # Transform and execute
        result = self.transform(
            X, session=session, run_kwargs=run_kwargs, execute=execute
        )
        if execute:
            result = result.execute(session=session, **(run_kwargs or dict()))
        return result

    def inverse_transform(self, X, execute=False, session=None, run_kwargs=None):
        """Convert the data back to the original representation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            The transformed data.

        execute : bool, default=False
            Whether to execute the computation immediately.

        session : MaxFrame session, optional
            The session to use for execution.

        run_kwargs : dict, optional
            Additional keyword arguments for execution.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Inverse transformed array.
        """
        check_is_fitted(self)

        X = mt.asarray(X)
        if hasattr(X, "issparse") and X.issparse():
            X = X.todense()  # Lazy conversion

        # Build lazy computation graph for inverse transform
        result_cols = []
        col_offset = 0

        for i, cats in enumerate(self.categories_):
            # Calculate number of columns for this feature
            n_cats = len(self._categories_np[i])

            # Account for dropped categories
            if self.drop_idx_ is not None and self.drop_idx_[i] is not None:
                n_cols = n_cats - 1
                drop_idx = int(self.drop_idx_[i])
            else:
                n_cols = n_cats
                drop_idx = None

            feature_data = X[:, col_offset : col_offset + n_cols]
            col_offset += n_cols

            if drop_idx is not None:
                # Need to handle dropped categories
                # Use map_chunk to compute indices with drop handling
                def _inverse_with_drop(feat_chunk, drop_idx, n_cats):
                    import numpy as np

                    # Check which rows are all zeros (dropped category)
                    is_dropped = np.all(feat_chunk == 0, axis=1)

                    # Get argmax for non-dropped rows
                    indices = np.argmax(feat_chunk, axis=1)

                    # Adjust indices: positions >= drop_idx need to be incremented
                    # because the dropped category was removed
                    indices[~is_dropped & (indices >= drop_idx)] += 1

                    # For all-zero rows, use drop_idx
                    indices[is_dropped] = drop_idx

                    return indices

                indices = feature_data.mf.apply_chunk(
                    _inverse_with_drop,
                    args=(drop_idx, n_cats),
                    dtype=np.dtype(np.int64),
                    shape=(feature_data.shape[0],),
                    elementwise=False,
                )
            else:
                # Simple case: just use argmax
                indices = mt.argmax(feature_data, axis=1)

            # Index into categories to get original values (lazy)
            col = cats[indices]
            result_cols.append(col.reshape(-1, 1))

        # Concatenate results (lazy)
        result = mt.concatenate(result_cols, axis=1)

        # Execute at the end if requested
        if execute:
            return result.execute(session=session, **(run_kwargs or {}))
        else:
            return result

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self)

        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                input_features = self.feature_names_in_
            else:
                input_features = [f"x{i}" for i in range(self.n_features_in_)]

        names = []
        for i, cats in enumerate(self._categories_np):
            for j, cat in enumerate(cats):
                if self.drop_idx_ is not None and self.drop_idx_[i] == j:
                    continue
                names.append(f"{input_features[i]}_{cat}")
        return np.array(names, dtype=object)

    def fetch(self, session=None, run_kwargs=None):
        obj = super().fetch(session=session, run_kwargs=run_kwargs)

        # sklearn <1.2 uses 'sparse' param instead of 'sparse_output'
        if hasattr(obj, "sparse") and not hasattr(obj, "sparse_output"):
            obj.sparse = self.sparse_output

        # Ensure drop_idx_ is set even when None - sklearn's transform
        # requires this attribute to exist
        if not hasattr(obj, "drop_idx_"):
            obj.drop_idx_ = self.drop_idx_

        # _check_infrequent_enabled must be called before _set_drop_idx
        # because _set_drop_idx references self._infrequent_enabled
        if callable(getattr(obj, "_check_infrequent_enabled", None)):
            obj._check_infrequent_enabled()

        # sklearn <1.2 uses _compute_drop_idx, >=1.2 uses _set_drop_idx
        if callable(getattr(obj, "_set_drop_idx", None)):
            obj._set_drop_idx()
        elif callable(getattr(obj, "_compute_drop_idx", None)):
            obj._compute_drop_idx()

        if callable(getattr(obj, "_compute_n_features_outs", None)):
            obj._n_features_outs = obj._compute_n_features_outs()
        return obj
