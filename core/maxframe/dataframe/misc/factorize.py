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

from ...utils import pd_release_version
from ..core import CATEGORICAL_TYPE, INDEX_TYPE, SERIES_TYPE

_na_position_last = pd_release_version < (2, 0)


def factorize(values, sort=False, use_na_sentinel=True):
    """
    Encode the object as an enumerated type or categorical variable.

    This method is useful for obtaining a numeric representation of an
    array when all that matters is identifying distinct values. `factorize`
    is available as both a top-level function :func:`pandas.factorize`,
    and as a method :meth:`Series.factorize` and :meth:`Index.factorize`.

    Parameters
    ----------
    values : sequence
        A 1-D sequence. Sequences that aren't pandas objects are
        coerced to ndarrays before factorization.
    sort : bool, default False
        Sort `uniques` and shuffle `codes` to maintain the
        relationship.

    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values. If False,
        NaN values will be encoded as non-negative integers and will not drop the
        NaN from the uniques of the values.

    Returns
    -------
    codes : ndarray
        An integer ndarray that's an indexer into `uniques`.
        ``uniques.take(codes)`` will have the same values as `values`.
    uniques : ndarray, Index, or Categorical
        The unique valid values. When `values` is Categorical, `uniques`
        is a Categorical. When `values` is some other pandas object, an
        `Index` is returned. Otherwise, a 1-D ndarray is returned.

        .. note::

           Even if there's a missing value in `values`, `uniques` will
           *not* contain an entry for it.

    See Also
    --------
    cut : Discretize continuous-valued array.
    unique : Find the unique value in an array.

    Notes
    -----
    Reference :ref:`the user guide <reshaping.factorize>` for more examples.

    Examples
    --------
    These examples all show factorize as a top-level method like
    ``pd.factorize(values)``. The results are identical for methods like
    :meth:`Series.factorize`.

    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> codes, uniques = md.factorize(mt.array(['b', 'b', 'a', 'c', 'b'], dtype="O"))
    >>> codes.execute()
    array([0, 0, 1, 2, 0])
    >>> uniques.execute()
    array(['b', 'a', 'c'], dtype=object)

    With ``sort=True``, the `uniques` will be sorted, and `codes` will be
    shuffled so that the relationship is the maintained.

    >>> codes, uniques = md.factorize(mt.array(['b', 'b', 'a', 'c', 'b'], dtype="O"),
    ...                               sort=True)
    >>> codes.execute()
    array([1, 1, 0, 2, 1])
    >>> uniques.execute()
    array(['a', 'b', 'c'], dtype=object)

    When ``use_na_sentinel=True`` (the default), missing values are indicated in
    the `codes` with the sentinel value ``-1`` and missing values are not
    included in `uniques`.

    >>> codes, uniques = md.factorize(mt.array(['b', None, 'a', 'c', 'b'], dtype="O"))
    >>> codes.execute()
    array([ 0, -1,  1,  2,  0])
    >>> uniques.execute()
    array(['b', 'a', 'c'], dtype=object)

    Thus far, we've only factorized lists (which are internally coerced to
    NumPy arrays). When factorizing pandas objects, the type of `uniques`
    will differ. For Categoricals, a `Categorical` is returned.

    >>> cat = md.Categorical(['a', 'a', 'c'], categories=['a', 'b', 'c'])
    >>> codes, uniques = md.factorize(cat)
    >>> codes.execute()
    array([0, 0, 1])
    >>> uniques.execute()
    ['a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Notice that ``'b'`` is in ``uniques.categories``, despite not being
    present in ``cat.values``.

    For all other pandas objects, an Index of the appropriate type is
    returned.

    >>> cat = md.Series(['a', 'a', 'c'])
    >>> codes, uniques = md.factorize(cat)
    >>> codes.execute()
    array([0, 0, 1])
    >>> uniques.execute()
    Index(['a', 'c'], dtype='object')

    If NaN is in the values, and we want to include NaN in the uniques of the
    values, it can be achieved by setting ``use_na_sentinel=False``.

    >>> values = mt.array([1, 2, 1, mt.nan])
    >>> codes, uniques = md.factorize(values)  # default: use_na_sentinel=True
    >>> codes.execute()
    array([ 0,  1,  0, -1])
    >>> uniques.execute()
    array([1., 2.])

    >>> codes, uniques = md.factorize(values, use_na_sentinel=False)
    >>> codes.execute()
    array([0, 1, 0, 2])
    >>> uniques.execute()
    array([ 1.,  2., nan])
    """
    from ... import tensor as mt
    from ..datasource.index import from_tileable as index_from_tileable

    uniques, indices = mt.unique(
        values,
        return_inverse=True,
        sort=sort,
        use_na_sentinel=use_na_sentinel,
        na_position="last" if _na_position_last else None,
    )

    if isinstance(values, (SERIES_TYPE, INDEX_TYPE)):
        uniques = index_from_tileable(uniques)
    elif isinstance(values, CATEGORICAL_TYPE):
        # fixme add categorical return when categorical initializers
        #  and accessors implemented
        raise NotImplementedError
    return indices, uniques
