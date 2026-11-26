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

import math

import numpy as np

from maxframe.liteframe.functional.utils import effective_groupby_input_dtypes
from maxframe.liteframe.merge.concat import concat
from maxframe.liteframe.operators.sample import LiteFrameSample
from maxframe.liteframe.utils import normalize_col_name
from maxframe.utils import tokenize_int


def _validate_sample_params(n, frac, replace):
    """Shared validation for whole-frame and groupby sample."""
    if n is not None and frac is not None:
        raise ValueError("Please enter a value for `frac` OR `n`, not both.")
    if frac is not None and frac < 0:
        raise ValueError("frac must be non-negative")
    if n is not None and n < 0:
        raise ValueError("n must be non-negative")
    if frac is not None and frac > 1.0 and not replace:
        raise ValueError(
            "Replace has to be set to `True` when "
            "upsampling the population `frac` > 1."
        )
    return 1 if n is None and frac is None else n  # pandas default: n=1


def _normalize_random_state(random_state):
    """Convert int/None to np.random.RandomState."""
    if random_state is None:
        return None
    if not isinstance(random_state, (int, np.integer, np.random.RandomState)):
        raise TypeError(
            "random_state must be int, np.integer, np.random.RandomState, or None"
        )
    if isinstance(random_state, (int, np.integer)):
        return np.random.RandomState(random_state)
    return random_state


def sample(
    self, n=None, frac=None, replace=False, weights=None, random_state=None, exact=False
):
    """Return a random sample of rows from the frame.

    By default, sampling is performed independently within each chunk,
    which is faster but approximate. Set ``exact=True`` for globally
    coordinated sampling that guarantees statistically correct results
    across the entire frame.

    Parameters
    ----------
    n : int, optional
        Number of rows to return. Cannot be used with ``frac``.
    frac : float, optional
        Fraction of rows to return (``0``–``1`` for ``replace=False``).
    replace : bool, default False
        Allow sampling the same row more than once.
    weights : str or LiteFrame, optional
        Column name (str) for column-based weights, or a single-column
        LiteFrame aligned with the data.
    random_state : int, RandomState, or None
        Seed for reproducibility.
    exact : bool, default False
        If ``True``, use globally coordinated sampling for exact
        statistical correctness. If ``False``, each chunk samples
        independently — faster but approximate.

    Returns
    -------
    LiteFrame
        Sampled frame.
    """
    n = _validate_sample_params(n, frac, replace)
    rs = _normalize_random_state(random_state)

    if n is not None and not replace:
        N = self.shape[0]
        if N is not None and not (isinstance(N, float) and math.isnan(N)) and n > N:
            raise ValueError(
                f"Cannot take a larger sample than population when "
                f"replace=False: n={n} > N={int(N)}"
            )

    # Normalize weights: str → column name, LiteFrame → merged via concat
    if isinstance(weights, str):
        weights_col = normalize_col_name(weights)
        if weights_col not in self.dtypes.index:
            raise KeyError(f"weights column '{weights}' not found in frame")
        merged = self
    elif weights is not None:
        if hasattr(weights, "_data") and hasattr(weights, "dtypes"):
            weights_col = normalize_col_name(weights.columns[0])
            merged = concat([self, weights], axis=1)
        else:
            raise TypeError(
                "weights must be a column name (str) or a LiteFrame, "
                f"got {type(weights).__name__}"
            )
    else:
        weights_col, merged = None, self

    result = LiteFrameSample(
        n=n,
        frac=frac,
        replace=replace,
        exact=exact,
        random_state=rs,
        weights_col=weights_col,
    )(merged)
    if weights_col is not None and not isinstance(weights, str):
        result = result.drop(weights_col)
    return result


def _groupby_sample(self, n, frac, replace, weights, random_state, exact=False):
    """GroupBy sample via groupby_apply_chunk.

    After groupby shuffle, each group's complete data is in one partition,
    so per-group sampling is fully correct for all n/frac/replace/weights
    combinations. The ``exact`` flag is accepted for API parity but unused.
    """
    n = _validate_sample_params(n, frac, replace)
    rs = _normalize_random_state(random_state)
    if isinstance(weights, str):
        weights = normalize_col_name(weights)

    # Derive per-group seeds from the base seed + a deterministic digest of
    # the group's row indices (tokenize_int = MurmurHash, stable across
    # processes). A single closed-over RandomState would be pickled per task
    # and replayed identically in every partition.
    base_seed = int(rs.randint(0, 2**31 - 1)) if rs is not None else None

    def _udf(g, _n=n, _frac=frac, _replace=replace, _weights=weights, _seed=base_seed):
        _rs = (
            np.random.RandomState(
                (_seed + tokenize_int(g.iloc[0].to_dict())) % (2**31)
            )
            if _seed is not None and len(g) > 0
            else None
        )
        kw = dict(replace=_replace, weights=_weights, random_state=_rs)
        return g.sample(frac=_frac, **kw) if _frac is not None else g.sample(n=_n, **kw)

    return self.apply_chunk(
        _udf,
        dtypes=effective_groupby_input_dtypes(self._liteframe, self.groupby_params),
    )
