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

import copy
from typing import List

import numpy as np

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData, get_output_types
from ...serialization.serializables import (
    AnyField,
    BoolField,
    Float64Field,
    Int8Field,
    Int64Field,
    KeyField,
)
from ...tensor.random import RandomStateField
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index, validate_axis


class DataFrameSample(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.RAND_SAMPLE

    size = Int64Field("size", default=None)
    frac = Float64Field("frac", default=None)
    replace = BoolField("replace", default=False)
    weights = AnyField("weights", default=None)
    axis = Int8Field("axis", default=None)
    seed = Int64Field("seed", default=None)
    random_state = RandomStateField("random_state", default=None)
    always_multinomial = BoolField("always_multinomial", default=None)

    # for chunks
    # num of instances for chunks
    chunk_samples = KeyField("chunk_samples", default=None)

    def __init__(self, random_state=None, seed=None, **kw):
        if random_state is None:
            random_state = np.random.RandomState(seed)
        super().__init__(random_state=random_state, seed=seed, **kw)

    @classmethod
    def _set_inputs(cls, op: "DataFrameSample", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        it = iter(inputs)
        next(it)
        if isinstance(op.weights, ENTITY_TYPE):
            op.weights = next(it)
        if isinstance(op.chunk_samples, ENTITY_TYPE):
            op.chunk_samples = next(it)

    def __call__(self, df):
        params = df.params
        new_shape = list(df.shape)

        if self.size is not None:
            new_shape[self.axis] = self.size
        elif self.frac is not None:
            new_shape[self.axis] = np.nan

        params["shape"] = tuple(new_shape)
        params["index_value"] = parse_index(df.index_value.to_pandas()[:0])

        input_dfs = [df]
        if isinstance(self.weights, ENTITY_TYPE):
            input_dfs.append(self.weights)

        self._output_types = get_output_types(df)
        return self.new_tileable(input_dfs, **params)


def sample(
    df_or_series,
    n=None,
    frac=None,
    replace=False,
    weights=None,
    random_state=None,
    axis=None,
    always_multinomial=False,
):
    """
    Return a random sample of items from an axis of object.

    You can use `random_state` for reproducibility.

    Parameters
    ----------
    n : int, optional
        Number of items from axis to return. Cannot be used with `frac`.
        Default = 1 if `frac` = None.
    frac : float, optional
        Fraction of axis items to return. Cannot be used with `n`.
    replace : bool, default False
        Allow or disallow sampling of the same row more than once.
    weights : str or ndarray-like, optional
        Default 'None' results in equal probability weighting.
        If passed a Series, will align with target object on index. Index
        values in weights not found in sampled object will be ignored and
        index values in sampled object not in weights will be assigned
        weights of zero.
        If called on a DataFrame, will accept the name of a column
        when axis = 0.
        Unless weights are a Series, weights must be same length as axis
        being sampled.
        If weights do not sum to 1, they will be normalized to sum to 1.
        Missing values in the weights column will be treated as zero.
        Infinite values not allowed.
    random_state : int, array-like, BitGenerator, np.random.RandomState, optional
        If int, array-like, or BitGenerator (NumPy>=1.17), seed for
        random number generator
        If np.random.RandomState, use as numpy RandomState object.
    axis : {0 or ‘index’, 1 or ‘columns’, None}, default None
        Axis to sample. Accepts axis number or name. Default is stat axis
        for given data type (0 for Series and DataFrames).
    always_multinomial : bool, default False
        If True, always treat distribution of sample counts between data chunks
        as multinomial distribution. This will accelerate sampling when data
        is huge, but may affect randomness of samples when number of instances
        is not very large.

    Returns
    -------
    Series or DataFrame
        A new object of same type as caller containing `n` items randomly
        sampled from the caller object.

    See Also
    --------
    DataFrameGroupBy.sample: Generates random samples from each group of a
        DataFrame object.
    SeriesGroupBy.sample: Generates random samples from each group of a
        Series object.
    numpy.random.choice: Generates a random sample from a given 1-D numpy
        array.

    Notes
    -----
    If `frac` > 1, `replacement` should be set to `True`.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'num_legs': [2, 4, 8, 0],
    ...                    'num_wings': [2, 0, 0, 0],
    ...                    'num_specimen_seen': [10, 2, 1, 8]},
    ...                   index=['falcon', 'dog', 'spider', 'fish'])
    >>> df.execute()
            num_legs  num_wings  num_specimen_seen
    falcon         2          2                 10
    dog            4          0                  2
    spider         8          0                  1
    fish           0          0                  8

    Extract 3 random elements from the ``Series`` ``df['num_legs']``:
    Note that we use `random_state` to ensure the reproducibility of
    the examples.

    >>> df['num_legs'].sample(n=3, random_state=1).execute()
    fish      0
    spider    8
    falcon    2
    Name: num_legs, dtype: int64

    A random 50% sample of the ``DataFrame`` with replacement:

    >>> df.sample(frac=0.5, replace=True, random_state=1).execute()
          num_legs  num_wings  num_specimen_seen
    dog          4          0                  2
    fish         0          0                  8

    An upsample sample of the ``DataFrame`` with replacement:
    Note that `replace` parameter has to be `True` for `frac` parameter > 1.

    >>> df.sample(frac=2, replace=True, random_state=1).execute()
            num_legs  num_wings  num_specimen_seen
    dog            4          0                  2
    fish           0          0                  8
    falcon         2          2                 10
    falcon         2          2                 10
    fish           0          0                  8
    dog            4          0                  2
    fish           0          0                  8
    dog            4          0                  2

    Using a DataFrame column as weights. Rows with larger value in the
    `num_specimen_seen` column are more likely to be sampled.

    >>> df.sample(n=2, weights='num_specimen_seen', random_state=1).execute()
            num_legs  num_wings  num_specimen_seen
    falcon         2          2                 10
    fish           0          0                  8
    """
    if frac and n:
        raise ValueError("Please enter a value for `frac` OR `n`, not both.")

    axis = validate_axis(axis or 0, df_or_series)
    if axis == 1:
        raise NotImplementedError("Currently cannot sample over columns")
    if frac is not None and frac < 0 or n is not None and n < 0:
        raise ValueError(
            "A negative number of rows requested. Please provide positive value."
        )
    rs = copy.deepcopy(
        random_state.to_numpy() if hasattr(random_state, "to_numpy") else random_state
    )
    if isinstance(rs, (int, np.ndarray)):
        rs = np.random.RandomState(rs)
    op = DataFrameSample(
        size=n,
        frac=frac,
        replace=replace,
        weights=weights,
        random_state=rs,
        axis=axis,
        always_multinomial=always_multinomial,
    )
    return op(df_or_series)
