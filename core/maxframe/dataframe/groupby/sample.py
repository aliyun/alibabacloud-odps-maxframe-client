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
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData, OutputType, get_output_types
from ...serialization.serializables import (
    BoolField,
    DictField,
    Float32Field,
    Int32Field,
    Int64Field,
    KeyField,
    NDArrayField,
    StringField,
)
from ...tensor.random import RandomStateField
from ..initializer import Series as asseries
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class GroupBySample(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.RAND_SAMPLE
    _op_module_ = "dataframe.groupby"

    groupby_params = DictField("groupby_params", default=None)
    size = Int64Field("size", default=None)
    frac = Float32Field("frac", default=None)
    replace = BoolField("replace", default=None)
    weights = KeyField("weights", default=None)
    seed = Int32Field("seed", default=None)
    _random_state = RandomStateField("random_state", default=None)
    errors = StringField("errors", default=None)
    # for chunks
    # num of instances for chunks
    input_nsplits = NDArrayField("input_nsplits", default=None)

    def __init__(self, random_state=None, **kw):
        super().__init__(_random_state=random_state, **kw)

    @property
    def random_state(self):
        return self._random_state

    @classmethod
    def _set_inputs(cls, op: "GroupBySample", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        input_iter = iter(inputs)
        next(input_iter)
        if isinstance(op.weights, ENTITY_TYPE):
            op.weights = next(input_iter)

    def __call__(self, groupby):
        df = groupby
        while df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            df = df.inputs[0]

        selection = groupby.op.groupby_params.pop("selection", None)
        if df.ndim > 1 and selection:
            if isinstance(selection, tuple) and selection not in df.dtypes:
                selection = list(selection)
            result_df = df[selection]
        else:
            result_df = df

        params = result_df.params
        params["shape"] = (
            (np.nan,) if result_df.ndim == 1 else (np.nan, result_df.shape[-1])
        )
        params["index_value"] = parse_index(result_df.index_value.to_pandas()[:0])

        input_dfs = [df]
        if isinstance(self.weights, ENTITY_TYPE):
            input_dfs.append(self.weights)

        self._output_types = get_output_types(result_df)
        return self.new_tileable(input_dfs, **params)


def groupby_sample(
    groupby,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    replace: bool = False,
    weights: Union[Sequence, pd.Series, None] = None,
    random_state: Optional[np.random.RandomState] = None,
    errors: str = "ignore",
):
    """
    Return a random sample of items from each group.

    You can use `random_state` for reproducibility.

    Parameters
    ----------
    n : int, optional
        Number of items to return for each group. Cannot be used with
        `frac` and must be no larger than the smallest group unless
        `replace` is True. Default is one if `frac` is None.
    frac : float, optional
        Fraction of items to return. Cannot be used with `n`.
    replace : bool, default False
        Allow or disallow sampling of the same row more than once.
    weights : list-like, optional
        Default None results in equal probability weighting.
        If passed a list-like then values must have the same length as
        the underlying DataFrame or Series object and will be used as
        sampling probabilities after normalization within each group.
        Values must be non-negative with at least one positive element
        within each group.
    random_state : int, array-like, BitGenerator, np.random.RandomState, optional
        If int, array-like, or BitGenerator (NumPy>=1.17), seed for
        random number generator
        If np.random.RandomState, use as numpy RandomState object.
    errors : {'ignore', 'raise'}, default 'ignore'
        If ignore, errors will not be raised when `replace` is False
        and size of some group is less than `n`.

    Returns
    -------
    Series or DataFrame
        A new object of same type as caller containing items randomly
        sampled within each group from the caller object.

    See Also
    --------
    DataFrame.sample: Generate random samples from a DataFrame object.
    numpy.random.choice: Generate a random sample from a given 1-D numpy
        array.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame(
    ...     {"a": ["red"] * 2 + ["blue"] * 2 + ["black"] * 2, "b": range(6)}
    ... )
    >>> df.execute()
           a  b
    0    red  0
    1    red  1
    2   blue  2
    3   blue  3
    4  black  4
    5  black  5

    Select one row at random for each distinct value in column a. The
    `random_state` argument can be used to guarantee reproducibility:

    >>> df.groupby("a").sample(n=1, random_state=1).execute()
           a  b
    4  black  4
    2   blue  2
    1    red  1

    Set `frac` to sample fixed proportions rather than counts:

    >>> df.groupby("a")["b"].sample(frac=0.5, random_state=2).execute()
    5    5
    2    2
    0    0
    Name: b, dtype: int64

    Control sample probabilities within groups by setting weights:

    >>> df.groupby("a").sample(
    ...     n=1,
    ...     weights=[1, 1, 1, 0, 0, 1],
    ...     random_state=1,
    ... ).execute()
           a  b
    5  black  5
    2   blue  2
    0    red  0
    """
    groupby_params = groupby.op.groupby_params.copy()
    groupby_params.pop("as_index", None)

    if weights is not None and not isinstance(weights, ENTITY_TYPE):
        weights = asseries(weights)

    n = 1 if n is None and frac is None else n
    rs = copy.deepcopy(
        random_state.to_numpy() if hasattr(random_state, "to_numpy") else random_state
    )
    if not isinstance(rs, np.random.RandomState):  # pragma: no cover
        rs = np.random.RandomState(rs)

    op = GroupBySample(
        size=n,
        frac=frac,
        replace=replace,
        weights=weights,
        random_state=rs,
        groupby_params=groupby_params,
        errors=errors,
    )
    return op(groupby)
