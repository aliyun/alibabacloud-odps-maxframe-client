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

import inspect
from typing import List, MutableMapping, Union

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import EntityData, OutputType
from ...serialization.serializables import AnyField, KeyField, StringField
from ...udf import BuiltinFunction, MarkedFunction, ODPSFunction
from ...utils import make_dtype, quiet_stdio
from ..core import SERIES_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import build_series, copy_func_scheduling_hints


class DataFrameMap(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.MAP

    input = KeyField("input", default=None)
    arg = AnyField("arg", default=None)
    na_action = StringField("na_action", default=None)

    def __init__(self, output_types=None, memory_scale=None, **kw):
        super().__init__(_output_types=output_types, _memory_scale=memory_scale, **kw)
        if not self.output_types:
            self.output_types = [OutputType.series]
        if hasattr(self, "arg"):
            self.arg = ODPSFunction.wrap(self.arg)
            copy_func_scheduling_hints(self.arg, self)

    @classmethod
    def _set_inputs(cls, op: "DataFrameMap", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op.input = op._inputs[0]
        if len(inputs) == 2:
            op.arg = op._inputs[1]

    def has_custom_code(self) -> bool:
        return not isinstance(
            self.arg, (dict, SERIES_TYPE, pd.Series)
        ) and not isinstance(self.arg, BuiltinFunction)

    def __call__(self, series, dtype, skip_infer=False):
        if dtype is not None:
            dtype = make_dtype(dtype)
        else:
            # obtain dtype from existing hints
            if isinstance(self.arg, ODPSFunction):
                if self.arg.result_dtype is not None:
                    dtype = self.arg.result_dtype
            elif callable(self.arg):
                # arg is a function, try to inspect the signature
                sig = inspect.signature(self.arg)
                return_type = sig.return_annotation
                if return_type is not inspect._empty:
                    dtype = np.dtype(return_type)

        err_prefix = None
        if dtype is None and not skip_infer:
            inferred_dtype = None
            if callable(self.arg):
                try:
                    with quiet_stdio():
                        # try to infer dtype by calling the function
                        inferred_dtype = (
                            build_series(series)
                            .map(self.arg, na_action=self.na_action)
                            .dtype
                        )
                except:  # noqa: E722  # nosec
                    pass
            else:
                if isinstance(self.arg, MutableMapping):
                    inferred_dtype = pd.Series(self.arg).dtype
                else:
                    inferred_dtype = self.arg.dtype
            if inferred_dtype is not None and np.issubdtype(inferred_dtype, np.number):
                if np.issubdtype(inferred_dtype, np.inexact):
                    # for the inexact e.g. float
                    # we can make the decision,
                    # but for int, due to the nan which may occur,
                    # we cannot infer the dtype
                    dtype = inferred_dtype
                else:
                    err_prefix = "int type may not be exact"
            else:
                dtype = inferred_dtype

        if dtype is None:
            if not skip_infer:
                err_prefix = err_prefix or "cannot infer dtype"
                raise ValueError(
                    f"{err_prefix}, it needs to be specified manually for `map`"
                )
        else:
            dtype = np.int64 if dtype is int else dtype
            dtype = np.dtype(dtype)

        inputs = [series]
        if isinstance(self.arg, SERIES_TYPE):
            inputs.append(self.arg)

        if isinstance(series, SERIES_TYPE):
            return self.new_series(
                inputs,
                shape=series.shape,
                dtype=dtype,
                index_value=series.index_value,
                name=series.name,
            )
        else:
            return self.new_index(
                inputs,
                shape=series.shape,
                dtype=dtype,
                index_value=series.index_value,
                name=series.name,
            )

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "DataFrameMap"
    ) -> None:
        if isinstance(op.arg, MarkedFunction):
            ctx[op.outputs[0].key] = float("inf")
        super().estimate_size(ctx, op)


def series_map(
    series, arg, na_action=None, dtype=None, memory_scale=None, skip_infer=False
):
    """
    Map values of Series according to input correspondence.

    Used for substituting each value in a Series with another value,
    that may be derived from a function, a ``dict`` or
    a :class:`Series`.

    Parameters
    ----------
    arg : function, collections.abc.Mapping subclass or Series
        Mapping correspondence.
    na_action : {None, 'ignore'}, default None
        If 'ignore', propagate NaN values, without passing them to the
        mapping correspondence.
    dtype : np.dtype, default None
        Specify return type of the function. Must be specified when
        we cannot decide the return type of the function.
    memory_scale : float
        Specify the scale of memory uses in the function versus
        input size.
    skip_infer: bool, default False
        Whether infer dtypes when dtypes or output_type is not specified

    Returns
    -------
    Series
        Same index as caller.

    See Also
    --------
    Series.apply : For applying more complex functions on a Series.
    DataFrame.apply : Apply a function row-/column-wise.
    DataFrame.applymap : Apply a function elementwise on a whole DataFrame.

    Notes
    -----
    When ``arg`` is a dictionary, values in Series that are not in the
    dictionary (as keys) are converted to ``NaN``. However, if the
    dictionary is a ``dict`` subclass that defines ``__missing__`` (i.e.
    provides a method for default values), then this default is used
    rather than ``NaN``.

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> import maxframe.dataframe as md
    >>> s = md.Series(['cat', 'dog', mt.nan, 'rabbit'])
    >>> s.execute()
    0      cat
    1      dog
    2      NaN
    3   rabbit
    dtype: object

    ``map`` accepts a ``dict`` or a ``Series``. Values that are not found
    in the ``dict`` are converted to ``NaN``, unless the dict has a default
    value (e.g. ``defaultdict``):

    >>> s.map({'cat': 'kitten', 'dog': 'puppy'}).execute()
    0   kitten
    1    puppy
    2      NaN
    3      NaN
    dtype: object

    It also accepts a function:

    >>> s.map('I am a {}'.format).execute()
    0       I am a cat
    1       I am a dog
    2       I am a nan
    3    I am a rabbit
    dtype: object

    To avoid applying the function to missing values (and keep them as
    ``NaN``) ``na_action='ignore'`` can be used:

    >>> s.map('I am a {}'.format, na_action='ignore').execute()
    0     I am a cat
    1     I am a dog
    2            NaN
    3  I am a rabbit
    dtype: object
    """
    op = DataFrameMap(arg=arg, na_action=na_action, memory_scale=memory_scale)
    return op(series, dtype=dtype, skip_infer=skip_infer)


def index_map(
    idx, mapper, na_action=None, dtype=None, memory_scale=None, skip_infer=False
):
    """
    Map values using input correspondence (a dict, Series, or function).

    Parameters
    ----------
    mapper : function, dict, or Series
        Mapping correspondence.
    na_action : {None, 'ignore'}
        If 'ignore', propagate NA values, without passing them to the
        mapping correspondence.
    dtype : np.dtype, default None
        Specify return type of the function. Must be specified when
        we cannot decide the return type of the function.
    memory_scale : float
        Specify the scale of memory uses in the function versus
        input size.
    skip_infer: bool, default False
        Whether infer dtypes when dtypes or output_type is not specified


    Returns
    -------
    applied : Union[Index, MultiIndex], inferred
        The output of the mapping function applied to the index.
        If the function returns a tuple with more than one element
        a MultiIndex will be returned.
    """
    op = DataFrameMap(arg=mapper, na_action=na_action, memory_scale=memory_scale)
    return op(idx, dtype=dtype, skip_infer=skip_infer)


def df_map(
    df, func, na_action=None, dtypes=None, dtype=None, skip_infer=False, **kwargs
):
    """
    Apply a function to a Dataframe elementwise.

    This method applies a function that accepts and returns a scalar
    to every element of a DataFrame.

    Parameters
    ----------
    func : callable
        Python function, returns a single value from a single value.
    na_action : {None, 'ignore'}, default None
        If 'ignore', propagate NaN values, without passing them to func.
    dtypes : Series, default None
        Specify dtypes of returned DataFrames.
    dtype : np.dtype, default None
        Specify dtypes of all columns of returned DataFrames, only
        effective when dtypes is not specified.
    skip_infer: bool, default False
        Whether infer dtypes when dtypes or dtype is not specified.
    **kwargs
        Additional keyword arguments to pass as keywords arguments to
        `func`.

    Returns
    -------
    DataFrame
        Transformed DataFrame.

    See Also
    --------
    DataFrame.apply : Apply a function along input axis of DataFrame.
    DataFrame.replace: Replace values given in `to_replace` with `value`.
    Series.map : Apply a function elementwise on a Series.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([[1, 2.12], [3.356, 4.567]])
    >>> df.execute()
           0      1
    0  1.000  2.120
    1  3.356  4.567

    >>> df.map(lambda x: len(str(x))).execute()
       0  1
    0  3  4
    1  5  5

    Like Series.map, NA values can be ignored:

    >>> df_copy = df.copy()
    >>> df_copy.iloc[0, 0] = md.NA
    >>> df_copy.map(lambda x: len(str(x)), na_action='ignore').execute()
         0  1
    0  NaN  4
    1  5.0  5

    It is also possible to use `map` with functions that are not
    `lambda` functions:

    >>> df.map(round, ndigits=1).execute()
         0    1
    0  1.0  2.1
    1  3.4  4.6

    Note that a vectorized version of `func` often exists, which will
    be much faster. You could square each number elementwise.

    >>> df.map(lambda x: x**2).execute()
               0          1
    0   1.000000   4.494400
    1  11.262736  20.857489

    But it's better to avoid map in that case.

    >>> (df ** 2).execute()
               0          1
    0   1.000000   4.494400
    1  11.262736  20.857489
    """
    if dtypes is None and dtype is not None:
        dtypes = pd.Series([dtype] * df.shape[1], index=df.dtypes.index)

    def _wrapper(row):
        return row.map(func, na_action=na_action, **kwargs)

    return df.apply(
        _wrapper, axis=1, dtypes=dtypes, skip_infer=skip_infer, elementwise=True
    )
