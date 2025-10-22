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

from functools import partial
from typing import List

from .. import opcodes
from ..core import ENTITY_TYPE, EntityData
from ..core.operator import ObjectOperator, ObjectOperatorMixin
from ..dataframe.core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE
from ..serialization.serializables import (
    BoolField,
    DictField,
    FunctionField,
    Int32Field,
    ListField,
)
from ..tensor.core import TENSOR_TYPE
from ..typing_ import TileableType
from ..udf import BuiltinFunction
from ..utils import find_objects, replace_objects


class RemoteFunction(ObjectOperatorMixin, ObjectOperator):
    _op_type_ = opcodes.REMOTE_FUNCATION
    _op_module_ = "remote"

    function = FunctionField("function")
    function_args = ListField("function_args")
    function_kwargs = DictField("function_kwargs")
    retry_when_fail = BoolField("retry_when_fail")
    resolve_tileable_input = BoolField("resolve_tileable_input", default=False)
    n_output = Int32Field("n_output", default=None)

    @property
    def output_limit(self):
        return self.n_output or 1

    @property
    def retryable(self) -> bool:
        return self.retry_when_fail

    @classmethod
    def _no_prepare(cls, tileable):
        return isinstance(
            tileable, (TENSOR_TYPE, DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE)
        )

    def has_custom_code(self) -> bool:
        return not isinstance(self.function, BuiltinFunction)

    def check_inputs(self, inputs: List[TileableType]):
        return

    @classmethod
    def _set_inputs(cls, op: "RemoteFunction", inputs: List[EntityData]):
        raw_inputs = getattr(op, "_inputs", None)
        super()._set_inputs(op, inputs)

        function_inputs = iter(inp for inp in op._inputs)
        mapping = {inp: new_inp for inp, new_inp in zip(inputs, op._inputs)}
        if raw_inputs is not None:
            for raw_inp in raw_inputs:
                mapping[raw_inp] = next(function_inputs)
        op.function_args = replace_objects(op.function_args, mapping)
        op.function_kwargs = replace_objects(op.function_kwargs, mapping)

    def __call__(self):
        find_inputs = partial(find_objects, types=ENTITY_TYPE)
        inputs = find_inputs(self.function_args) + find_inputs(self.function_kwargs)
        if self.n_output is None:
            return self.new_tileable(inputs)
        else:
            return self.new_tileables(
                inputs, kws=[dict(i=i) for i in range(self.n_output)]
            )


def spawn(
    func,
    args=(),
    kwargs=None,
    retry_when_fail=True,
    resolve_tileable_input=False,
    n_output=None,
    **kw,
):
    """
    Spawn a function and return a MaxFrame Object which can be executed later.

    Parameters
    ----------
    func : function
        Function to spawn.
    args: tuple
       Args to pass to function
    kwargs: dict
       Kwargs to pass to function
    retry_when_fail: bool, default False
       If True, retry when function failed.
    resolve_tileable_input: bool default False
       If True, resolve tileable inputs as values.
    n_output: int
       Count of outputs for the function

    Returns
    -------
    Object
        MaxFrame Object.

    Examples
    --------
    >>> import maxframe.remote as mr
    >>> def inc(x):
    >>>     return x + 1
    >>>
    >>> result = mr.spawn(inc, args=(0,))
    >>> result
    Object <op=RemoteFunction, key=e0b31261d70dd9b1e00da469666d72d9>
    >>> result.execute().fetch()
    1

    List of spawned functions can be converted to :class:`maxframe.remote.ExecutableTuple`,
    and `.execute()` can be called to run together.

    >>> results = [mr.spawn(inc, args=(i,)) for i in range(10)]
    >>> mr.ExecutableTuple(results).execute().fetch()
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    MaxFrame Object returned by :meth:`maxframe.remote.spawn` can be treated
    as arguments for other spawn functions.

    >>> results = [mr.spawn(inc, args=(i,)) for i in range(10)]   # list of spawned functions
    >>> def sum_all(xs):
            return sum(xs)
    >>> mr.spawn(sum_all, args=(results,)).execute().fetch()
    55

    inside a spawned function, new functions can be spawned.

    >>> def driver():
    >>>     results = [mr.spawn(inc, args=(i,)) for i in range(10)]
    >>>     return mr.ExecutableTuple(results).execute().fetch()
    >>>
    >>> mr.spawn(driver).execute().fetch()
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    MaxFrame tensor, DataFrame and so forth is available in spawned functions as well.

    >>> import maxframe.tensor as mt
    >>> def driver2():
    >>>     t = mt.random.rand(10, 10)
    >>>     return t.sum().to_numpy()
    >>>
    >>> mr.spawn(driver2).execute().fetch()
    52.47844223908132

    Argument of `n_output` can indicate that the spawned function will return multiple outputs.
    This is important when some of the outputs may be passed to different functions.

    >>> def triage(alist):
    >>>     ret = [], []
    >>>     for i in alist:
    >>>         if i < 0.5:
    >>>             ret[0].append(i)
    >>>         else:
    >>>             ret[1].append(i)
    >>>     return ret
    >>>
    >>> def sum_all(xs):
    >>>     return sum(xs)
    >>>
    >>> l = [0.4, 0.7, 0.2, 0.8]
    >>> la, lb = mr.spawn(triage, args=(l,), n_output=2)
    >>>
    >>> sa = mr.spawn(sum_all, args=(la,))
    >>> sb = mr.spawn(sum_all, args=(lb,))
    >>> mr.ExecutableTuple([sa, sb]).execute().fetch()
    >>> [0.6000000000000001, 1.5]
    """
    if not isinstance(args, tuple):
        args = [args]
    else:
        args = list(args)
    if kwargs is None:
        kwargs = dict()
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs has to be a dict")

    op = RemoteFunction(
        function=func,
        function_args=args,
        function_kwargs=kwargs,
        retry_when_fail=retry_when_fail,
        resolve_tileable_input=resolve_tileable_input,
        n_output=n_output,
        **kw,
    )
    if op.extra_params:
        raise ValueError(f"Unexpected kw: {list(op.extra_params)[0]}")
    return op()
