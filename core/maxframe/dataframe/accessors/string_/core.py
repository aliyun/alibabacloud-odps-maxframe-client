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
import pandas as pd

from .... import opcodes
from ....core import EntityData, OutputType
from ....serialization.serializables import DictField, KeyField, StringField, TupleField
from ....tensor import tensor as astensor
from ....tensor.core import TENSOR_TYPE
from ...core import SERIES_TYPE
from ...initializer import Series as asseries
from ...operators import DataFrameOperator, DataFrameOperatorMixin
from ...utils import build_empty_series, infer_index_value, parse_index


class SeriesStringMethod(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.STRING_METHOD

    _input = KeyField("input")
    method = StringField("method", default=None)
    method_args = TupleField("method_args", default=None)
    method_kwargs = DictField("method_kwargs", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if not self.output_types:
            self.output_types = [OutputType.series]

    @property
    def input(self):
        return self._input

    @classmethod
    def _set_inputs(cls, op: "SeriesStringMethod", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        op._input = op._inputs[0]
        if len(op._inputs) == 2:
            # for method cat
            op.method_kwargs["others"] = op._inputs[1]

    def __call__(self, inp):
        return string_method_to_handlers[self.method].call(self, inp)


class SeriesStringMethodBaseHandler:
    @classmethod
    def call(cls, op: SeriesStringMethod, inp):
        if op.method == "__getitem__":
            op.method = "slice"
            kwargs = {
                "start": op.method_args[0].start,
                "stop": op.method_args[0].stop,
                "step": op.method_args[0].step,
            }
            op.method_args = ()
            op.method_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        empty_series = build_empty_series(inp.dtype)
        dtype = getattr(empty_series.str, op.method)(
            *op.method_args, **op.method_kwargs
        ).dtype
        return op.new_series(
            [inp],
            shape=inp.shape,
            dtype=dtype,
            index_value=inp.index_value,
            name=inp.name,
        )


class SeriesStringSplitHandler(SeriesStringMethodBaseHandler):
    @classmethod
    def call(cls, op, inp):
        method_kwargs = op.method_kwargs
        if method_kwargs.get("expand", False) is False:
            return super().call(op, inp)
        n = method_kwargs.get("n", -1)
        # does not support if expand and n == -1
        if n == -1:  # pragma: no cover
            raise NotImplementedError("`n` needs to be specified when expand=True")

        op.output_types = [OutputType.dataframe]
        columns = pd.RangeIndex(n + 1)
        columns_value = parse_index(columns, store_data=True)
        dtypes = pd.Series([inp.dtype] * len(columns), index=columns)
        return op.new_dataframe(
            [inp],
            shape=(inp.shape[0], len(columns)),
            dtypes=dtypes,
            columns_value=columns_value,
            index_value=inp.index_value,
        )


class SeriesStringCatHandler(SeriesStringMethodBaseHandler):
    CAT_TYPE_ERROR = (
        "others must be Series, Index, DataFrame, "
        "Tensor, np.ndarrary or list-like "
        "(either containing only strings or "
        "containing only objects of "
        "type Series/Index/Tensor/np.ndarray[1-dim])"
    )
    CAT_LEN_ERROR = (
        "If `others` contains arrays or lists (or other list-likes without an index), "
        "these must all be of the same length as the calling Series/Index."
    )

    @classmethod
    def call(cls, op, inp):
        method_kwargs = op.method_kwargs
        others = method_kwargs.get("others")

        if others is None:
            from ...reduction import build_str_concat_object

            return build_str_concat_object(
                inp,
                sep=op.method_kwargs.get("sep"),
                na_rep=op.method_kwargs.get("na_rep"),
            )
        elif isinstance(others, (tuple, list, np.ndarray, TENSOR_TYPE)):
            others = astensor(others, dtype=object)
            if others.ndim != 1:
                raise TypeError(cls.CAT_TYPE_ERROR)
            if (
                not np.isnan(inp.shape[0])
                and not np.isnan(others.shape[0])
                and inp.shape[0] != others.shape[0]
            ):
                raise ValueError(cls.CAT_LEN_ERROR)
            inputs = [inp]
            if isinstance(others, TENSOR_TYPE):
                inputs.append(others)
            return op.new_series(
                inputs,
                shape=inp.shape,
                dtype=inp.dtype,
                index_value=inp.index_value,
                name=inp.name,
            )
        elif isinstance(others, (pd.Series, SERIES_TYPE)):
            others = asseries(others)
            if op.method_kwargs.get("join") != "outer":  # pragma: no cover
                raise NotImplementedError("only outer join supported for now")
            return op.new_series(
                [inp, others],
                shape=inp.shape,
                dtype=inp.dtype,
                index_value=infer_index_value(inp.index_value, others.index_value),
                name=inp.name,
            )
        elif isinstance(others, str) and op.method_kwargs.get("sep") is None:
            raise ValueError("Did you mean to supply a `sep` keyword?")
        else:
            raise TypeError(cls.CAT_TYPE_ERROR)


class SeriesStringExtractHandler(SeriesStringMethodBaseHandler):
    @classmethod
    def call(cls, op, inp):
        empty_series = build_empty_series(
            inp.dtype, index=inp.index_value.to_pandas()[:0]
        )
        test_df = getattr(empty_series.str, op.method)(
            *op.method_args, **op.method_kwargs
        )
        if test_df.ndim == 1:
            return op.new_series(
                [inp],
                shape=inp.shape,
                dtype=test_df.dtype,
                index_value=inp.index_value,
                name=inp.name,
            )
        else:
            op.output_types = [OutputType.dataframe]
            if op.method == "extractall":
                index_value = parse_index(test_df.index, inp)
                shape = (np.nan, test_df.shape[1])
            else:
                index_value = inp.index_value
                shape = (inp.shape[0], test_df.shape[1])
            return op.new_dataframe(
                [inp],
                shape=shape,
                dtypes=test_df.dtypes,
                index_value=index_value,
                columns_value=parse_index(test_df.columns, store_data=True),
            )


string_method_to_handlers = {}
_not_implements = ["get_dummies"]
# start to register handlers for string methods
# register special methods first
string_method_to_handlers["split"] = SeriesStringSplitHandler
string_method_to_handlers["rsplit"] = SeriesStringSplitHandler
string_method_to_handlers["cat"] = SeriesStringCatHandler
string_method_to_handlers["extract"] = SeriesStringExtractHandler
string_method_to_handlers["extractall"] = SeriesStringExtractHandler
# then come to the normal methods
for method in dir(pd.Series.str):
    if method.startswith("_") and method != "__getitem__":
        continue
    if method in _not_implements:
        continue
    if method in string_method_to_handlers:
        continue
    string_method_to_handlers[method] = SeriesStringMethodBaseHandler
