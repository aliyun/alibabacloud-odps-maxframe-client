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

import functools
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa

from .... import opcodes
from ....config import options
from ....core import EntityData, OutputType
from ....lib.dtypes_extension import ArrowDtype
from ....serialization.serializables import DictField, KeyField, StringField, TupleField
from ....tensor import tensor as astensor
from ....tensor.core import TENSOR_TYPE
from ....utils import wrap_arrow_dtype
from ...core import SERIES_TYPE
from ...initializer import Series as asseries
from ...operators import DataFrameOperator, DataFrameOperatorMixin
from ...utils import build_series, infer_index_value, parse_index


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
    @staticmethod
    def _get_mock_method_dtype(in_dtype, method: str, *args, **kwargs):
        dummy_series = build_series(pd.Series([], dtype=in_dtype))
        if isinstance(in_dtype, ArrowDtype):
            dummy_series = dummy_series.astype("O")
        ret_series = getattr(dummy_series.str, method)(*args, **kwargs)
        if isinstance(in_dtype, ArrowDtype):
            ret_arrow = pa.array(ret_series)
            ret_series = pd.Series(ret_arrow, dtype=wrap_arrow_dtype(ret_arrow.type))
        return ret_series.dtype

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _get_cached_method_dtype(in_dtype, method: str):
        return SeriesStringMethodBaseHandler._get_mock_method_dtype(in_dtype, method)

    @classmethod
    def _get_method_dtype(cls, in_dtype, method: str, *args, **kwargs):
        if method == "encode":
            return wrap_arrow_dtype(pa.binary())
        elif method == "decode":
            return (
                wrap_arrow_dtype(pa.string())
                if options.dataframe.dtype_backend == "pyarrow"
                else np.dtype("object")
            )

        if not args and not kwargs:
            return cls._get_cached_method_dtype(in_dtype, method)
        return cls._get_mock_method_dtype(in_dtype, method, *args, **kwargs)

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

        dtype = cls._get_method_dtype(
            inp.dtype, op.method, *op.method_args, **op.method_kwargs
        )
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

        op.output_types = [OutputType.dataframe]

        if n >= 0:
            columns = pd.RangeIndex(n + 1)
            n_cols = n + 1
            columns_value = parse_index(columns, store_data=True)
            dtypes = pd.Series([inp.dtype] * len(columns), index=columns)
        else:
            dtypes = columns_value = None
            n_cols = np.nan

        return op.new_dataframe(
            [inp],
            shape=(inp.shape[0], n_cols),
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
        mock_series = build_series(inp)
        test_df = getattr(mock_series.str, op.method)(
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
for _method in dir(pd.Series.str):
    if _method.startswith("_") and _method != "__getitem__":
        continue
    if _method in _not_implements:
        continue
    if _method in string_method_to_handlers:
        continue
    string_method_to_handlers[_method] = SeriesStringMethodBaseHandler
del _method
