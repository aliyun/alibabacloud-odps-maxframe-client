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

from collections import OrderedDict
from typing import Any, Dict, List, MutableMapping, Union

import numpy as np
import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, EntityData, OutputType
from ...serialization.serializables import AnyField, KeyField
from ...tensor.core import Tensor
from ...tensor.datasource import tensor as astensor
from ...typing_ import EntityType, TileableType
from ..core import INDEX_TYPE, SERIES_TYPE
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class DataFrameFromTensor(DataFrameOperator, DataFrameOperatorMixin):
    """
    Represents data from maxframe tensor
    """

    _op_type_ = opcodes.DATAFRAME_FROM_TENSOR

    input = AnyField("input")
    index = AnyField("index")
    columns = AnyField("columns")
    axis = AnyField("axis")

    def __init__(self, *args, **kwargs):
        kwargs["_output_types"] = [OutputType.dataframe]
        super().__init__(*args, **kwargs)

    @classmethod
    def _set_inputs(cls, op: "DataFrameFromTensor", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs)
        if op.input is not None:
            if not isinstance(op.input, dict):
                op.input = next(inputs_iter)
            else:
                # check each value for input
                new_input = OrderedDict()
                for k, v in op.input.items():
                    if isinstance(v, ENTITY_TYPE):
                        new_input[k] = next(inputs_iter)
                    else:
                        new_input[k] = v
                op.input = new_input

        if isinstance(op.index, ENTITY_TYPE):
            op.index = next(inputs_iter)

    def __call__(
        self,
        input_tensor: Tensor,
        index: Union[TileableType, pd.Index],
        columns: pd.Index,
        dtypes: pd.Series,
    ):
        if isinstance(input_tensor, dict):
            return self._call_input_1d_tileables(input_tensor, index, columns, dtypes)
        elif input_tensor is not None:
            return self._call_input_tensor(input_tensor, index, columns, dtypes)
        else:
            return self._call_tensor_none(index, columns, dtypes)

    def _process_index(
        self, index: Union[TileableType, pd.Index], inputs: List[EntityType]
    ):
        if not isinstance(index, pd.Index):
            if isinstance(index, INDEX_TYPE):
                index_value = index.index_value
                inputs.append(index)
            elif isinstance(index, ENTITY_TYPE):
                index = astensor(index)
                if index.ndim != 1:
                    raise ValueError(f"index should be 1-d, got {index.ndim}-d")
                index_value = parse_index(
                    pd.Index([], dtype=index.dtype), index, type(self).__name__
                )
                inputs.append(index)
            else:
                index = pd.Index(index)
                index_value = parse_index(index)
        else:
            index_value = parse_index(index)
        return index_value

    def _call_input_1d_tileables(
        self,
        input_1d_tileables: Dict[Any, TileableType],
        index: Union[TileableType, pd.Index],
        columns: pd.Index,
        dtypes: pd.Series,
    ):
        tileables = []
        shape = None
        for tileable in input_1d_tileables.values():
            tileable_shape = astensor(tileable).shape
            if len(tileable_shape) > 0:
                if shape is None:
                    shape = tileable_shape
                elif shape != tileable_shape:
                    raise ValueError("input 1-d tensors should have same shape")

            if isinstance(tileable, ENTITY_TYPE):
                tileables.append(tileable)

        if self.axis == 0:
            if index is not None:
                raise NotImplementedError("Cannot accept index when axis=0")
            else:
                index = pd.Index(list(input_1d_tileables.keys()))
                index_value = parse_index(index, store_data=True)
                self.index = index

            if columns is not None:
                tileable_size = tileables[0].shape[0] if tileables else 0
                if not isinstance(columns, pd.Index):
                    columns = self.columns = pd.Index(columns)
                column_size = columns.shape[0]
                if (
                    not pd.isna(tileable_size)
                    and not pd.isna(column_size)
                    and tileable_size != column_size
                ):
                    raise ValueError(
                        f"columns {columns} should have the same shape "
                        f"with tensor: {tileable_size}"
                    )
                columns_value = self._process_index(columns, tileables)
            else:
                if not tileables or np.isnan(tileables[0].shape[0]):
                    columns = columns_value = None
                else:
                    columns = pd.RangeIndex(0, tileables[0].shape[0])
                    columns_value = parse_index(columns, store_data=True)
                self.columns = columns

            shape = (len(input_1d_tileables), shape[0] if shape else 0)
        else:
            if index is not None:
                tileable_size = tileables[0].shape[0] if tileables else 0
                if hasattr(index, "shape"):
                    index_size = index.shape[0]
                else:
                    index_size = len(index)
                if (
                    not pd.isna(tileable_size)
                    and not pd.isna(index_size)
                    and tileable_size != index_size
                ):
                    raise ValueError(
                        f"index {index} should have the same shape "
                        f"with tensor: {tileable_size}"
                    )
                index_value = self._process_index(index, tileables)
            else:
                if not tileables or np.isnan(tileables[0].shape[0]):
                    index = pd.RangeIndex(0)
                else:
                    index = pd.RangeIndex(0, tileables[0].shape[0])
                self.index = index
                index_value = parse_index(index)

            if columns is not None:
                if len(input_1d_tileables) != len(columns):
                    raise ValueError(
                        f"columns {columns} should have size {len(input_1d_tileables)}"
                    )
                if not isinstance(columns, pd.Index):
                    if isinstance(columns, ENTITY_TYPE):
                        raise NotImplementedError(
                            "The columns value cannot be a tileable"
                        )
                    columns = pd.Index(columns)
                columns_value = parse_index(columns, store_data=True)
            else:
                columns_value = parse_index(
                    pd.RangeIndex(0, len(input_1d_tileables)), store_data=True
                )

            shape = (shape[0] if shape else 0, len(input_1d_tileables))

        return self.new_dataframe(
            tileables,
            shape,
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
        )

    def _call_input_tensor(
        self,
        input_tensor: Tensor,
        index: Union[TileableType, pd.Index],
        columns: pd.Index,
        dtypes: pd.Series,
    ):
        if input_tensor.ndim not in {1, 2}:
            raise ValueError("Must pass 1-d or 2-d input")
        inputs = [input_tensor]

        if index is not None:
            if input_tensor.shape[0] != len(index):
                raise ValueError(
                    f"index {index} should have the same shape with tensor: {input_tensor.shape[0]}"
                )
            index_value = self._process_index(index, inputs)
        elif isinstance(input_tensor, SERIES_TYPE):
            index_value = input_tensor.index_value
        else:
            stop = input_tensor.shape[0]
            stop = -1 if np.isnan(stop) else stop
            index = self.index = pd.RangeIndex(start=0, stop=stop)
            index_value = parse_index(index)

        if columns is not None:
            if not (
                input_tensor.ndim == 1
                and len(columns) == 1
                or input_tensor.shape[1] == len(columns)
            ):
                raise ValueError(
                    f"columns {columns} should have the same shape with tensor: {input_tensor.shape[1]}"
                )
            if not isinstance(columns, pd.Index):
                if isinstance(columns, ENTITY_TYPE):
                    raise NotImplementedError("The columns value cannot be a tileable")
                columns = pd.Index(columns)
            columns_value = parse_index(columns, store_data=True)
        else:
            if input_tensor.ndim == 1:
                # convert to 1-d DataFrame
                columns_value = parse_index(
                    pd.RangeIndex(start=0, stop=1), store_data=True
                )
            else:
                columns_value = parse_index(
                    pd.RangeIndex(start=0, stop=input_tensor.shape[1]), store_data=True
                )

        if input_tensor.ndim == 1:
            shape = (input_tensor.shape[0], 1)
        else:
            shape = input_tensor.shape

        return self.new_dataframe(
            inputs,
            shape,
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
        )

    def _call_tensor_none(
        self, index: Union[TileableType, pd.Index], columns: pd.Index, dtypes: pd.Series
    ):
        inputs = []
        shape = []
        if index is not None:
            index_value = self._process_index(index, inputs)
            shape.append(index.shape[0])
        else:
            index = self.index = pd.Index([], dtype=object)
            index_value = parse_index(index)
            shape.append(0)

        if columns is not None:
            if not isinstance(columns, pd.Index):
                if isinstance(columns, ENTITY_TYPE):
                    raise NotImplementedError("The columns value cannot be a tileable")
                columns = pd.Index(columns)
            columns_value = parse_index(columns, store_data=True)
            shape.append(columns.shape[0])
        else:
            columns_value = parse_index(pd.Index([], dtype=object), store_data=True)
            shape.append(0)

        return self.new_dataframe(
            inputs,
            shape=tuple(shape),
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
        )

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "DataFrameFromTensor"
    ):  # pragma: no cover
        # todo implement this to facilitate local computation
        ctx[op.outputs[0].key] = float("inf")


def dataframe_from_tensor(
    tensor: Tensor,
    index: Union[TileableType, pd.Index] = None,
    columns: Union[pd.Index, list] = None,
    gpu: bool = None,
    sparse: bool = False,
):
    if isinstance(columns, list) and columns and isinstance(columns[0], tuple):
        columns = pd.MultiIndex.from_tuples(columns)

    if tensor is not None:
        if tensor.ndim > 2 or tensor.ndim <= 0:
            raise TypeError(
                f"Not support create DataFrame from {tensor.ndim} dims tensor"
            )
        try:
            col_num = tensor.shape[1]
        except IndexError:
            col_num = 1
        gpu = tensor.op.gpu if gpu is None else gpu
        dtypes = pd.Series([tensor.dtype] * col_num, index=columns)
        if columns is None:
            columns = dtypes.index
    else:
        gpu = None
        if columns is not None:
            dtypes = pd.Series([], index=columns)
        else:
            dtypes = pd.Series([], index=pd.Index([], dtype=object))
    if index is not None and not isinstance(index, ENTITY_TYPE):
        index = pd.Index(index)
        if isinstance(index[0], tuple):
            index = pd.MultiIndex.from_tuples(index)
    op = DataFrameFromTensor(
        input=tensor, index=index, columns=columns, gpu=gpu, sparse=sparse
    )
    return op(tensor, index, columns, dtypes)


def dataframe_from_1d_tileables(
    d: Dict[Any, TileableType],
    index: Union[TileableType, pd.Index, list] = None,
    columns: Union[pd.Index, list] = None,
    gpu: bool = None,
    sparse: bool = False,
    axis: int = 1,
):
    from pandas.core.dtypes.cast import find_common_type

    data = dict()
    for k, v in d.items():
        if isinstance(v, (list, tuple)) and any(
            isinstance(sv, ENTITY_TYPE) for sv in v
        ):
            data[k] = astensor(v)
        else:
            data[k] = v
    d = data
    if columns is not None:
        tileables = [d.get(c) for c in columns] if axis == 1 else list(d.values())
    else:
        columns = list(d.keys()) if axis == 1 else None
        tileables = list(d.values())

    gpu = (
        next((t.op.gpu for t in tileables if hasattr(t, "op")), False)
        if gpu is None
        else gpu
    )

    if axis == 0:
        col_num = (
            tileables[0].shape[0]
            if hasattr(tileables[0], "shape")
            else len(tileables[0])
        )
        if pd.isna(col_num):
            dtypes = None
        else:
            common_dtype = find_common_type(
                [
                    t.dtype if hasattr(t, "dtype") else pd.Series(t).dtype
                    for t in tileables
                ]
            )
            dtypes = pd.Series(
                [common_dtype] * col_num,
                index=columns if columns is not None else pd.RangeIndex(col_num),
            )
    else:
        dtypes = pd.Series(
            [t.dtype if hasattr(t, "dtype") else pd.Series(t).dtype for t in tileables],
            index=columns,
        )

    if index is not None and not isinstance(index, ENTITY_TYPE):
        index = pd.Index(index)

    op = DataFrameFromTensor(
        input=d, index=index, columns=columns, gpu=gpu, sparse=sparse, axis=axis
    )
    return op(d, index, columns, dtypes)


class SeriesFromTensor(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.SERIES_FROM_TENSOR

    input = KeyField("input")
    index = AnyField("index")

    @classmethod
    def _set_inputs(cls, op: "SeriesFromTensor", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if op.input is not None:
            op.input = op.inputs[0]
        if op.index is not None and hasattr(op.index, "key"):
            op.index = op.inputs[-1]

    def __call__(
        self,
        input_tensor: Tensor,
        index: Union[TileableType, pd.Index],
        dtype: np.dtype,
        name: Any,
    ):
        inputs = [input_tensor] if input_tensor is not None else []
        if index is not None:
            if not isinstance(index, pd.Index):
                if isinstance(index, INDEX_TYPE):
                    self.index = index
                    index_value = index.index_value
                    inputs.append(index)
                elif isinstance(index, ENTITY_TYPE):
                    self.index = index
                    index = astensor(index)
                    if index.ndim != 1:
                        raise ValueError(f"index should be 1-d, got {index.ndim}-d")
                    index_value = parse_index(
                        pd.Index([], dtype=index.dtype), index, type(self).__name__
                    )
                    inputs.append(index)
                else:
                    self.index = index = pd.Index(index)
                    index_value = parse_index(index)
            else:
                self.index = index
                index_value = parse_index(index)
        elif input_tensor is not None:
            if pd.isna(input_tensor.shape[0]):
                pd_index = pd.RangeIndex(-1)
            else:
                pd_index = pd.RangeIndex(start=0, stop=input_tensor.shape[0])
            index_value = parse_index(pd_index)
            self.index = pd_index
        else:
            self.index = index = pd.Index([], dtype=object)
            index_value = parse_index(index)

        if input_tensor is not None:
            shape = input_tensor.shape
        elif index is not None:
            shape = index.shape
        else:
            shape = (0,)

        return self.new_series(
            inputs, shape=shape, dtype=dtype, index_value=index_value, name=name
        )


def series_from_tensor(
    tensor: Tensor,
    index: Union[TileableType, pd.Index, list] = None,
    name: Any = None,
    dtype: np.dtype = None,
    gpu: bool = None,
    sparse: bool = False,
):
    if tensor is not None:
        if tensor.ndim > 1 or tensor.ndim <= 0:
            raise TypeError(f"Not support create Series from {tensor.ndim} dims tensor")
        gpu = tensor.op.gpu if gpu is None else gpu
        dtype = dtype or tensor.dtype
    else:
        gpu = None
        dtype = dtype or np.dtype(float)
    op = SeriesFromTensor(input=tensor, gpu=gpu, sparse=sparse)
    return op(tensor, index, dtype, name)
