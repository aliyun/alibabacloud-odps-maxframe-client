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

import dataclasses
import functools
import inspect
from typing import Any, Callable, Generic, List, Optional, TypeVar

import pandas as pd

from ..core import OutputType
from ..typing_ import PandasDType
from ..utils import make_dtype
from .utils import InferredDataFrameMeta, parse_index

# TypeVars
T = TypeVar("T")


@dataclasses.dataclass
class _FieldDef:
    name: Any
    dtype: PandasDType


def _item_to_field_def(item_):
    if isinstance(item_, tuple):
        tp = make_dtype(item_[1])
        return _FieldDef(name=item_[0], dtype=tp)
    elif isinstance(item_, slice):
        assert item_.step is None, "Should not specify step when specifying type hints"
        return _FieldDef(name=item_.start, dtype=item_.stop)
    else:
        tp = make_dtype(item_)
        return _FieldDef(name=None, dtype=tp)


class IndexType:
    def __init__(self, index_fields: List[_FieldDef]):
        self.index_fields = index_fields

    def __repr__(self):
        return f"IndexType({[f.dtype for f in self.index_fields]})"

    @classmethod
    def from_getitem_args(cls, item) -> "IndexType":
        if isinstance(item, (dict, pd.Series)):
            item = list(item.items())

        if isinstance(item, list) or (
            item and isinstance(item, tuple) and isinstance(item[0], slice)
        ):
            return IndexType([_item_to_field_def(tp) for tp in item])
        else:
            return IndexType([_item_to_field_def(item)])


class SeriesType(Generic[T]):
    def __init__(
        self, index_fields: Optional[List[_FieldDef]], name_and_dtype: _FieldDef
    ):
        self.index_fields = index_fields
        self.name_and_dtype = name_and_dtype

    def __repr__(self) -> str:
        return "SeriesType[{}]".format(self.name_and_dtype.dtype)

    @classmethod
    def from_getitem_args(cls, item) -> "SeriesType":
        if not isinstance(item, tuple):
            item = (item,)
        if len(item) == 1:
            tp = _item_to_field_def(item[0])
            return SeriesType(None, tp)
        else:
            tp = _item_to_field_def(item[1])
            idx_fields = IndexType.from_getitem_args(item[0]).index_fields
            return SeriesType(idx_fields, tp)


class DataFrameType:
    def __init__(
        self,
        index_fields: Optional[List[_FieldDef]],
        data_fields: List[_FieldDef],
    ):
        self.index_fields = index_fields
        self.data_fields = data_fields

    def __repr__(self) -> str:
        types = [field.dtype for field in self.data_fields]
        return f"DataFrameType[{types}]"

    @classmethod
    def from_getitem_args(cls, item) -> "DataFrameType":
        if not isinstance(item, tuple):
            item = (item,)
        if isinstance(item[0], slice):
            value_defs = item
            idx_defs = None
        else:
            value_defs = item[-1]
            idx_defs = item[0] if len(item) > 1 else None
        fields = IndexType.from_getitem_args(value_defs).index_fields
        if idx_defs is None:
            return DataFrameType(None, fields)
        else:
            idx_fields = IndexType.from_getitem_args(item[0]).index_fields
            return DataFrameType(idx_fields, fields)


def get_function_output_meta(
    func: Callable, df_obj=None
) -> Optional[InferredDataFrameMeta]:
    try:
        func_argspec = inspect.getfullargspec(func)
        ret_type = (func_argspec.annotations or {}).get("return")
        if ret_type is None:
            return None
    except:
        return None

    dtypes = dtype = name = None
    index_fields = None
    if isinstance(ret_type, DataFrameType):
        output_type = OutputType.dataframe
        dtypes = pd.Series(
            [fd.dtype for fd in ret_type.data_fields],
            index=[fd.name for fd in ret_type.data_fields],
        )
        index_fields = ret_type.index_fields
    elif isinstance(ret_type, SeriesType):
        output_type = OutputType.series
        dtype = ret_type.name_and_dtype.dtype
        name = ret_type.name_and_dtype.name
        index_fields = ret_type.index_fields
    elif isinstance(ret_type, IndexType):
        output_type = OutputType.index
        index_fields = ret_type.index_fields
    else:
        output_type = OutputType.scalar
        try:
            dtype = make_dtype(ret_type)
        except:
            return None

    if index_fields is not None:
        if len(index_fields) == 1:
            mock_idx = pd.Index(
                [], dtype=index_fields[0].dtype, name=index_fields[0].name
            )
        else:
            col_names = [index_field.name for index_field in index_fields]
            col_dtypes = pd.Series(
                [index_field.dtype for index_field in index_fields], index=col_names
            )
            mock_df = pd.DataFrame([], columns=col_names).astype(col_dtypes)
            mock_idx = pd.MultiIndex.from_frame(mock_df)
        index_value = parse_index(mock_idx, df_obj, store_data=False)
    else:
        index_value = None

    return InferredDataFrameMeta(
        output_type=output_type,
        index_value=index_value,
        dtypes=dtypes,
        dtype=dtype,
        name=name,
    )


def register_pandas_typing_funcs():
    def _cls_getitem_func(cls, item, type_cls):
        return type_cls.from_getitem_args(item)

    for pd_cls, type_cls in [
        (pd.DataFrame, DataFrameType),
        (pd.Series, SeriesType),
        (pd.Index, IndexType),
    ]:
        if hasattr(pd_cls, "__class_getitem__"):  # pragma: no cover
            continue
        pd_cls.__class_getitem__ = classmethod(
            functools.partial(_cls_getitem_func, type_cls=type_cls)
        )
