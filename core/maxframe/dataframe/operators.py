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

import numpy as np
import pandas as pd

from ..core import OutputType
from ..core.operator import Operator, ShuffleProxy, TileableOperatorMixin
from ..tensor.core import TENSOR_TYPE
from ..tensor.datasource import tensor as astensor
from .core import DATAFRAME_TYPE, SERIES_TYPE


class DataFrameOperatorMixin(TileableOperatorMixin):
    __slots__ = ()
    _op_module_ = "dataframe"

    def new_dataframes(
        self,
        inputs,
        shape=None,
        dtypes=None,
        index_value=None,
        columns_value=None,
        chunks=None,
        nsplits=None,
        output_limit=None,
        kws=None,
        **kw
    ):
        setattr(self, "_output_types", [OutputType.dataframe])
        return self.new_tileables(
            inputs,
            shape=shape,
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
            chunks=chunks,
            nsplits=nsplits,
            output_limit=output_limit,
            kws=kws,
            **kw
        )

    def new_dataframe(
        self,
        inputs,
        shape=None,
        dtypes=None,
        index_value=None,
        columns_value=None,
        **kw
    ):
        if getattr(self, "output_limit") != 1:
            raise TypeError("cannot new DataFrame with more than 1 outputs")

        return self.new_dataframes(
            inputs,
            shape=shape,
            dtypes=dtypes,
            index_value=index_value,
            columns_value=columns_value,
            **kw
        )[0]

    def new_seriess(
        self,
        inputs,
        shape=None,
        dtype=None,
        index_value=None,
        name=None,
        chunks=None,
        nsplits=None,
        output_limit=None,
        kws=None,
        **kw
    ):
        setattr(self, "_output_types", [OutputType.series])
        return self.new_tileables(
            inputs,
            shape=shape,
            dtype=dtype,
            index_value=index_value,
            name=name,
            chunks=chunks,
            nsplits=nsplits,
            output_limit=output_limit,
            kws=kws,
            **kw
        )

    def new_series(
        self, inputs, shape=None, dtype=None, index_value=None, name=None, **kw
    ):
        if getattr(self, "output_limit") != 1:
            raise TypeError("cannot new Series with more than 1 outputs")

        return self.new_seriess(
            inputs, shape=shape, dtype=dtype, index_value=index_value, name=name, **kw
        )[0]

    def new_df_or_series(self, inputs, **kw):
        setattr(self, "_output_types", [OutputType.df_or_series])
        return self.new_tileables(inputs, **kw)[0]

    def new_indexes(
        self,
        inputs,
        shape=None,
        dtype=None,
        index_value=None,
        name=None,
        chunks=None,
        nsplits=None,
        output_limit=None,
        kws=None,
        **kw
    ):
        setattr(self, "_output_types", [OutputType.index])
        return self.new_tileables(
            inputs,
            shape=shape,
            dtype=dtype,
            index_value=index_value,
            name=name,
            chunks=chunks,
            nsplits=nsplits,
            output_limit=output_limit,
            kws=kws,
            **kw
        )

    def new_index(
        self, inputs, shape=None, dtype=None, index_value=None, name=None, **kw
    ):
        if getattr(self, "output_limit") != 1:
            raise TypeError("cannot new Index with more than 1 outputs")

        return self.new_indexes(
            inputs, shape=shape, dtype=dtype, index_value=index_value, name=name, **kw
        )[0]

    def new_scalars(
        self, inputs, dtype=None, chunks=None, output_limit=None, kws=None, **kw
    ):
        setattr(self, "_output_types", [OutputType.scalar])
        return self.new_tileables(
            inputs,
            shape=(),
            dtype=dtype,
            chunks=chunks,
            nsplits=(),
            output_limit=output_limit,
            kws=kws,
            **kw
        )

    def new_scalar(self, inputs, dtype=None, **kw):
        if getattr(self, "output_limit") != 1:
            raise TypeError("cannot new tensor with more than 1 outputs")

        return self.new_scalars(inputs, dtype=dtype, **kw)[0]

    def new_categoricals(
        self,
        inputs,
        shape=None,
        dtype=None,
        categories_value=None,
        chunks=None,
        nsplits=None,
        output_limit=None,
        kws=None,
        **kw
    ):
        setattr(self, "_output_types", [OutputType.categorical])
        return self.new_tileables(
            inputs,
            shape=shape,
            dtype=dtype,
            categories_value=categories_value,
            chunks=chunks,
            nsplits=nsplits,
            output_limit=output_limit,
            kws=kws,
            **kw
        )

    def new_categorical(
        self, inputs, shape=None, dtype=None, categories_value=None, **kw
    ):
        if getattr(self, "output_limit") != 1:
            raise TypeError("cannot new Categorical with more than 1 outputs")

        return self.new_categoricals(
            inputs, shape=shape, dtype=dtype, categories_value=categories_value, **kw
        )[0]

    @staticmethod
    def _process_input(x):
        from .initializer import DataFrame, Series

        if isinstance(x, (DATAFRAME_TYPE, SERIES_TYPE)) or pd.api.types.is_scalar(x):
            return x
        elif isinstance(x, pd.Series):
            return Series(x)
        elif isinstance(x, pd.DataFrame):
            return DataFrame(x)
        elif isinstance(x, (list, tuple, np.ndarray, TENSOR_TYPE)):
            return astensor(x)
        raise NotImplementedError


DataFrameOperator = Operator


class DataFrameShuffleProxy(ShuffleProxy, DataFrameOperatorMixin):
    def __init__(self, sparse=None, output_types=None, **kwargs):
        super().__init__(sparse=sparse, _output_types=output_types, **kwargs)
