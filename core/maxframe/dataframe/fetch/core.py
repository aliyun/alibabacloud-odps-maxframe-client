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

from typing import MutableMapping, Union

from ...core import OutputType, register_fetch_class
from ...core.operator import Fetch, FetchMixin, FetchShuffle, Operator
from ...serialization.serializables import FieldTypes, TupleField
from ...utils import on_deserialize_shape, on_serialize_shape
from ..operators import DataFrameOperatorMixin


class DataFrameFetchMixin(DataFrameOperatorMixin, FetchMixin):
    __slots__ = ()


class DataFrameFetch(Fetch, DataFrameFetchMixin):
    # required fields
    _shape = TupleField(
        "shape",
        FieldTypes.tuple,
        on_serialize=on_serialize_shape,
        on_deserialize=on_deserialize_shape,
    )

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def _extract_dataframe_or_series_kws(self, kws, **kw):
        if kws is None:
            kws = [kw]
        new_kws = []
        new_output_types = []
        for output_type, kwargs in zip(self._output_types, kws):
            if output_type == OutputType.df_or_series:
                data_params = kwargs["data_params"]
                data_type = kwargs["data_type"]
                if data_type == "series":
                    new_output_types.append(OutputType.series)
                else:
                    new_output_types.append(OutputType.dataframe)
                new_kws.append(data_params)
            else:
                new_output_types.append(output_type)
                new_kws.append(kwargs)
        self._output_types = new_output_types
        return new_kws

    def _new_tileables(self, inputs, kws=None, **kw):
        if "_key" in kw and self.source_key is None:
            self.source_key = kw["_key"]
        new_kws = self._extract_dataframe_or_series_kws(kws, **kw)
        return super()._new_tileables(inputs, kws=new_kws, **kw)

    @classmethod
    def estimate_size(
        cls, ctx: MutableMapping[str, Union[int, float]], op: "Operator"
    ) -> None:
        # use infinity to show that the size cannot be inferred
        # todo when local catalyst is implemented, and it should get the estimated size
        # from the source.
        ctx[op.outputs[0].key] = float("inf")


class DataFrameFetchShuffle(FetchShuffle, DataFrameFetchMixin):
    # required fields
    _shape = TupleField(
        "shape",
        FieldTypes.int64,
        on_serialize=on_serialize_shape,
        on_deserialize=on_deserialize_shape,
    )

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)


register_fetch_class(OutputType.dataframe, DataFrameFetch, DataFrameFetchShuffle)
register_fetch_class(
    OutputType.dataframe_groupby, DataFrameFetch, DataFrameFetchShuffle
)
register_fetch_class(OutputType.df_or_series, DataFrameFetch, DataFrameFetchShuffle)
register_fetch_class(OutputType.series, DataFrameFetch, DataFrameFetchShuffle)
register_fetch_class(OutputType.series_groupby, DataFrameFetch, DataFrameFetchShuffle)
register_fetch_class(OutputType.index, DataFrameFetch, DataFrameFetchShuffle)
register_fetch_class(OutputType.categorical, DataFrameFetch, DataFrameFetchShuffle)
