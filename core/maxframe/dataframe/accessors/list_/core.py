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

from .... import opcodes
from ....core import OutputType
from ....serialization.serializables import DictField, StringField, TupleField
from ....utils import no_default
from ...operators import DataFrameOperator, DataFrameOperatorMixin
from ..compat import LegacySeriesMethodOperator


class SeriesListMethod(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.SERIES_LIST_METHOD

    method = StringField("method", default=None)
    method_args = TupleField("method_args", default_factory=list)
    method_kwargs = DictField("method_kwargs", default_factory=dict)

    def __init__(self, output_types=None, **kw):
        output_types = output_types or [OutputType.series]
        kw["_output_types"] = kw.get("_output_types") or output_types
        super().__init__(**kw)

    def __call__(self, inp, dtype=None, name=no_default):
        dtype = dtype or inp.dtype
        name = inp.name if name is no_default else name
        return self.new_series(
            [inp],
            shape=inp.shape,
            dtype=dtype,
            index_value=inp.index_value,
            name=name,
        )


class LegacySeriesListOperator(LegacySeriesMethodOperator):
    _method_cls = SeriesListMethod
