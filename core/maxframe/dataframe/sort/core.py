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

from ...serialization.serializables import (
    AnyField,
    BoolField,
    FieldTypes,
    Int32Field,
    Int64Field,
    ListField,
    StringField,
)
from ..operators import DataFrameOperator


class DataFrameSortOperator(DataFrameOperator):
    axis = Int32Field("axis")
    ascending = AnyField("ascending")
    inplace = BoolField("inplace")
    kind = StringField("kind")
    na_position = StringField("na_position")
    ignore_index = BoolField("ignore_index")
    parallel_kind = StringField("parallel_kind")
    psrs_kinds = ListField("psrs_kinds", FieldTypes.string, default=None)
    nrows = Int64Field("nrows", default=None)
    keep_kind = StringField("keep_kind", default="head")
