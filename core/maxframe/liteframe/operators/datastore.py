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

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.liteframe.operators.core import (
    LiteFrameDataFormat,
    LiteFrameOperator,
    LiteFrameOperatorMixin,
)
from maxframe.serialization.serializables import (
    AnyField,
    BoolField,
    DictField,
    EnumField,
    FieldTypes,
    Int64Field,
    ListField,
    SeriesField,
    StringField,
)
from maxframe.typing_ import TileableType


class LiteFrameDataStore(LiteFrameOperator, LiteFrameOperatorMixin):
    """Base class for LiteFrame datastore operators"""

    pass


class LiteFrameLakeDataStore(LiteFrameDataStore):
    """Lake-format datastore operator for Parquet, CSV, and JSON writes.

    A single operator with a ``data_format`` discriminator that selects
    the appropriate handler at execution and codegen time. Format-specific
    I/O parameters are packed into ``write_kwargs``.
    """

    _op_type_ = opcodes.LITEFRAME_LAKE_DATA_STORE

    path = AnyField("path")
    compression = AnyField("compression", default=None)
    partition_cols = ListField("partition_cols", default=None)
    storage_options = DictField("storage_options", default=None)
    write_stage = StringField("write_stage", default=None)
    data_format = EnumField(
        "data_format",
        LiteFrameDataFormat,
        FieldTypes.int8,
        default=None,
        primitive=True,
    )
    write_kwargs = DictField("write_kwargs", default=None)

    @property
    def one_file(self):
        if not isinstance(self.path, str):
            return True
        return "*" not in self.path and not self.partition_cols

    def __call__(self, lf):
        return self.new_liteframe(
            [lf],
            shape=(0, 0),
            physical_dtypes=lf._data._physical_dtypes[:0],
        )


class LiteFrameToLance(LiteFrameLakeDataStore):
    _op_type_ = opcodes.TO_LANCE

    mode = StringField("mode", default=None)
    left_on = StringField("left_on", default=None)
    right_on = StringField("right_on", default=None)
    schema_bytes = AnyField("schema_bytes", default=None)
    lance_kwargs = DictField("lance_kwargs", default=None)

    @property
    def one_file(self):
        return False

    def __call__(self, lf):
        return self.new_liteframe(
            [lf],
            shape=(0, 0),
            physical_dtypes=lf.dtypes[:0],
        )


class LiteFrameToODPSTable(LiteFrameDataStore):
    _op_type_ = opcodes.TO_ODPS_TABLE

    dtypes = SeriesField("dtypes")
    table_name = StringField("table_name", default=None)
    partition_spec = StringField("partition_spec", default=None)
    partition_columns = ListField("partition_columns", FieldTypes.string, default=None)
    mode = StringField("mode", default=None)
    unknown_as_string = BoolField("unknown_as_string", default=None)
    lifecycle = Int64Field("lifecycle", default=None)
    table_properties = DictField("table_properties", default=None)
    primary_key = ListField("primary_key", FieldTypes.string, default=None)

    def __init__(self, **kw):
        super().__init__(_output_types=[OutputType.liteframe], **kw)

    def can_fuse_with_custom_code(self) -> bool:
        return False

    def __call__(self, lf: TileableType):
        shape = (0,) * len(lf.shape)
        return self.new_liteframe(
            [lf],
            shape=shape,
            physical_dtypes=lf._data._physical_dtypes[:0],
        )
