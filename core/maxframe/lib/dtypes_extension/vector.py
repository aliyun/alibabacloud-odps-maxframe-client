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

import json

import pyarrow as pa


class ArrowVectorType(pa.ExtensionType):
    """PyArrow extension type for ODPS Vector type.

    Stores vector data as pa.list_(float_type) but preserves element_type
    and dimension metadata needed for ODPS Vector schema round-tripping.
    """

    def __init__(self, element_type: pa.DataType, dimension: int):
        self._element_type = element_type
        self._dimension = dimension
        super().__init__(pa.list_(element_type), "maxframe.vector")

    @property
    def element_type(self) -> pa.DataType:
        return self._element_type

    @property
    def dimension(self) -> int:
        return self._dimension

    def __arrow_ext_serialize__(self):
        return json.dumps(
            {"element_type": str(self._element_type), "dimension": self._dimension}
        ).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        meta = json.loads(serialized)
        element_type = pa.type_for_alias(meta["element_type"])
        return cls(element_type, meta["dimension"])

    def __eq__(self, other):
        return (
            isinstance(other, ArrowVectorType)
            and self._element_type == other._element_type
            and self._dimension == other._dimension
        )

    def __hash__(self):
        return hash(("maxframe.vector", str(self._element_type), self._dimension))

    def __str__(self):
        return f"maxframe.vector({self._element_type},{self._dimension})"

    def __repr__(self):
        return f"ArrowVectorType(element_type={self._element_type}, dimension={self._dimension})"

    def to_pandas_dtype(self):
        from maxframe.lib.dtypes_extension.dtypes import ArrowDtype

        return ArrowDtype(self)


# Register the extension type with a default instance.
# Note: ArrowVectorType instances with different element_type/dimension
# are not singletons, so we register a placeholder and rely on
# pa.ExtensionType.__arrow_ext_deserialize__ for deserialization.
try:
    pa.register_extension_type(ArrowVectorType(pa.float32(), 32))
except pa.ArrowKeyError:
    # Already registered (e.g. in a reused interpreter)
    pass
