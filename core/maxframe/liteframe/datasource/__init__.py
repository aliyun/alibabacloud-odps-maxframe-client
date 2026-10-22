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

from maxframe.liteframe.datasource.from_local import from_local_df
from maxframe.liteframe.datasource.from_tensor import liteframe_from_tensor
from maxframe.liteframe.datasource.read_csv import read_csv
from maxframe.liteframe.datasource.read_json import read_json
from maxframe.liteframe.datasource.read_lance import read_lance
from maxframe.liteframe.datasource.read_odps_table import read_odps_table
from maxframe.liteframe.datasource.read_parquet import read_parquet


def _install():
    from maxframe.liteframe.core import LITEFRAME_TYPE

    for cls in LITEFRAME_TYPE:
        setattr(cls, "from_local_df", staticmethod(from_local_df))
        setattr(cls, "from_tensor", staticmethod(liteframe_from_tensor))


_install()
del _install
