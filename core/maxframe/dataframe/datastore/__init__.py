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

from .direct import df_to_dict, series_to_dict, series_to_list, to_clipboard
from .to_csv import to_csv
from .to_odps import to_odps_table


def _install():
    from ..core import DATAFRAME_TYPE, SERIES_TYPE

    for t in DATAFRAME_TYPE:
        t.to_clipboard = to_clipboard
        t.to_csv = to_csv
        t.to_dict = df_to_dict
        t.to_odps_table = to_odps_table
    for t in SERIES_TYPE:
        t.to_clipboard = to_clipboard
        t.to_csv = to_csv
        t.to_dict = series_to_dict
        t.to_list = series_to_list


_install()
del _install
