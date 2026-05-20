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

import itertools
from typing import Any, List, Optional

from maxframe.dataframe.operators import DataFrameOperator, DataFrameOperatorMixin
from maxframe.serialization.serializables import (
    AnyField,
    DictField,
    ListField,
    StringField,
)


def get_index_mapping(
    index_label: Optional[List[str]],
    raw_index_levels: List[Any],
) -> List[str]:
    """
    Compute column names for DataFrame index levels when writing to storage.

    This is a shared utility for determining how to name index columns
    in formats like CSV, Parquet, Lance, and ODPS tables.

    Parameters
    ----------
    index_label : list of str or None
        User-provided labels for index columns. If None or shorter than
        the number of levels, defaults will be used.
    raw_index_levels : list
        The names of the index levels from the DataFrame (e.g., df.index.names).

    Returns
    -------
    list of str
        Column names for each index level.

    Examples
    --------
    >>> get_index_mapping(None, [None])  # Single unnamed index
    ['index']
    >>> get_index_mapping(None, ['user_id'])  # Single named index
    ['user_id']
    >>> get_index_mapping(['id'], [None])  # User-provided label
    ['id']
    >>> get_index_mapping(None, [None, None])  # Multi-level unnamed
    ['level_0', 'level_1']
    """
    def_labels = index_label or itertools.repeat(None)
    def_labels = itertools.chain(def_labels, itertools.repeat(None))
    names = raw_index_levels

    # Default labels depend on number of levels
    if len(names) == 1:
        default_labels = ["index"]
    else:
        default_labels = [f"level_{i}" for i in range(len(names))]

    # Priority: user-provided label > existing name > default label
    indexes = [
        def_label or name or label
        for def_label, name, label in zip(def_labels, names, default_labels)
    ]
    return indexes


class DataFrameDataStore(DataFrameOperator, DataFrameOperatorMixin):
    pass


class LakeDataStore(DataFrameDataStore):
    path = AnyField("path")
    compression = AnyField("compression", default=None)
    partition_cols = ListField("partition_cols", default=None)
    storage_options = DictField("storage_options", default=None)
    write_stage = StringField("write_stage", default=None)
