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

from collections.abc import Iterable

from ... import opcodes
from ...core import ENTITY_TYPE, OutputType
from ...serialization.serializables import AnyField
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import parse_index


class GroupByIndex(DataFrameOperatorMixin, DataFrameOperator):
    _op_type_ = opcodes.INDEX
    _op_module_ = "dataframe.groupby"

    selection = AnyField("selection", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    @property
    def groupby_params(self):
        params = self.inputs[0].op.groupby_params
        params["selection"] = self.selection
        return params

    def build_mock_groupby(self, **kwargs):
        groupby_op = self.inputs[0].op
        selection = kwargs.pop("selection", None) or self.selection
        return groupby_op.build_mock_groupby(**kwargs)[selection]

    def __call__(self, groupby):
        indexed = groupby.op.build_mock_groupby()[self.selection]

        if indexed.ndim == 1:
            self.output_types = [OutputType.series_groupby]
            params = dict(
                shape=(groupby.shape[0],),
                name=self.selection,
                dtype=groupby.dtypes[self.selection],
                index_value=groupby.index_value,
                key_dtypes=groupby.key_dtypes,
            )
        else:
            self.output_types = [OutputType.dataframe_groupby]

            if (
                isinstance(self.selection, Iterable)
                and not isinstance(self.selection, str)
                and not isinstance(self.selection, ENTITY_TYPE)
            ):
                item_list = list(self.selection)
            else:
                item_list = [self.selection]

            params = groupby.params.copy()
            params["dtypes"] = new_dtypes = groupby.dtypes[item_list]
            params["selection"] = self.selection
            params["shape"] = (groupby.shape[0], len(item_list))
            params["columns_value"] = parse_index(new_dtypes.index, store_data=True)

        return self.new_tileable([groupby], **params)


def df_groupby_getitem(df_groupby, item):
    try:
        hash(item)
        hashable = True
    except TypeError:
        hashable = False

    if hashable and item in df_groupby.dtypes:
        output_types = [OutputType.series_groupby]
    elif (
        isinstance(item, Iterable)
        and not isinstance(item, ENTITY_TYPE)
        and all(it in df_groupby.dtypes for it in item)
    ):
        output_types = [OutputType.dataframe_groupby]
    else:
        raise NameError(f"Cannot slice groupby with {item!r}")

    if df_groupby.selection:
        raise IndexError(f"Column(s) {df_groupby.selection!r} already selected")

    if (
        isinstance(item, tuple)
        and item not in df_groupby.dtypes
        and item not in df_groupby.index.names
    ):
        item = list(item)
    op = GroupByIndex(selection=item, output_types=output_types)
    return op(df_groupby)
