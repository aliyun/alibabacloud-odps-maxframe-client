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

from collections import namedtuple
from typing import List

import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, Entity, EntityData, OutputType
from ...core.operator import MapReduceOperator
from ...serialization.serializables import AnyField, BoolField, Int32Field
from ...utils import lazy_import, no_default
from ..core import SERIES_TYPE
from ..initializer import Series as asseries
from ..operators import DataFrameOperatorMixin
from ..utils import build_df, build_series, parse_index

cudf = lazy_import("cudf")


NamedAgg = namedtuple("NamedAgg", ["column", "aggfunc"])


class DataFrameGroupByOp(MapReduceOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.GROUPBY
    _legacy_name = "DataFrameGroupByOperator"

    by = AnyField(
        "by",
        default=None,
        on_serialize=lambda x: x.data if isinstance(x, Entity) else x,
    )
    level = AnyField("level", default=None)
    as_index = BoolField("as_index", default=None)
    sort = BoolField("sort", default=None)
    group_keys = BoolField("group_keys", default=None)

    shuffle_size = Int32Field("shuffle_size", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if output_types:
            if output_types[0] in (
                OutputType.dataframe,
                OutputType.dataframe_groupby,
            ):
                output_types = [OutputType.dataframe_groupby]
            elif output_types[0] == OutputType.series:
                output_types = [OutputType.series_groupby]
            self.output_types = output_types

    @property
    def is_dataframe_obj(self):
        return self.output_types[0] in (
            OutputType.dataframe_groupby,
            OutputType.dataframe,
        )

    @property
    def groupby_params(self):
        return dict(
            by=self.by,
            level=self.level,
            as_index=self.as_index,
            sort=self.sort,
            group_keys=self.group_keys,
        )

    def build_mock_groupby(self, **kwargs):
        in_df = self.inputs[0]
        if self.is_dataframe_obj:
            mock_obj = build_df(
                in_df, size=[2, 2], fill_value=[1, 2], ensure_string=True
            )
        else:
            mock_obj = build_series(
                in_df,
                size=[2, 2],
                fill_value=[1, 2],
                name=in_df.name,
                ensure_string=True,
            )

        new_kw = self.groupby_params
        new_kw.update(kwargs)
        if isinstance(new_kw["by"], list):
            new_by = []
            for v in new_kw["by"]:
                if isinstance(v, ENTITY_TYPE):
                    build_fun = build_df if v.ndim == 2 else build_series
                    mock_by = pd.concat(
                        [
                            build_fun(v, size=2, fill_value=1, name=v.name),
                            build_fun(v, size=2, fill_value=2, name=v.name),
                        ]
                    )
                    new_by.append(mock_by)
                else:
                    new_by.append(v)
            new_kw["by"] = new_by
        return mock_obj.groupby(**new_kw)

    @classmethod
    def _set_inputs(cls, op: "DataFrameGroupByOp", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs[1:])
        if len(inputs) > 1:
            by = []
            for k in op.by:
                if isinstance(k, SERIES_TYPE):
                    by.append(next(inputs_iter))
                else:
                    by.append(k)
            op.by = by

    def __call__(self, df):
        params = df.params.copy()
        params["index_value"] = parse_index(None, df.key, df.index_value.key)
        if df.ndim == 2:
            if isinstance(self.by, list):
                index, types = [], []
                for k in self.by:
                    if isinstance(k, SERIES_TYPE):
                        index.append(k.name)
                        types.append(k.dtype)
                    elif k in df.dtypes:
                        index.append(k)
                        types.append(df.dtypes[k])
                    else:
                        raise KeyError(k)
                params["key_dtypes"] = pd.Series(types, index=index)

        inputs = [df]
        if isinstance(self.by, list):
            for k in self.by:
                if isinstance(k, SERIES_TYPE):
                    inputs.append(k)

        return self.new_tileable(inputs, **params)


DataFrameGroupByOperator = DataFrameGroupByOp


def groupby(df, by=None, level=None, as_index=True, sort=True, group_keys=True):
    """
    Group DataFrame using a mapper or by a Series of columns.

    A groupby operation involves some combination of splitting the
    object, applying a function, and combining the results. This can be
    used to group large amounts of data and compute operations on these
    groups.

    Parameters
    ----------
    by : mapping, function, label, or list of labels
        Used to determine the groups for the groupby.
        If ``by`` is a function, it's called on each value of the object's
        index. If a dict or Series is passed, the Series or dict VALUES
        will be used to determine the groups (the Series' values are first
        aligned; see ``.align()`` method). If an ndarray is passed, the
        values are used as-is to determine the groups. A label or list of
        labels may be passed to group by the columns in ``self``. Notice
        that a tuple is interpreted as a (single) key.
    as_index : bool, default True
        For aggregated output, return object with group labels as the
        index. Only relevant for DataFrame input. as_index=False is
        effectively "SQL-style" grouped output.
    sort : bool, default True
        Sort group keys. Get better performance by turning this off.
        Note this does not influence the order of observations within each
        group. Groupby preserves the order of rows within each group.
    group_keys : bool
        When calling apply, add group keys to index to identify pieces.

    Notes
    -----
    MaxFrame only supports groupby with axis=0.
    Default value of `group_keys` will be decided given the version of local
    pandas library, which is True since pandas 2.0.

    Returns
    -------
    DataFrameGroupBy
        Returns a groupby object that contains information about the groups.

    See Also
    --------
    resample : Convenience method for frequency conversion and resampling
        of time series.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame({'Animal': ['Falcon', 'Falcon',
    ...                               'Parrot', 'Parrot'],
    ...                    'Max Speed': [380., 370., 24., 26.]})
    >>> df.execute()
       Animal  Max Speed
    0  Falcon      380.0
    1  Falcon      370.0
    2  Parrot       24.0
    3  Parrot       26.0
    >>> df.groupby(['Animal']).mean().execute()
            Max Speed
    Animal
    Falcon      375.0
    Parrot       25.0
    """
    if not as_index and df.op.output_types[0] == OutputType.series:
        raise TypeError("as_index=False only valid with DataFrame")

    output_types = (
        [OutputType.dataframe_groupby] if df.ndim == 2 else [OutputType.series_groupby]
    )
    if isinstance(by, (SERIES_TYPE, pd.Series)):
        if isinstance(by, pd.Series):
            by = asseries(by)
        by = [by]
    elif df.ndim > 1 and by is not None and not isinstance(by, list):
        by = [by]
    op = DataFrameGroupByOp(
        by=by,
        level=level,
        as_index=as_index,
        sort=sort,
        group_keys=group_keys if group_keys is not no_default else None,
        output_types=output_types,
    )
    return op(df)
