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

import os
import warnings
from typing import Any, Dict, List

import pandas as pd

from ... import opcodes
from ...core import ENTITY_TYPE, Entity, EntityData, OutputType
from ...core.operator import MapReduceOperator
from ...env import MAXFRAME_INSIDE_TASK
from ...serialization import PickleContainer
from ...serialization.serializables import AnyField, BoolField, DictField, Int32Field
from ...udf import BuiltinFunction
from ...utils import find_objects, lazy_import, no_default
from ..core import GROUPBY_TYPE, SERIES_TYPE
from ..initializer import Series as asseries
from ..operators import DataFrameOperator, DataFrameOperatorMixin
from ..utils import (
    build_df,
    build_series,
    call_groupby_with_params,
    make_column_list,
    parse_index,
)

cudf = lazy_import("cudf")


class DataFrameGroupByOp(MapReduceOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.GROUPBY
    _legacy_name = "DataFrameGroupByOperator"  # since v2.0.0

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

    def has_custom_code(self) -> bool:
        callable_bys = find_objects(self.by, types=PickleContainer, checker=callable)
        if not callable_bys:
            return False
        return any(not isinstance(fun, BuiltinFunction) for fun in callable_bys)

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

        new_kw = self.groupby_params.copy()
        new_kw.update({k: v for k, v in kwargs.items()})
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
        return call_groupby_with_params(mock_obj, new_kw)

    @classmethod
    def _set_inputs(cls, op: "DataFrameGroupByOp", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        inputs_iter = iter(op._inputs[1:])
        if len(inputs) > 1:
            by = []
            for k in op.by or ():
                if isinstance(k, ENTITY_TYPE):
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


class BaseGroupByWindowOp(DataFrameOperatorMixin, DataFrameOperator):
    _op_module_ = "dataframe.groupby"

    groupby_params = DictField("groupby_params", default=None)
    window_params = DictField("window_params", default=None)

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)

    def _calc_mock_result_df(self, mock_groupby):
        raise NotImplementedError

    def get_sort_cols_to_asc(self) -> Dict[Any, bool]:
        order_cols = self.window_params.get("order_cols") or []
        asc_list = self.window_params.get("ascending") or [True]
        if len(asc_list) < len(order_cols):
            asc_list = [asc_list[0]] * len(order_cols)
        return dict(zip(order_cols, asc_list))

    def _calc_out_dtypes(self, in_groupby):
        in_obj = in_groupby
        groupby_params = in_groupby.op.groupby_params
        while isinstance(in_obj, GROUPBY_TYPE):
            in_obj = in_obj.inputs[0]

        if in_groupby.ndim == 1:
            selection = None
        else:
            by_cols = (
                make_column_list(groupby_params.get("by"), in_groupby.dtypes) or []
            )
            selection = groupby_params.get("selection")
            if not selection:
                selection = [c for c in in_obj.dtypes.index if c not in by_cols]

        mock_groupby = in_groupby.op.build_mock_groupby(
            group_keys=False, selection=selection
        )

        result_df = self._calc_mock_result_df(mock_groupby)

        if isinstance(result_df, pd.DataFrame):
            self.output_types = [OutputType.dataframe]
            return result_df.dtypes
        else:
            self.output_types = [OutputType.series]
            return result_df.name, result_df.dtype

    def __call__(self, groupby):
        in_df = groupby
        while in_df.op.output_types[0] not in (OutputType.dataframe, OutputType.series):
            in_df = in_df.inputs[0]

        out_dtypes = self._calc_out_dtypes(groupby)

        kw = in_df.params.copy()
        if self.output_types[0] == OutputType.dataframe:
            kw.update(
                dict(
                    columns_value=parse_index(out_dtypes.index, store_data=True),
                    dtypes=out_dtypes,
                    shape=(groupby.shape[0], len(out_dtypes)),
                )
            )
        else:
            name, dtype = out_dtypes
            kw.update(dtype=dtype, name=name, shape=(groupby.shape[0],))
        return self.new_tileable([in_df], **kw)


def _make_named_agg_compat(name):  # pragma: no cover
    # to make imports compatible
    from ..reduction import NamedAgg

    if name == "NamedAgg":
        if MAXFRAME_INSIDE_TASK not in os.environ:
            warnings.warn(
                "Please import NamedAgg from maxframe.dataframe",
                DeprecationWarning,
            )
        return NamedAgg
    raise AttributeError(f"module {__name__} has no attribute {name}")


__getattr__ = _make_named_agg_compat
