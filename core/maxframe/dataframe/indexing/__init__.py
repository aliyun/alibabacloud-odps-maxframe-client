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


def _install():
    from maxframe.dataframe.core import DATAFRAME_TYPE, INDEX_TYPE, SERIES_TYPE
    from maxframe.dataframe.indexing.add_prefix_suffix import (
        df_add_prefix,
        df_add_suffix,
        series_add_prefix,
        series_add_suffix,
    )
    from maxframe.dataframe.indexing.align import align
    from maxframe.dataframe.indexing.at import at
    from maxframe.dataframe.indexing.droplevel import (
        df_series_droplevel,
        index_droplevel,
    )
    from maxframe.dataframe.indexing.filter import filter_dataframe
    from maxframe.dataframe.indexing.get_level_values import get_level_values
    from maxframe.dataframe.indexing.getitem import dataframe_getitem, series_getitem
    from maxframe.dataframe.indexing.iat import iat
    from maxframe.dataframe.indexing.iloc import (
        head,
        iloc,
        index_getitem,
        index_setitem,
        tail,
    )
    from maxframe.dataframe.indexing.insert import df_insert, index_insert
    from maxframe.dataframe.indexing.loc import loc
    from maxframe.dataframe.indexing.reindex import reindex, reindex_like
    from maxframe.dataframe.indexing.rename import (
        df_rename,
        index_rename,
        index_set_names,
        series_rename,
    )
    from maxframe.dataframe.indexing.rename_axis import rename_axis
    from maxframe.dataframe.indexing.reorder_levels import (
        df_reorder_levels,
        series_reorder_levels,
    )
    from maxframe.dataframe.indexing.reset_index import (
        df_reset_index,
        series_reset_index,
    )
    from maxframe.dataframe.indexing.sample import sample
    from maxframe.dataframe.indexing.set_axis import df_set_axis, series_set_axis
    from maxframe.dataframe.indexing.set_index import set_index
    from maxframe.dataframe.indexing.setitem import dataframe_setitem
    from maxframe.dataframe.indexing.swaplevel import df_swaplevel, series_swaplevel
    from maxframe.dataframe.indexing.take import take
    from maxframe.dataframe.indexing.truncate import truncate
    from maxframe.dataframe.indexing.where import mask, where
    from maxframe.dataframe.indexing.xs import xs

    for cls in DATAFRAME_TYPE + SERIES_TYPE:
        setattr(cls, "at", property(fget=at))
        setattr(cls, "droplevel", df_series_droplevel)
        setattr(cls, "filter", filter_dataframe)
        setattr(cls, "head", head)
        setattr(cls, "iat", property(fget=iat))
        setattr(cls, "iloc", property(fget=iloc))
        setattr(cls, "loc", property(fget=loc))
        setattr(cls, "mask", mask)
        setattr(cls, "reindex", reindex)
        setattr(cls, "reindex_like", reindex_like)
        setattr(cls, "rename_axis", rename_axis)
        setattr(cls, "sample", sample)
        setattr(cls, "tail", tail)
        setattr(cls, "take", take)
        setattr(cls, "truncate", truncate)
        setattr(cls, "where", where)
        setattr(cls, "xs", xs)

    for cls in DATAFRAME_TYPE:
        setattr(cls, "add_prefix", df_add_prefix)
        setattr(cls, "add_suffix", df_add_suffix)
        setattr(cls, "align", align)
        setattr(cls, "__getitem__", dataframe_getitem)
        setattr(cls, "insert", df_insert)
        setattr(cls, "rename", df_rename)
        setattr(cls, "reorder_levels", df_reorder_levels)
        setattr(cls, "reset_index", df_reset_index)
        setattr(cls, "set_axis", df_set_axis)
        setattr(cls, "set_index", set_index)
        setattr(cls, "__setitem__", dataframe_setitem)
        setattr(cls, "swaplevel", df_swaplevel)

    for cls in SERIES_TYPE:
        setattr(cls, "add_prefix", series_add_prefix)
        setattr(cls, "add_suffix", series_add_suffix)
        setattr(cls, "align", align)
        setattr(cls, "__getitem__", series_getitem)
        setattr(cls, "rename", series_rename)
        setattr(cls, "reorder_levels", series_reorder_levels)
        setattr(cls, "reset_index", series_reset_index)
        setattr(cls, "set_axis", series_set_axis)
        setattr(cls, "swaplevel", series_swaplevel)

    for cls in INDEX_TYPE:
        setattr(cls, "droplevel", index_droplevel)
        setattr(cls, "get_level_values", get_level_values)
        setattr(cls, "__getitem__", index_getitem)
        setattr(cls, "insert", index_insert)
        setattr(cls, "rename", index_rename)
        setattr(cls, "__setitem__", index_setitem)
        setattr(cls, "set_names", index_set_names)


_install()
del _install
