# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

from abc import ABC, abstractmethod
from numbers import Integral
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import pyarrow as pa
from odps import ODPS
from odps.lib import tzlocal
from odps.models import ExternalVolume, PartedVolume
from odps.tunnel import TableTunnel
from tornado import httpclient

from maxframe.config import options
from maxframe.core import OBJECT_TYPE
from maxframe.dataframe.core import DATAFRAME_TYPE
from maxframe.lib import wrapped_pickle as pickle
from maxframe.odpsio import HaloTableIO, arrow_to_pandas, build_dataframe_table_meta
from maxframe.protocol import (
    DataFrameTableMeta,
    ODPSTableResultInfo,
    ODPSVolumeResultInfo,
    ResultInfo,
    ResultType,
)
from maxframe.tensor.core import TENSOR_TYPE
from maxframe.typing_ import PandasObjectTypes, TileableType
from maxframe.utils import ToThreadMixin

_result_fetchers: Dict[ResultType, Type["ResultFetcher"]] = dict()


def register_fetcher(fetcher_cls: Type["ResultFetcher"]):
    _result_fetchers[fetcher_cls.result_type] = fetcher_cls
    return fetcher_cls


def get_fetcher_cls(result_type: ResultType) -> Type["ResultFetcher"]:
    return _result_fetchers[result_type]


class ResultFetcher(ABC):
    result_type = None

    def __init__(self, odps_entry: ODPS):
        self._odps_entry = odps_entry

    @abstractmethod
    async def update_tileable_meta(
        self,
        tileable: TileableType,
        info: ResultInfo,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def fetch(
        self,
        tileable: TileableType,
        info: ResultInfo,
        indexes: List[Union[None, Integral, slice]],
    ) -> Any:
        raise NotImplementedError


@register_fetcher
class NullFetcher(ResultFetcher):
    result_type = ResultType.NULL

    async def update_tileable_meta(
        self,
        tileable: TileableType,
        info: ResultInfo,
    ) -> None:
        return

    async def fetch(
        self,
        tileable: TileableType,
        info: ODPSTableResultInfo,
        indexes: List[Union[None, Integral, slice]],
    ) -> None:
        return


@register_fetcher
class ODPSTableFetcher(ToThreadMixin, ResultFetcher):
    result_type = ResultType.ODPS_TABLE

    def _get_table_comment(self, table_name: str) -> Optional[str]:
        table = self._odps_entry.get_table(table_name)
        return getattr(table, "comment", None)

    async def update_tileable_meta(
        self,
        tileable: TileableType,
        info: ODPSTableResultInfo,
    ) -> None:
        if (
            isinstance(tileable, DATAFRAME_TYPE)
            and tileable.dtypes is None
            and info.table_meta is not None
        ):
            tileable.refresh_from_table_meta(info.table_meta)

        if tileable.shape and any(pd.isna(x) for x in tileable.shape):
            part_specs = [None] if not info.partition_specs else info.partition_specs
            tunnel = TableTunnel(self._odps_entry)
            total_records = 0
            for part_spec in part_specs:
                session = tunnel.create_download_session(
                    info.full_table_name, part_spec
                )
                total_records += session.count
            new_shape_list = list(tileable.shape)
            new_shape_list[0] = total_records
            tileable.params = {"shape": tuple(new_shape_list)}

    @staticmethod
    def _align_selection_with_shape(
        row_sel: slice, shape: Tuple[Optional[int], ...]
    ) -> dict:
        size = shape[0]
        if not row_sel.start and not row_sel.stop:
            return {}
        is_reversed = row_sel.step is not None and row_sel.step < 0
        read_kw = {
            "start": row_sel.start,
            "stop": row_sel.stop,
            "reverse_range": is_reversed,
        }
        if pd.isna(size):
            return read_kw

        if is_reversed and row_sel.start is not None:
            read_kw["start"] = min(size - 1, row_sel.start)
        if not is_reversed and row_sel.stop is not None:
            read_kw["stop"] = min(size, row_sel.stop)
        return read_kw

    def _read_single_source(
        self,
        table_meta: DataFrameTableMeta,
        info: ODPSTableResultInfo,
        indexes: List[Union[None, Integral, slice]],
        shape: Tuple[Optional[int], ...],
    ):
        table_io = HaloTableIO(self._odps_entry)
        read_kw = {}
        row_step = None
        if indexes:
            if len(indexes) < 2:
                indexes += [None]
            row_sel, col_sel = indexes
            if isinstance(row_sel, slice):
                row_step = row_sel.step
                read_kw = self._align_selection_with_shape(row_sel, shape)
            elif isinstance(row_sel, int):
                read_kw["start"] = row_sel
                read_kw["stop"] = row_sel + 1
                row_step = None
            elif row_sel is not None:  # pragma: no cover
                raise NotImplementedError(f"Does not support row index {row_sel!r}")

            if isinstance(col_sel, (int, slice)):
                data_cols = table_meta.table_column_names[col_sel]
                if isinstance(col_sel, int):
                    data_cols = [data_cols]
                read_kw["columns"] = table_meta.table_index_column_names + data_cols
            elif col_sel is not None:  # pragma: no cover
                raise NotImplementedError(f"Does not support column index {row_sel!r}")

        with table_io.open_reader(
            info.full_table_name, info.partition_specs, **read_kw
        ) as reader:
            reader_count = reader.count
            result = reader.read_all()

        if not row_step:
            return result

        if row_step >= 0:
            slice_start = 0
            slice_stop = reader_count
        else:
            slice_start = reader_count - 1
            slice_stop = None
        return result[slice_start:slice_stop:row_step]

    async def fetch(
        self,
        tileable: TileableType,
        info: ODPSTableResultInfo,
        indexes: List[Union[None, Integral, slice]],
    ) -> PandasObjectTypes:
        table_meta = build_dataframe_table_meta(tileable)
        arrow_table: pa.Table = await self.to_thread(
            self._read_single_source, table_meta, info, indexes, tileable.shape
        )
        # deal datetime timezone convert
        res = arrow_to_pandas(arrow_table, table_meta)
        timezone = options.local_timezone or tzlocal.get_localzone()
        for index, col_type in table_meta.pd_column_dtypes.items():
            if pd.api.types.is_datetime64_any_dtype(
                col_type
            ) and not pd.api.types.is_datetime64_ns_dtype(col_type):
                if index is None:
                    res = res.dt.tz_convert(timezone)
                else:
                    res[index] = res[index].dt.tz_convert(timezone)
        return res


@register_fetcher
class ODPSVolumeFetcher(ToThreadMixin, ResultFetcher):
    result_type = ResultType.ODPS_VOLUME

    async def update_tileable_meta(
        self,
        tileable: TileableType,
        info: ODPSVolumeResultInfo,
    ) -> None:
        return

    async def _read_parted_volume_data(
        self, volume: PartedVolume, partition: str, file_name: str
    ) -> bytes:
        def sync_read():
            with volume.open_reader(partition, file_name) as reader:
                return reader.read()

        return await self.to_thread(sync_read)

    async def _read_external_volume_data(
        self, volume: ExternalVolume, path: str, file_name: str
    ) -> bytes:
        signed_url = await self.to_thread(
            volume.get_sign_url, path + "/" + file_name, "GET"
        )
        http_client = httpclient.AsyncHTTPClient()

        resp = await http_client.fetch(signed_url)
        if hasattr(resp, "status_code") and resp.code >= 400:
            try:
                import oss2.exceptions

                oss_exc = oss2.exceptions.make_exception(resp.body)
                raise oss_exc
            except ImportError:
                raise SystemError(resp.body)
        return resp.body

    async def _fetch_object(self, info: ODPSVolumeResultInfo) -> Any:
        volume = await self.to_thread(self._odps_entry.get_volume, info.volume_name)
        if isinstance(volume, PartedVolume):
            byte_data = await self._read_parted_volume_data(
                volume, info.volume_path, "data"
            )
        elif isinstance(volume, ExternalVolume):
            byte_data = await self._read_external_volume_data(
                volume, info.volume_path, "data"
            )
        else:
            raise NotImplementedError(f"Volume type {type(volume)} not supported")
        return pickle.loads(byte_data)

    async def fetch(
        self,
        tileable: TileableType,
        info: ODPSVolumeResultInfo,
        indexes: List[Union[Integral, slice]],
    ) -> Any:
        if isinstance(tileable, (OBJECT_TYPE, TENSOR_TYPE)):
            return await self._fetch_object(info)
        raise NotImplementedError(f"Fetching {type(tileable)} not implemented")
