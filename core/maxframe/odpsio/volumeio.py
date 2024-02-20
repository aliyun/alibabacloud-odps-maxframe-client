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

from typing import List, Optional

from odps import ODPS
from odps.models import ExternalVolume, PartedVolume
from odps.tunnel.volumetunnel import VolumeTunnel


class ODPSVolumeReader:
    def __init__(self, odps_entry: ODPS, volume_name: str, volume_dir: str):
        self._odps_entry = odps_entry
        self._volume = odps_entry.get_volume(volume_name)
        self._volume_dir = volume_dir

    def list_files(self) -> List[str]:
        def _get_file_name(vol_file):
            if hasattr(vol_file, "name"):
                return vol_file.name
            return vol_file.path.rsplit("/", 1)[-1]

        return [
            _get_file_name(f)
            for f in self._odps_entry.list_volume_files(
                f"/{self._volume.name}/{self._volume_dir}"
            )
        ]

    def read_file(self, file_name: str) -> bytes:
        if isinstance(self._volume, PartedVolume):
            vol_tunnel = VolumeTunnel(self._odps_entry)
            session = vol_tunnel.create_download_session(
                self._volume, self._volume_dir, file_name
            )
            with session.open() as reader:
                return reader.read()
        elif isinstance(self._volume, ExternalVolume):
            with self._volume.open_reader(self._volume_dir + "/" + file_name) as reader:
                return reader.read()


class ODPSVolumeWriter:
    def __init__(self, odps_entry: ODPS, volume_name: str, volume_dir: str):
        self._odps_entry = odps_entry
        self._volume = odps_entry.get_volume(volume_name)
        self._volume_dir = volume_dir
        self._session_cache = None

    def create_write_session(self) -> Optional[str]:
        if not isinstance(self._volume, PartedVolume):
            return None
        vol_tunnel = VolumeTunnel(self._odps_entry)
        session = self._session_cache = vol_tunnel.create_upload_session(
            self._volume, self._volume_dir
        )
        return session.id

    def _get_existing_upload_session(self, write_session_id: Optional[str]):
        if self._session_cache is not None and (
            write_session_id is None or write_session_id == self._session_cache.id
        ):
            return self._session_cache
        vol_tunnel = VolumeTunnel(self._odps_entry)
        return vol_tunnel.create_upload_session(
            self._volume, self._volume_dir, write_session_id
        )

    def write_file(
        self, file_name: str, data: bytes, write_session_id: Optional[str] = None
    ):
        if isinstance(self._volume, PartedVolume):
            session = self._get_existing_upload_session(write_session_id)
            with session.open(file_name) as writer:
                writer.write(data)
        elif isinstance(self._volume, ExternalVolume):
            with self._volume.open_writer(self._volume_dir + "/" + file_name) as writer:
                writer.write(data)

    def commit(self, files: List[str], write_session_id: Optional[str] = None):
        if not isinstance(self._volume, PartedVolume):
            return None
        session = self._get_existing_upload_session(write_session_id)
        session.commit(files)
