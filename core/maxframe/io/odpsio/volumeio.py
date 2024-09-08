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

import inspect
from typing import Iterator, List, Optional, Union

from odps import ODPS


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
        with self._volume.open_reader(self._volume_dir + "/" + file_name) as reader:
            return reader.read()


class ODPSVolumeWriter:
    def __init__(
        self,
        odps_entry: ODPS,
        volume_name: str,
        volume_dir: str,
        schema_name: Optional[str] = None,
    ):
        self._odps_entry = odps_entry
        self._volume = odps_entry.get_volume(volume_name, schema=schema_name)
        self._volume_dir = volume_dir

    def write_file(self, file_name: str, data: Union[bytes, Iterator[bytes]]):
        with self._volume.open_writer(self._volume_dir + "/" + file_name) as writer:
            if not inspect.isgenerator(data):
                writer.write(data)
            else:
                for chunk in data:
                    writer.write(chunk)
