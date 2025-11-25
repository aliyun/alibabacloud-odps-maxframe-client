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

try:
    from pyarrow import PythonFile
    from pyarrow.fs import FileSystemHandler, PyFileSystem
except ImportError:  # pragma: no cover
    FileSystemHandler = object
    PythonFile = PyFileSystem = None

from .base import FileSystem


class MFFileSystemHandler(FileSystemHandler):
    """
    Implementation of FileSystemHandler for MaxFrame
    FileSystem object to provide support for PyFileSystem
    in pyarrow.
    """

    def __init__(self, mf_fs: FileSystem):
        super().__init__()
        self.fs = mf_fs

    def get_type_name(self):
        protocol = self.fs.protocol
        if isinstance(protocol, list):
            protocol = protocol[0]
        return f"fsspec+{protocol}"

    def normalize_path(self, path):
        return path

    def copy_file(self, src_path, dest_path):
        self.fs.cp(src_path, dest_path)

    def create_dir(self, path, recursive):
        return self.fs.mkdir(path, create_parents=recursive)

    def delete_dir(self, path):
        return self.fs.delete(path, recursive=True)

    def delete_dir_contents(self, path, missing_dir_ok=False):
        for item in self.fs.ls(path):
            self.fs.delete(item, recursive=True)

    def delete_root_dir_contents(self):
        return self.delete_dir_contents("/")

    def delete_file(self, path):
        return self.fs.delete(path, recursive=False)

    def get_file_info(self, paths):
        from pyarrow.fs import FileInfo, FileType

        file_info_list = []
        for path in paths:
            try:
                stat_result = self.fs.stat(path)
                file_type = (
                    FileType.File
                    if stat_result.get("type") == "file"
                    else FileType.Directory
                )
                file_info = FileInfo(
                    path,
                    type=file_type,
                    size=stat_result.get("size", 0),
                    mtime=stat_result.get("modified_time", 0),
                )
                file_info_list.append(file_info)
            except FileNotFoundError:
                file_info_list.append(FileInfo(path, FileType.NotFound))
        return file_info_list

    def get_file_info_selector(self, selector):
        if not self.fs.isdir(selector.base_dir):
            if self.fs.exists(selector.base_dir):
                raise NotADirectoryError(selector.base_dir)
            else:
                if selector.allow_not_found:
                    return []
                else:
                    raise FileNotFoundError(selector.base_dir)

        find_str = (
            f"{selector.base_dir}/*"
            if not selector.recursive
            else f"{selector.base_dir}/**/*"
        )
        selected_files = self.fs.glob(find_str, recursive=selector.recursive)
        infos = self.get_file_info(selected_files)
        new_infos = []
        for path, info in zip(selected_files, infos):
            _path = path.strip("/")
            base_dir = selector.base_dir.strip("/")
            # Need to exclude base directory from selected files if present
            # (fsspec filesystems, see GH-37555)
            if _path != base_dir:
                new_infos.append(info)

        return new_infos

    def move(self, src_path, dest_path):
        return self.fs.rename(src_path, dest_path)

    def open_input_file(self, path):
        return PythonFile(self.fs.open(path, "rb"), mode="r")

    def open_input_stream(self, path):
        return PythonFile(self.fs.open(path, "rb"), mode="r")

    def open_output_stream(self, path, metadata):
        return PythonFile(self.fs.open(path, "wb"), mode="w")

    def open_append_stream(self, path, metadata):
        return PythonFile(self.fs.open(path, "ab"), mode="w")


def to_arrow_file_system(mf_fs: FileSystem):
    from .arrow import ArrowBasedFileSystem

    if isinstance(mf_fs, ArrowBasedFileSystem):
        return mf_fs._arrow_fs
    return PyFileSystem(MFFileSystemHandler(mf_fs))
