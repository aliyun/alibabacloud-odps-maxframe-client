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
import tempfile

import pytest
from pyarrow.fs import FileSelector

from .. import LocalFileSystem
from ..fshandler import MFFileSystemHandler


def test_init_and_get_type_name():
    local_fs = LocalFileSystem.get_instance()
    handler = MFFileSystemHandler(local_fs)

    # Test get_type_name
    assert handler.get_type_name() == "fsspec+file"


def test_normalize_path():
    local_fs = LocalFileSystem.get_instance()
    handler = MFFileSystemHandler(local_fs)

    test_path = "/some/test/path"
    assert handler.normalize_path(test_path) == test_path


def test_create_and_delete_dir():
    local_fs = LocalFileSystem.get_instance()
    handler = MFFileSystemHandler(local_fs)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test create_dir with recursive=True
        test_dir = os.path.join(temp_dir, "level1", "level2")
        handler.create_dir(test_dir, recursive=True)
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)

        # Test delete_dir
        handler.delete_dir(test_dir)
        assert not os.path.exists(test_dir)


def test_copy_file():
    local_fs = LocalFileSystem.get_instance()
    handler = MFFileSystemHandler(local_fs)

    with tempfile.TemporaryDirectory() as temp_dir:
        src_file = os.path.join(temp_dir, "source.txt")
        dest_file = os.path.join(temp_dir, "dest.txt")

        # Create source file
        with open(src_file, "w") as f:
            f.write("test content")

        # Copy file
        handler.copy_file(src_file, dest_file)

        # Check if destination file exists and has same content
        assert os.path.exists(dest_file)
        with open(dest_file, "r") as f:
            assert f.read() == "test content"


def test_delete_file():
    local_fs = LocalFileSystem.get_instance()
    handler = MFFileSystemHandler(local_fs)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")

        # Create test file
        with open(test_file, "w") as f:
            f.write("test content")

        # Check file exists
        assert os.path.exists(test_file)

        # Delete file
        handler.delete_file(test_file)

        # Check file is deleted
        assert not os.path.exists(test_file)


def test_move():
    local_fs = LocalFileSystem.get_instance()
    handler = MFFileSystemHandler(local_fs)

    with tempfile.TemporaryDirectory() as temp_dir:
        src_file = os.path.join(temp_dir, "source.txt")
        dest_file = os.path.join(temp_dir, "dest.txt")

        # Create source file
        with open(src_file, "w") as f:
            f.write("test content")

        # Move file
        handler.move(src_file, dest_file)

        # Check source file doesn't exist and destination file exists
        assert not os.path.exists(src_file)
        assert os.path.exists(dest_file)

        with open(dest_file, "r") as f:
            assert f.read() == "test content"


def test_get_file_info():
    local_fs = LocalFileSystem.get_instance()
    handler = MFFileSystemHandler(local_fs)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test file
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        # Create test directory
        test_dir = os.path.join(temp_dir, "test_dir")
        os.mkdir(test_dir)

        # Get file info
        infos = handler.get_file_info([test_file, test_dir, "/nonexistent"])

        # Check results
        assert len(infos) == 3

        # Check file info
        file_info = infos[0]
        assert file_info.path == test_file
        assert file_info.type.name == "File"  # FileType.File
        assert file_info.size > 0

        # Check directory info
        dir_info = infos[1]
        assert dir_info.path == test_dir
        assert dir_info.type.name == "Directory"  # FileType.Directory

        # Check nonexistent file info
        nonexistent_info = infos[2]
        assert nonexistent_info.path == "/nonexistent"
        assert nonexistent_info.type.name == "NotFound"  # FileType.NotFound


def test_get_file_info_selector():
    local_fs = LocalFileSystem.get_instance()
    handler = MFFileSystemHandler(local_fs)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        dir1 = os.path.join(temp_dir, "dir1")
        os.mkdir(dir1)

        file1 = os.path.join(temp_dir, "file1.txt")
        with open(file1, "w") as f:
            f.write("content1")

        file2 = os.path.join(dir1, "file2.txt")
        with open(file2, "w") as f:
            f.write("content2")

        # Test non-recursive selector
        selector = FileSelector(temp_dir, recursive=False)
        infos = handler.get_file_info_selector(selector)

        # Should contain file1.txt and dir1, but not file2.txt
        paths = [info.path for info in infos]
        assert file1 in paths
        assert dir1 in paths
        assert file2 not in paths

        # Test recursive selector
        selector = FileSelector(temp_dir, recursive=True)
        infos = handler.get_file_info_selector(selector)

        # Should contain file1.txt, dir1, and file2.txt
        paths = [info.path for info in infos]
        assert file1 in paths
        assert dir1 in paths
        assert file2 in paths


def test_get_file_info_selector_errors():
    local_fs = LocalFileSystem.get_instance()
    handler = MFFileSystemHandler(local_fs)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with file as base_dir (should raise NotADirectoryError)
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("content")

        selector = FileSelector(test_file, recursive=False)
        with pytest.raises(NotADirectoryError):
            handler.get_file_info_selector(selector)

        # Test with nonexistent directory and allow_not_found=False
        selector = FileSelector("/nonexistent", recursive=False, allow_not_found=False)
        with pytest.raises(FileNotFoundError):
            handler.get_file_info_selector(selector)

        # Test with nonexistent directory and allow_not_found=True
        selector = FileSelector("/nonexistent", recursive=False, allow_not_found=True)
        infos = handler.get_file_info_selector(selector)
        assert infos == []


def test_open_methods():
    local_fs = LocalFileSystem.get_instance()
    handler = MFFileSystemHandler(local_fs)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")

        # Test open_output_stream
        output_stream = handler.open_output_stream(test_file, None)
        output_stream.write(b"test content")
        output_stream.close()

        # Test open_input_file
        input_file = handler.open_input_file(test_file)
        content = input_file.read()
        input_file.close()
        assert content == b"test content"

        # Test open_input_stream
        input_stream = handler.open_input_stream(test_file)
        content = input_stream.read()
        input_stream.close()
        assert content == b"test content"

        # Test open_append_stream
        append_stream = handler.open_append_stream(test_file, None)
        append_stream.write(b" appended")
        append_stream.close()

        # Check final content
        input_file = handler.open_input_file(test_file)
        content = input_file.read()
        input_file.close()
        assert content == b"test content appended"


def test_delete_dir_contents():
    local_fs = LocalFileSystem.get_instance()
    handler = MFFileSystemHandler(local_fs)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        dir1 = os.path.join(temp_dir, "dir1")
        os.mkdir(dir1)

        file1 = os.path.join(temp_dir, "file1.txt")
        with open(file1, "w") as f:
            f.write("content1")

        file2 = os.path.join(dir1, "file2.txt")
        with open(file2, "w") as f:
            f.write("content2")

        # Delete directory contents
        handler.delete_dir_contents(temp_dir)

        # Check that contents are deleted but directory still exists
        assert os.path.exists(temp_dir)
        assert not os.path.exists(file1)
        assert not os.path.exists(dir1)
