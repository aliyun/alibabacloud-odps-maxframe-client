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

import glob
import json
import os
from collections import OrderedDict
from typing import List, NamedTuple, Optional, Tuple, Union

import pandas as pd
from pandas.api.types import is_dict_like, is_list_like

from ...core import OutputType
from ...lib.filesystem import FileSystem, get_fs
from ...utils import make_dtypes, no_default
from ..core import IndexValue
from ..utils import get_index_value_by_default_index_type, parse_index


def find_partitions_and_sample_file(
    fs: FileSystem, current_path, input_parts=None
) -> Tuple[Optional[List[str]], Optional[str]]:
    _, dirs, files = next(fs.walk(current_path))
    if len(files) == 0:
        if len(dirs) == 0:
            return None, None  # return empty partition
        for d in dirs:
            dir_parts = (input_parts or []) + [d.split("=", 1)[0]]
            next_dir = os.path.join(current_path, d)
            next_parts, next_sample = find_partitions_and_sample_file(
                fs, next_dir, dir_parts
            )
            if next_sample is not None:
                return next_parts, next_sample
        # all parts are empty
        return None, None  # return empty partition
    elif len(dirs) == 0:
        return input_parts, os.path.join(current_path, files[0])
    else:  # pragma: no cover
        raise ValueError("Files and directories are mixed in an intermediate directory")


def iter_partition_keys_and_files(
    fs: FileSystem,
    root_path: str,
    partition_cols: Optional[List[str]] = None,
):
    if partition_cols is None and fs.isdir(root_path):
        partition_cols, _ = find_partitions_and_sample_file(fs, root_path)
    for part_root, dirs, files in fs.walk(root_path):
        if not files:
            continue
        relpath = os.path.relpath(part_root, root_path).strip("/")
        partition_keys = [tuple(s.split("=", 1)) for s in relpath.split("/")]
        assert len(partition_keys) == len(
            partition_cols
        ), "Partition keys isn't consistent through dataset"
        for file in files:
            yield os.path.join(part_root, file), partition_keys


def iter_local_files(path: Union[str, List[str]], resolve_partitions: bool = True):
    paths = path if isinstance(path, list) else [path]
    for path in paths:
        fs = get_fs(path)
        parts = None
        if resolve_partitions and fs.isdir(path):
            parts, _ = find_partitions_and_sample_file(fs, path)

        if parts is not None:
            yield from iter_partition_keys_and_files(fs, path, parts)
        else:
            if os.path.isdir(path):
                path = os.path.join(path, "*")

            if "*" in path or "?" in path:
                for p in glob.glob(path):
                    yield (p, None) if resolve_partitions else p
            else:
                yield (path, None) if resolve_partitions else path


def _resolve_dict_dtypes(dtypes_list) -> pd.Series:
    dtypes_dict = OrderedDict([(d["key"], d["value"]) for d in dtypes_list])
    names = list(dtypes_dict.keys())
    vals = [pd.api.types.pandas_dtype(dt) for dt in dtypes_dict.values()]
    return pd.Series(vals, index=names)


class LakeOutputInfoResult(NamedTuple):
    dtypes: pd.Series
    index_dtypes: pd.Series
    index_value: IndexValue
    is_partitioned: bool
    output_type: OutputType


def get_lake_output_info(
    reader_cls,
    path: Union[str, list],
    columns=None,
    index_col=None,
    dtype=None,
    names=None,
    index_dtypes=None,
    dtype_backend=no_default,
    use_nullable_dtypes=no_default,
    read_kwargs=None,
    default_index_type=None,
    storage_options=None,
    session=None,
    run_kwargs=None,
    **kwargs,
) -> LakeOutputInfoResult:
    """
    Extract dtypes, index_dtypes and index_value for lake data sources.

    Parameters
    ----------
    reader_cls : class
        The reader class (e.g., DataFrameReadCSV)
    path : str or list
        File path(s)
    columns : list, optional
        Column names to read
    index_col : int, str or list, optional
        Column(s) to use as index
    dtype : dict or type, optional
        Data type(s) for columns
    names : list, optional
        Column names
    dtype_backend : str, optional
        Backend for dtype handling
    use_nullable_dtypes : bool, optional
        Whether to use nullable dtypes
    read_kwargs : dict, optional
        Additional keyword arguments for reading
    default_index_type : str, optional
        Default index type
    storage_options : dict, optional
        Storage options
    session : session, optional
        Execution session
    run_kwargs : dict, optional
        Run keyword arguments
    **kwargs
        Additional keyword arguments

    Returns
    -------
    LakeOutputInfoResult
        A named tuple containing dtypes, index_dtypes, index_value, is_partitioned, and output_type
    """
    dtypes = None
    is_partitioned = False
    output_type = OutputType.dataframe

    # Handle dtype parameter if provided
    if dtype is not None:
        if is_dict_like(dtype):
            if not names:
                dtypes = make_dtypes(dtype)
            else:
                assert len(names) >= len(dtype), "Need to provide more names"
                dtypes = make_dtypes({name: dtype[name] for name in names})
        elif dtype is not None and names:
            dtypes = make_dtypes({name: dtype for name in names})

        # Handle index_col if provided
        if dtypes is not None and index_col is not None:
            if not is_list_like(index_col):
                index_col = [index_col]

            # Extract index dtypes from overall dtypes
            if index_dtypes is None:
                index_dtypes = dtypes[index_col]
            remaining_cols = [col for col in dtypes.index if col not in index_col]
            dtypes = dtypes[remaining_cols]

    # If dtypes still not determined, need to submit job to get them
    if dtypes is None:
        # Prepare operator for getting dtypes
        dt_op_kwargs = {
            "path": path,
            "dtype": dtype,
            "dtype_backend": dtype_backend,
            "use_nullable_dtypes": use_nullable_dtypes,
            "read_kwargs": read_kwargs or {},
            "default_index_type": default_index_type,
            "storage_options": storage_options,
            "read_stage": "get_dtypes",
        }

        # Add format-specific parameters
        if columns is not None:
            dt_op_kwargs["columns"] = (
                columns if hasattr(reader_cls, "columns") else columns
            )
            dt_op_kwargs["usecols"] = (
                columns if hasattr(reader_cls, "usecols") else columns
            )

        if hasattr(reader_cls, "index_col"):
            dt_op_kwargs["index_col"] = index_col

        if hasattr(reader_cls, "names"):
            dt_op_kwargs["names"] = names

        # Add other format-specific kwargs
        for key, value in kwargs.items():
            if hasattr(reader_cls, key):
                dt_op_kwargs[key] = value

        dt_op = reader_cls(**dt_op_kwargs)
        run_kwargs = run_kwargs or {}
        dt_result = json.loads(
            dt_op().execute(session=session, **run_kwargs).fetch(session=session),
        )
        is_partitioned = dt_result["is_partitioned"]
        if dt_result.get("index_dtypes"):
            index_dtypes = _resolve_dict_dtypes(dt_result["index_dtypes"])
        if dt_result.get("output_type"):
            output_type = getattr(OutputType, dt_result["output_type"])

        if output_type == OutputType.dataframe:
            dtypes = _resolve_dict_dtypes(dt_result["dtypes"])
        else:
            dtypes = pd.Series(
                [make_dtypes(dt_result["dtype"])], name=dt_result.get("name")
            )

    # Apply column filtering if needed
    if columns and hasattr(reader_cls, "columns"):
        dtypes = dtypes[columns]

    # Create index_value from index_dtypes
    if index_dtypes is None:
        index_value = get_index_value_by_default_index_type(
            default_index_type, args=(path, index_dtypes)
        )
    else:
        mock_index = pd.MultiIndex.from_frame(
            pd.DataFrame([], columns=index_dtypes.index).astype(index_dtypes)
        )
        if mock_index.nlevels == 1:
            mock_index = mock_index.get_level_values(0)
        index_value = parse_index(mock_index, store_data=False)

    return LakeOutputInfoResult(
        dtypes=dtypes,
        index_dtypes=index_dtypes,
        index_value=index_value,
        is_partitioned=is_partitioned,
        output_type=output_type,
    )
