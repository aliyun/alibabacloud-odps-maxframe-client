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

import os

from maxframe.lib.filesystem import get_fs, has_magic
from maxframe.liteframe.datatypes import to_arrow_dtypes
from maxframe.liteframe.utils import normalize_dtypes_index


def iter_local_files(path):
    """Yield (file_path, partition_keys) for local file paths.

    partition_keys is always None (no partition support yet).
    Handles glob patterns and directory listing.
    """
    paths = path if isinstance(path, list) else [path]
    for p in paths:
        fs = get_fs(p)
        if has_magic(p):
            # Glob pattern — use fs.glob directly to avoid a redundant
            # get_fs call inside the module-level glob() helper.
            for f in fs.glob(p):
                yield f, None
        elif fs.isdir(p):
            for f in fs.ls(p):
                if not fs.isdir(f) and not os.path.basename(f).startswith(("_", ".")):
                    yield f, None
        else:
            yield p, None


def infer_dtypes_from_pandas(pdf):
    """Infer Arrow dtypes from a pandas DataFrame.

    Column names are lowercased via normalize_dtypes_index so that all
    internal representations use normalized (lowercase) names.
    The input DataFrame is not modified.
    """
    dtypes = pdf.dtypes
    dtypes = to_arrow_dtypes(dtypes)
    return normalize_dtypes_index(dtypes)
