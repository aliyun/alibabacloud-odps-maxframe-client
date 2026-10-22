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

from maxframe.liteframe.operators.core import LiteFrameDataFormat
from maxframe.liteframe.operators.datastore import LiteFrameLakeDataStore


def to_json(
    lf,
    path,
    *,
    lines=True,
    orient="records",
    compression=None,
    partition_cols=None,
    storage_options=None,
    **kwargs,
):
    """
    Write a LiteFrame to JSON format (line-delimited by default).

    Parameters
    ----------
    path : str
        File path. If path contains wildcard '*', each chunk writes to its
        own file. If no wildcard, all data writes to a single file.
    lines : bool, default True
        Write line-delimited JSON (JSONL/NDJSON). Each row is a separate
        JSON object on its own line.
    orient : str, default 'records'
        JSON orientation. Only 'records' is supported with lines=True.
    compression : str or None, default None
        Compression to use for output file.
    partition_cols : list, optional
        Column names by which to partition the dataset.
    storage_options : dict, optional
        Extra options for storage connection.

    Returns
    -------
    LiteFrame
        Returns a stub LiteFrame for execution.

    See Also
    --------
    read_json : Read a JSON file into a LiteFrame.
    to_csv : Write a LiteFrame to a CSV file.
    """
    _VALID_ORIENTS = {"split", "records", "index", "columns", "values", "table"}
    if orient not in _VALID_ORIENTS:
        raise ValueError(
            f"Invalid orient '{orient}', must be one of {sorted(_VALID_ORIENTS)}"
        )
    if lines and orient != "records":
        raise ValueError("orient must be 'records' when lines=True")

    if "*" not in path and not partition_cols and not lines:
        raise NotImplementedError(
            "Single-file JSON write is only supported with lines=True "
            "(NDJSON); use lines=True, a wildcard path ('*'), or "
            "partition_cols for multi-file output."
        )

    op = LiteFrameLakeDataStore(
        path=path,
        data_format=LiteFrameDataFormat.JSON,
        compression=compression,
        partition_cols=partition_cols,
        storage_options=storage_options,
        write_kwargs={
            "lines": lines,
            "orient": orient,
            **kwargs,
        }
        or None,
    )
    return op(lf)
