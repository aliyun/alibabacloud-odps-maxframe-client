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


import pandas as pd

from maxframe.liteframe.operators.datastore import LiteFrameToLance


def to_lance(
    lf,
    path,
    *,
    mode="overwrite",
    on=None,
    left_on=None,
    right_on=None,
    schema=None,
    storage_options=None,
    **kwargs,
):
    """Write a LiteFrame to a Lance dataset.

    Parameters
    ----------
    path : str
        Target path for the Lance dataset. Supports local, OSS, and S3 URIs.
    mode : {'overwrite', 'append', 'merge'}, default 'overwrite'
        Write mode. 'merge' updates existing rows and/or adds new columns
        in the target dataset by aligning on a join key.
    on : str, optional
        Column name to align both sides on. Mutually exclusive with
        left_on/right_on. Defaults to '_rowaddr' when mode='merge'.
    left_on : str, optional
        Column name in the existing dataset to align on. Mutually exclusive
        with on.
    right_on : str, optional
        Column name in the new data to align on. Mutually exclusive with
        on.
    schema : pyarrow.Schema, optional
        PyArrow schema to override auto-detected column types for columns
        being written from the input. For ``mode='overwrite'``/``'append'``
        this overrides types of input columns (columns not specified keep
        their auto-detected types). For ``mode='merge'`` the existing
        dataset schema is always respected first: fields naming existing
        dataset columns are ignored and those columns keep their stored
        types, so ``schema=`` only sets types for newly added columns.
    storage_options : dict, optional
        Storage connection options.
    **kwargs
        Additional keyword arguments passed to lance.

    Returns
    -------
    LiteFrame
        A stub LiteFrame for execution.

    See Also
    --------
    read_lance : Read a Lance dataset into a LiteFrame.
    to_parquet : Write a LiteFrame to a Parquet file.
    """
    if mode not in ("overwrite", "append", "merge"):
        raise ValueError(
            f"mode must be 'overwrite', 'append', or 'merge', got '{mode}'"
        )

    # Resolve on -> left_on / right_on
    if on is not None:
        if left_on is not None or right_on is not None:
            raise ValueError("Cannot specify both 'on' and 'left_on'/'right_on'")
        left_on = on
        right_on = on

    # Merge-mode validation
    if mode == "merge":
        # Default alignment key
        if left_on is None and right_on is None:
            left_on = "_rowaddr"
            right_on = "_rowaddr"

        # The merge tiler derives _fragment_id from _rowaddr to align
        # incoming rows to their target fragments before joining on
        # left_on/right_on (Lance's per-fragment merge is fragment-local).
        # _rowaddr is therefore required for *every* merge, regardless of
        # the join key — not only when the key itself is _rowaddr.
        hidden = (
            lf.frame_metadata.hidden_columns
            if lf.frame_metadata and lf.frame_metadata.hidden_columns
            else []
        )
        if "_rowaddr" not in hidden:
            raise ValueError(
                "LiteFrame must have '_rowaddr' hidden column for merge. "
                "Use read_lance(..., with_row_address=True) to include it."
            )

    schema_bytes = None
    if schema is not None:
        schema_bytes = schema.serialize().to_pybytes()

    # Non-merge modes write the visible frame only; Lance reserves names
    # like _rowaddr/_rowid, so hidden columns must be dropped before the
    # executor materializes the Arrow table. Merge keeps _rowaddr (and any
    # other hidden columns) since it is the join key / fragment-aligner.
    if mode != "merge" and lf._hidden_columns:
        from maxframe.liteframe.indexing.select import drop as lf_drop

        lf = lf_drop(lf, list(lf._hidden_columns))

    # For overwrite/append, apply the schema= type override via lf.astype so
    # the cast runs distributed (per-chunk) in the projection before the
    # executor materializes the Arrow table.  Merge mode handles the cast in
    # the tiler (only new columns are cast; existing columns keep their
    # stored types).
    if schema is not None and mode != "merge":
        dtype_map = {}
        lf_cols = {c.lower() for c in lf.columns}
        for field in schema:
            if field.name.lower() in lf_cols:
                dtype_map[field.name] = pd.ArrowDtype(field.type)
        if dtype_map:
            lf = lf.astype(dtype_map)

    op = LiteFrameToLance(
        path=path,
        mode=mode,
        left_on=left_on,
        right_on=right_on,
        schema_bytes=schema_bytes,
        storage_options=storage_options,
        lance_kwargs=kwargs or None,
    )
    return op(lf)
