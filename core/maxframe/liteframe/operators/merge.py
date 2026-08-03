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

import numpy as np
import pandas as pd

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.core.operator import MapReduceOperator, OperatorStage
from maxframe.liteframe.core import FrameMetadata
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import (
    AnyField,
    DictField,
    Int32Field,
    ReferenceField,
    StringField,
    TupleField,
)


def _build_empty_df(dtypes):
    """Build empty pandas DataFrame from dtypes Series."""
    return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes.items()})


class LiteFrameMerge(LiteFrameOperatorMixin, LiteFrameOperator):
    """
    Merge two LiteFrame objects with a database-style join operation.

    This operator performs merge operations similar to pandas.DataFrame.merge(),
    supporting various join types and merge strategies. It handles the computation
    graph construction for distributed merge operations in MaxFrame.

    Parameters
    ----------
    how : str, default 'inner'
        Type of merge to perform:
        - 'left': use only keys from left frame (SQL: LEFT OUTER JOIN)
        - 'right': use only keys from right frame (SQL: RIGHT OUTER JOIN)
        - 'outer': use union of keys from both frames (SQL: FULL OUTER JOIN)
        - 'inner': use intersection of keys from both frames (SQL: INNER JOIN)
    on : str or list, optional
        Column or index level names to join on. Must be found in both the left
        and right LiteFrame objects. If None and left_on/right_on are also None,
        defaults to the intersection of the columns in both DataFrames.
    left_on : str or list, optional
        Column or index level names to join on in the left LiteFrame.
    right_on : str or list, optional
        Column or index level names to join on in the right LiteFrame.
    suffixes : tuple, default ('', '_y')
        Suffixes to apply to overlapping column names in the left and right
        LiteFrame objects respectively.
    method : str, default 'auto'
        Merge method to use:
        - 'auto': automatically choose merge method
        - 'shuffle': use shuffle-based merge for large datasets
        - 'broadcast': use broadcast-based merge for small datasets
    auto_merge : str, default 'both'
        Strategy for automatic merge optimization.
    auto_merge_threshold : int, default 8
        Threshold for automatic merge method selection.
    bloom_filter : str or bool, default 'auto'
        Whether to use bloom filter for merge optimization:
        - 'auto': automatically determine based on data characteristics
        - True: always use bloom filter
        - False: never use bloom filter
    bloom_filter_options : dict, optional
        Additional options for bloom filter configuration.
    split_info : Any, optional
        Information about data splitting for distributed execution.

    Returns
    -------
    LiteFrame
        Merged LiteFrame object with combined data from both inputs.

    Examples
    --------
    >>> import maxframe.liteframe as ml
    >>> # Inner join on common columns
    >>> left = ml.DataFrame({'key': [1, 2, 3], 'A': ['a', 'b', 'c']})
    >>> right = ml.DataFrame({'key': [2, 3, 4], 'B': ['d', 'e', 'f']})
    >>> merged = left.merge(right, on='key')
    >>> # Result contains rows with keys [2, 3] present in both frames

    >>> # Left outer join
    >>> merged = left.merge(right, on='key', how='left')
    >>> # Result contains all rows from left frame

    Notes
    -----
    - The merge operation is optimized for distributed execution in MaxFrame
    - Column names are handled according to pandas merge semantics
    - For large datasets, shuffle-based merge is automatically selected
    """

    _op_type_ = opcodes.LITEFRAME_MERGE

    how = StringField("how", default="inner")
    on = AnyField("on", default=None)
    left_on = AnyField("left_on", default=None)
    right_on = AnyField("right_on", default=None)
    suffixes = TupleField("suffixes", default=("", "_y"))
    method = StringField("method", default="auto")

    auto_merge = StringField("auto_merge", default="both")
    auto_merge_threshold = Int32Field("auto_merge_threshold", default=8)

    bloom_filter = AnyField("bloom_filter", default="auto")
    bloom_filter_options = DictField("bloom_filter_options", default=None)

    split_info = AnyField("split_info", default=None)

    def __init__(self, **kw):
        super().__init__(**kw)
        if self.suffixes is None:
            self.suffixes = ("", "_y")

    def __call__(self, left, right):
        on = self.on
        left_on = self.left_on
        right_on = self.right_on

        # Use _physical_dtypes for the mock merge — these are the columns
        # that physically exist in the executor's Polars frame. Virtual range
        # columns are excluded (merge drops range_columns per design), but
        # hidden columns are included because they are physically present and
        # pandas must resolve their suffixes naturally.
        left_pdtypes = left._data._physical_dtypes
        right_pdtypes = right._data._physical_dtypes

        if on is None and left_on is None and right_on is None:
            left_cols = set(left_pdtypes.index)
            right_cols = set(right_pdtypes.index)
            common = left_cols & right_cols
            on = sorted(common)
            self.on = on

        empty_left = _build_empty_df(left_pdtypes)
        empty_right = _build_empty_df(right_pdtypes)

        merged = empty_left.merge(
            empty_right,
            how=self.how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=self.suffixes,
        )

        out_pdtypes = merged.dtypes

        # Determine which output columns are hidden.
        # Hidden columns from either side may get suffixed by pandas merge.
        left_suffix, right_suffix = self.suffixes
        left_hidden = left._hidden_columns
        right_hidden = right._hidden_columns
        out_hidden = set()

        for hc in left_hidden:
            # After merge, this hidden column may have the left suffix if
            # it collided with a right hidden/visible column of the same name.
            out_name = hc if hc in out_pdtypes.index else hc + left_suffix
            out_hidden.add(out_name)

        for hc in right_hidden:
            # If the left side already claimed the unsuffixed name, the right
            # hidden column must use the right suffix even if the bare name
            # exists in the output (it belongs to the left side).
            if hc in out_hidden:
                out_name = hc + right_suffix
            else:
                out_name = hc if hc in out_pdtypes.index else hc + right_suffix
            out_hidden.add(out_name)

        out_frame_metadata = None
        if out_hidden:
            out_frame_metadata = FrameMetadata(hidden_columns=list(out_hidden))

        return self.new_liteframe(
            [left, right],
            shape=(np.nan, len(out_pdtypes)),
            physical_dtypes=out_pdtypes,
            frame_metadata=out_frame_metadata,
        )


class LiteFrameMergeAlign(MapReduceOperator, LiteFrameOperatorMixin):
    """
    Align and shuffle data for distributed merge operations.

    This operator handles the shuffle map and reduce stages for merge operations
    in distributed execution. It ensures data is properly partitioned and aligned
    across different chunks to enable efficient distributed merge operations.

    Parameters
    ----------
    stage : str
        Stage of the shuffle operation:
        - 'map': shuffle map stage that partitions and redistributes data
        - 'reduce': shuffle reduce stage that collects and merges partitioned data
    shuffle_on : str or list
        Column or index names to shuffle on. Determines the partitioning key
        for data distribution across workers.
    mapper_group_id : int, default 0
        Group identifier for the mapper in shuffle operations. Used to track
        and coordinate multiple mappers in distributed execution.
    index_shuffle_size : int
        Number of partitions for shuffle operation. Determines how data is
        distributed across workers for parallel processing.

    Returns
    -------
    LiteFrame
        Shuffled and aligned LiteFrame ready for merge operation.

    Examples
    --------
    >>> # Shuffle map stage
    >>> align_map = LiteFrameMergeAlign(
    ...     stage='map',
    ...     shuffle_on='key_column',
    ...     index_shuffle_size=10
    ... )
    >>> shuffled_data = align_map(input_frame)

    >>> # Shuffle reduce stage
    >>> align_reduce = LiteFrameMergeAlign(
    ...     stage='reduce',
    ...     shuffle_on='key_column',
    ...     mapper_group_id=0
    ... )
    >>> aligned_data = align_reduce(shuffled_chunks)

    Notes
    -----
    - Shuffle operations are essential for efficient distributed merge
    - Map stage partitions data based on merge keys
    - Reduce stage collects and aligns data from multiple mappers
    - The operator works in conjunction with LiteFrameMerge for full merge workflow
    """

    _op_type_ = opcodes.LITEFRAME_SHUFFLE_MERGE_ALIGN

    stage = ReferenceField("stage", OperatorStage, default=None)
    shuffle_on = AnyField("shuffle_on")
    mapper_group_id = Int32Field("mapper_group_id", default=0)
    index_shuffle_size = Int32Field("index_shuffle_size")

    def __init__(self, output_types=None, **kw):
        super().__init__(_output_types=output_types, **kw)
        if output_types is None:
            if self.stage == OperatorStage.map:
                output_types = [OutputType.liteframe]
            elif self.stage == OperatorStage.reduce:
                output_types = [OutputType.liteframe] * 2
        self._output_types = output_types

    @property
    def output_limit(self) -> int:
        return len(self.output_types)


class LiteFrameConcat(LiteFrameOperatorMixin, LiteFrameOperator):
    """
    Concatenate multiple LiteFrame objects along a specified axis.

    This operator performs chunk concatenation for LiteFrame objects, combining
    multiple input frames into a single output frame. It supports concatenation
    along row axis (vertical stacking) or column axis (horizontal stacking).

    Parameters
    ----------
    axis : int, default 0
        Axis along which to concatenate:
        - 0: concatenate along rows (vertical stacking), creating longer DataFrame
        - 1: concatenate along columns (horizontal stacking), creating wider DataFrame

    Returns
    -------
    LiteFrame
        Concatenated LiteFrame object combining all input frames.

    Examples
    --------
    >>> import maxframe.liteframe as ml
    >>> # Concatenate along rows (axis=0)
    >>> df1 = ml.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = ml.DataFrame({'A': [5, 6], 'B': [7, 8]})
    >>> concat_op = LiteFrameConcat(axis=0)
    >>> result = concat_op([df1, df2])
    >>> # Result: DataFrame with 4 rows and columns A, B

    >>> # Concatenate along columns (axis=1)
    >>> df1 = ml.DataFrame({'A': [1, 2]})
    >>> df2 = ml.DataFrame({'B': [3, 4]})
    >>> concat_op = LiteFrameConcat(axis=1)
    >>> result = concat_op([df1, df2])
    >>> # Result: DataFrame with 2 rows and columns A, B

    Notes
    -----
    - For axis=0: all inputs must have same column structure
    - For axis=1: all inputs must have same row structure (same length)
    - Concatenation preserves data types from the first input frame
    - The operator supports distributed concatenation of chunks
    - Output shape is inferred based on axis and input shapes
    """

    _op_type_ = opcodes.CONCATENATE
    _output_type_ = OutputType.liteframe

    axis = Int32Field("axis", default=0)

    def __call__(self, inputs):
        first = inputs[0]
        hidden_columns = None
        if self.axis == 0:
            # Row stacking: use first input's hidden columns
            if first.frame_metadata and first.frame_metadata.hidden_columns:
                hidden_columns = first.frame_metadata.hidden_columns
        else:
            # Column stacking: union of all inputs' hidden columns
            all_hidden = set()
            for inp in inputs:
                all_hidden |= inp._hidden_columns
            hidden_columns = list(all_hidden) if all_hidden else None

        frame_metadata = None
        if hidden_columns:
            frame_metadata = FrameMetadata(hidden_columns=hidden_columns)

        return self.new_liteframe(
            inputs,
            shape=(np.nan, len(first._data._physical_dtypes)),
            physical_dtypes=first._data._physical_dtypes,
            frame_metadata=frame_metadata,
        )
