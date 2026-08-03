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

from maxframe import opcodes
from maxframe.core import OutputType
from maxframe.core.operator import OperatorStage
from maxframe.liteframe.operators.core import LiteFrameOperator, LiteFrameOperatorMixin
from maxframe.serialization.serializables import (
    AnyField,
    Float32Field,
    Int32Field,
    ReferenceField,
)


class LiteFrameBloomFilter(LiteFrameOperatorMixin, LiteFrameOperator):
    """
    Bloom filter optimization for LiteFrame inner join operations.

    Implements a three-stage bloom filter pipeline that reduces data volume
    before shuffle merge by eliminating rows that cannot match the join
    condition. Only applicable to inner joins.

    Parameters
    ----------
    stage : str
        Stage of the bloom filter pipeline:
        - 'build': Constructs a bloom filter from the smaller side's join keys
        - 'union': Tree-reduction to combine bloom filters from multiple chunks
        - 'filter': Filters the larger side's chunks against the combined filter
    left_on : str or list, optional
        Column names to join on from the left LiteFrame.
    right_on : str or list, optional
        Column names to join on from the right LiteFrame.
    on : str or list, optional
        Column names to join on, present in both LiteFrames.
    max_elements : int, default 10000
        Maximum number of elements expected in the bloom filter.
    error_rate : float, default 0.1
        Desired false positive rate for the bloom filter.
    combine_size : int, default 4
        Number of bloom filters to combine in each union tree-reduction step.

    Notes
    -----
    - Only applicable to inner joins where filtering is safe
    - Bloom filter construction is on the smaller side, filtering on the larger
    - The 'union' stage uses tree-reduction for efficient distributed combination
    - Operates on Polars DataFrames at execution time
    """

    _op_type_ = opcodes.LITEFRAME_BLOOM_FILTER

    stage = ReferenceField("stage", OperatorStage, default=None)
    left_on = AnyField("left_on", default=None)
    right_on = AnyField("right_on", default=None)
    on = AnyField("on", default=None)
    max_elements = Int32Field("max_elements", default=10000)
    error_rate = Float32Field("error_rate", default=0.1)
    combine_size = Int32Field("combine_size", default=4)

    def __init__(self, output_types=None, **kwargs):
        stage = kwargs.get("stage", self.stage)
        if output_types is None:
            if stage in (OperatorStage.map, OperatorStage.combine):
                output_types = [OutputType.object]
            else:
                output_types = [OutputType.liteframe]
        kwargs["_output_types"] = output_types
        super().__init__(**kwargs)

    def __call__(self, *inputs):
        if self.stage in (OperatorStage.map, OperatorStage.combine):
            # Build/union: single input, produces bloom filter object
            inp = inputs[0]
            return self.new_tileables(
                [inp],
                shape=(),
            )[0]
        else:
            # Filter: two inputs, produces LiteFrame
            liteframe = inputs[0]
            return self.new_liteframe(
                list(inputs),
                shape=(np.nan, liteframe.shape[1]),
                physical_dtypes=liteframe._physical_dtypes,
                frame_metadata=liteframe.frame_metadata,
            )
