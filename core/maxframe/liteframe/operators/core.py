# Copyright 1999-2026 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum

from maxframe.core import OutputType
from maxframe.core.operator import Operator, TileableOperatorMixin
from maxframe.liteframe.core import FrameMetadata


def _alignment_id_of(tileable):
    """Return the alignment_id of *tileable*, or ``None`` if it has no
    metadata or the metadata has no alignment_id."""
    md = getattr(tileable, "frame_metadata", None)
    return md.alignment_id if md is not None else None


class LiteFrameDataFormat(enum.Enum):
    """File format for lake data source and store operators."""

    PARQUET = 0
    CSV = 1
    JSON = 2


class LiteFrameOperatorMixin(TileableOperatorMixin):
    __slots__ = ()
    _op_module_ = "liteframe"
    _supports_dynamic_schema = False

    def _inherit_alignment_id(self, inputs):
        """Return the alignment_id(s) the output(s) should inherit.

        May return a single value (applied to every output) or a list
        with one entry per output. ``None`` in any position means "no
        inheritance" — that output keeps its own random
        ``alignment_id`` (the field default). Operators that are 1:1
        row-preserving (projection, rank-with-place-back, aligned
        axis=1 concat) override this to return the input's id.
        """
        return None

    @staticmethod
    def _inherit_single_input_alignment_id(inputs):
        """Helper for 1:1 row-preserving operators: return the sole data
        input's alignment_id, or ``None`` if there are no inputs or the
        input has no metadata."""
        if not inputs:
            return None
        return _alignment_id_of(inputs[0])

    def _apply_alignment_id(self, tileables, inputs):
        """Ensure every liteframe output carries a proper alignment_id.

        Attach metadata if missing, then either inherit the input's id
        (for 1:1 row-preserving operators) or keep the metadata's own
        random id. Explicit ``alignment_id=None`` on the metadata is
        respected (means "unknown").
        """
        inherited = self._inherit_alignment_id(inputs)
        if not isinstance(inherited, list):
            inherited = [inherited] * len(tileables)
        for tileable, inherited_id in zip(tileables, inherited):
            data = getattr(tileable, "_data", None)
            if data is None:
                data = getattr(tileable, "data", None)
            if data is None:
                continue
            md = data.frame_metadata
            if md is None:
                md = FrameMetadata()
                data.frame_metadata = md
            if inherited_id is not None:
                md.alignment_id = inherited_id
            else:
                # Materialize the default random id so it is stable across
                # copies/rebuilds (Field default_factory fires on access).
                _ = md.alignment_id

    def new_liteframes(
        self, inputs, shape=None, physical_dtypes=None, frame_metadata=None, **kw
    ):
        setattr(self, "_output_types", [OutputType.liteframe])
        tileables = self.new_tileables(
            inputs,
            shape=shape,
            physical_dtypes=physical_dtypes,
            frame_metadata=frame_metadata,
            **kw,
        )
        self._apply_alignment_id(tileables, inputs)
        return tileables

    def new_liteframe(self, inputs, **kw):
        return self.new_liteframes(inputs, **kw)[0]

    def new_scalars(self, inputs, dtype=None, **kw):
        setattr(self, "_output_types", [OutputType.scalar])
        return self.new_tileables(inputs, shape=(), dtype=dtype, **kw)

    def new_scalar(self, inputs, dtype=None, **kw):
        return self.new_scalars(inputs, dtype=dtype, **kw)[0]


LiteFrameOperator = Operator
