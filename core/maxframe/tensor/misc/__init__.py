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

from .astype import TensorAstype, astype
from .atleast_1d import atleast_1d
from .atleast_2d import atleast_2d
from .atleast_3d import atleast_3d
from .broadcast_to import TensorBroadcastTo, broadcast_to
from .copy import copy
from .diff import TensorDiff, diff
from .flatten import flatten
from .in1d import in1d
from .isin import TensorIsIn, isin
from .ndim import ndim
from .ravel import ravel
from .repeat import TensorRepeat, repeat
from .searchsorted import TensorSearchsorted, searchsorted
from .setdiff1d import setdiff1d
from .squeeze import TensorSqueeze, squeeze
from .swapaxes import TensorSwapAxes, swapaxes
from .transpose import TensorTranspose, transpose
from .trapezoid import TensorTrapezoid, trapezoid
from .unique import TensorUnique, unique
from .where import TensorWhere, where


def _install():
    from ..core import Tensor, TensorData

    for cls in (Tensor, TensorData):
        setattr(cls, "astype", astype)
        setattr(cls, "flatten", flatten)
        setattr(cls, "swapaxes", swapaxes)
        setattr(cls, "ravel", ravel)


_install()
del _install
