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

from .argwhere import TensorArgwhere, argwhere
from .array_split import array_split
from .astype import TensorAstype, astype
from .atleast_1d import atleast_1d
from .atleast_2d import atleast_2d
from .atleast_3d import atleast_3d
from .broadcast_arrays import broadcast_arrays
from .broadcast_to import TensorBroadcastTo, broadcast_to
from .copy import copy
from .copyto import TensorCopyTo, copyto
from .delete import TensorDelete, delete
from .diff import TensorDiff, diff
from .dsplit import dsplit
from .ediff1d import ediff1d
from .expand_dims import expand_dims
from .flatten import flatten
from .flip import flip
from .fliplr import fliplr
from .flipud import flipud
from .hsplit import hsplit
from .in1d import in1d
from .insert import insert
from .isin import TensorIsIn, isin
from .moveaxis import moveaxis
from .ndim import ndim
from .ravel import ravel
from .repeat import TensorRepeat, repeat
from .result_type import result_type
from .roll import roll
from .rollaxis import rollaxis
from .searchsorted import TensorSearchsorted, searchsorted
from .setdiff1d import setdiff1d
from .shape import TensorGetShape, shape
from .split import TensorSplit, split
from .squeeze import TensorSqueeze, squeeze
from .swapaxes import TensorSwapAxes, swapaxes
from .tile import tile
from .transpose import TensorTranspose, transpose
from .trapezoid import TensorTrapezoid, trapezoid
from .unique import TensorUnique, unique
from .vsplit import vsplit
from .where import TensorWhere, where


def _install():
    from ..core import Tensor, TensorData

    for cls in (Tensor, TensorData):
        setattr(cls, "astype", astype)
        setattr(cls, "flatten", flatten)
        setattr(cls, "ravel", ravel)
        setattr(cls, "repeat", repeat)
        setattr(cls, "squeeze", squeeze)
        setattr(cls, "swapaxes", swapaxes)


_install()
del _install
