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

from maxframe.tensor.misc.argwhere import TensorArgwhere, argwhere
from maxframe.tensor.misc.array_split import array_split
from maxframe.tensor.misc.astype import TensorAstype, astype
from maxframe.tensor.misc.atleast_1d import atleast_1d
from maxframe.tensor.misc.atleast_2d import atleast_2d
from maxframe.tensor.misc.atleast_3d import atleast_3d
from maxframe.tensor.misc.broadcast_arrays import broadcast_arrays
from maxframe.tensor.misc.broadcast_to import TensorBroadcastTo, broadcast_to
from maxframe.tensor.misc.copy import copy
from maxframe.tensor.misc.copyto import TensorCopyTo, copyto
from maxframe.tensor.misc.delete import TensorDelete, delete
from maxframe.tensor.misc.diff import TensorDiff, diff
from maxframe.tensor.misc.dsplit import dsplit
from maxframe.tensor.misc.ediff1d import ediff1d
from maxframe.tensor.misc.expand_dims import expand_dims
from maxframe.tensor.misc.flatten import flatten
from maxframe.tensor.misc.flip import flip
from maxframe.tensor.misc.fliplr import fliplr
from maxframe.tensor.misc.flipud import flipud
from maxframe.tensor.misc.hsplit import hsplit
from maxframe.tensor.misc.in1d import in1d
from maxframe.tensor.misc.insert import insert
from maxframe.tensor.misc.isin import TensorIsIn, isin
from maxframe.tensor.misc.moveaxis import moveaxis
from maxframe.tensor.misc.ndim import ndim
from maxframe.tensor.misc.ravel import ravel
from maxframe.tensor.misc.repeat import TensorRepeat, repeat
from maxframe.tensor.misc.result_type import result_type
from maxframe.tensor.misc.roll import roll
from maxframe.tensor.misc.rollaxis import rollaxis
from maxframe.tensor.misc.searchsorted import TensorSearchsorted, searchsorted
from maxframe.tensor.misc.setdiff1d import setdiff1d
from maxframe.tensor.misc.shape import TensorGetShape, shape
from maxframe.tensor.misc.split import TensorSplit, split
from maxframe.tensor.misc.squeeze import TensorSqueeze, squeeze
from maxframe.tensor.misc.swapaxes import TensorSwapAxes, swapaxes
from maxframe.tensor.misc.tile import tile
from maxframe.tensor.misc.transpose import TensorTranspose, transpose
from maxframe.tensor.misc.trapezoid import TensorTrapezoid, trapezoid
from maxframe.tensor.misc.unique import TensorUnique, unique
from maxframe.tensor.misc.vsplit import vsplit
from maxframe.tensor.misc.where import TensorWhere, where


def _install():
    from maxframe.tensor.core import Tensor, TensorData

    for cls in (Tensor, TensorData):
        setattr(cls, "astype", astype)
        setattr(cls, "flatten", flatten)
        setattr(cls, "ravel", ravel)
        setattr(cls, "repeat", repeat)
        setattr(cls, "squeeze", squeeze)
        setattr(cls, "swapaxes", swapaxes)


_install()
del _install
