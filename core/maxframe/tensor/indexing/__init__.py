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

from maxframe.tensor.indexing.choose import TensorChoose, choose
from maxframe.tensor.indexing.compress import compress
from maxframe.tensor.indexing.extract import extract
from maxframe.tensor.indexing.fill_diagonal import TensorFillDiagonal, fill_diagonal
from maxframe.tensor.indexing.flatnonzero import flatnonzero
from maxframe.tensor.indexing.getitem import TensorIndex
from maxframe.tensor.indexing.nonzero import TensorNonzero, nonzero
from maxframe.tensor.indexing.setitem import TensorIndexSetValue
from maxframe.tensor.indexing.slice import TensorSlice
from maxframe.tensor.indexing.take import take
from maxframe.tensor.indexing.unravel_index import TensorUnravelIndex, unravel_index


def _install():
    from maxframe.tensor.core import Tensor, TensorData
    from maxframe.tensor.indexing.getitem import _getitem
    from maxframe.tensor.indexing.setitem import _setitem

    setattr(Tensor, "__getitem__", _getitem)
    setattr(TensorData, "__getitem__", _getitem)
    setattr(Tensor, "__setitem__", _setitem)
    setattr(Tensor, "take", take)
    setattr(
        Tensor,
        "compress",
        lambda a, condition, axis=None: compress(condition, a, axis=axis),
    )
    setattr(Tensor, "choose", choose)
    setattr(Tensor, "nonzero", nonzero)


_install()
del _install
