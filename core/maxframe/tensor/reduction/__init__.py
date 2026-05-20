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

from maxframe.tensor.reduction.all import TensorAll, all
from maxframe.tensor.reduction.allclose import allclose
from maxframe.tensor.reduction.any import TensorAny, any
from maxframe.tensor.reduction.argmax import TensorArgmax, argmax
from maxframe.tensor.reduction.argmin import TensorArgmin, argmin
from maxframe.tensor.reduction.array_equal import array_equal
from maxframe.tensor.reduction.count_nonzero import TensorCountNonzero, count_nonzero
from maxframe.tensor.reduction.cumprod import TensorCumprod, cumprod
from maxframe.tensor.reduction.cumsum import TensorCumsum, cumsum
from maxframe.tensor.reduction.max import TensorMax, max
from maxframe.tensor.reduction.mean import TensorMean, mean
from maxframe.tensor.reduction.min import TensorMin, min
from maxframe.tensor.reduction.nanargmax import TensorNanArgmax, nanargmax
from maxframe.tensor.reduction.nanargmin import TensorNanArgmin, nanargmin
from maxframe.tensor.reduction.nancumprod import TensorNanCumprod, nancumprod
from maxframe.tensor.reduction.nancumsum import TensorNanCumsum, nancumsum
from maxframe.tensor.reduction.nanmax import TensorNanMax, nanmax
from maxframe.tensor.reduction.nanmean import TensorNanMean, nanmean
from maxframe.tensor.reduction.nanmin import TensorNanMin, nanmin
from maxframe.tensor.reduction.nanprod import TensorNanProd, nanprod
from maxframe.tensor.reduction.nanstd import nanstd
from maxframe.tensor.reduction.nansum import TensorNanSum, nansum
from maxframe.tensor.reduction.nanvar import TensorNanMoment, TensorNanVar, nanvar
from maxframe.tensor.reduction.prod import TensorProd, prod
from maxframe.tensor.reduction.std import std
from maxframe.tensor.reduction.sum import TensorSum, sum
from maxframe.tensor.reduction.var import TensorMoment, TensorVar, var


def _install():
    from maxframe.tensor.core import Tensor, TensorData

    for cls in (Tensor, TensorData):
        setattr(cls, "sum", sum)
        setattr(cls, "prod", prod)
        setattr(cls, "max", max)
        setattr(cls, "min", min)
        setattr(cls, "all", all)
        setattr(cls, "any", any)
        setattr(cls, "mean", mean)
        setattr(cls, "argmax", argmax)
        setattr(cls, "argmin", argmin)
        setattr(cls, "cumsum", cumsum)
        setattr(cls, "cumprod", cumprod)
        setattr(cls, "var", var)
        setattr(cls, "std", std)


_install()
del _install
