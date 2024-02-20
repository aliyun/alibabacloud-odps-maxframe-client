# Copyright 1999-2024 Alibaba Group Holding Ltd.
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

from .astype import TensorAstype
from .broadcast_to import TensorBroadcastTo, broadcast_to
from .ravel import ravel
from .transpose import transpose
from .where import TensorWhere, where


def _install():
    from ..core import Tensor, TensorData
    from .astype import _astype

    for cls in (Tensor, TensorData):
        setattr(cls, "astype", _astype)
        setattr(cls, "ravel", ravel)


_install()
del _install
