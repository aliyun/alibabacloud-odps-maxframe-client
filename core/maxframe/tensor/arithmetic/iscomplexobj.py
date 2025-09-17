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

import numpy as np


def iscomplexobj(x):
    """
    Check for a complex type or an array of complex numbers.

    The type of the input is checked, not the value. Even if the input
    has an imaginary part equal to zero, `iscomplexobj` evaluates to True.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    iscomplexobj : bool
        The return value, True if `x` is of a complex type or has at least
        one complex element.

    See Also
    --------
    isrealobj, iscomplex

    Examples
    --------
    >>> import maxframe.tensor as mt
    >>> mt.iscomplexobj(1)
    False
    >>> mt.iscomplexobj(1+0j)
    True
    >>> mt.iscomplexobj([3, 1+0j, True])
    True
    """
    from ..datasource.array import tensor as astensor

    t = astensor(x)
    return issubclass(t.dtype.type, np.complexfloating)
