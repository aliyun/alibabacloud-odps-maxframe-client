#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import numpy as np
import pytest

from ...datasource import empty, ones
from .. import stack


def test_stack():
    raw_arrs = [ones((3, 4), chunk_size=2) for _ in range(10)]
    arr2 = stack(raw_arrs, axis=0)

    assert arr2.shape == (10, 3, 4)

    arr3 = stack(raw_arrs, axis=1)

    assert arr3.shape == (3, 10, 4)

    arr4 = stack(raw_arrs, axis=2)

    assert arr4.shape == (3, 4, 10)

    with pytest.raises(ValueError):
        raw_arrs2 = [ones((3, 4), chunk_size=2), ones((4, 3), chunk_size=2)]
        stack(raw_arrs2)

    with pytest.raises(np.AxisError):
        stack(raw_arrs, axis=3)

    with pytest.raises(np.AxisError):
        stack(raw_arrs, axis=-4)

    with pytest.raises(TypeError):
        stack(raw_arrs, out=1)

    with pytest.raises(ValueError):
        stack(raw_arrs, empty((1, 10, 3, 4)))
