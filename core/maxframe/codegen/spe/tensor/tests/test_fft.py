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

import maxframe.tensor as mt

from ...core import SPECodeContext
from ..fft import (
    TensorFFTAdapter,
    TensorFFTFreqAdapter,
    TensorFFTNAdapter,
    TensorFFTShiftAdapter,
)


def test_fft():
    result = mt.fft.fft(mt.exp(2j * mt.pi * mt.arange(8) / 8))

    adapter = TensorFFTAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = np.fft.fft(var_0, axis=-1)"]
    assert results == expected_results


def test_fft2():
    a = mt.mgrid[:5, :5][0]
    result = mt.fft.fft2(a)

    adapter = TensorFFTNAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = np.fft.fft2(var_0, None, axes=(-2, -1))"]
    assert results == expected_results


def test_fftshift():
    result = mt.fft.fftshift(mt.tensor(10), 0.1)

    adapter = TensorFFTShiftAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_1 = np.fft.fftshift(var_0, axes=(0.1,))"]
    assert results == expected_results


def test_fftfreq():
    result = mt.fft.fftfreq(10, 0.1)

    adapter = TensorFFTFreqAdapter()
    context = SPECodeContext()
    results = adapter.generate_code(result.op, context)
    expected_results = ["var_0 = np.fft.fftfreq(10, d=0.1)"]
    assert results == expected_results
