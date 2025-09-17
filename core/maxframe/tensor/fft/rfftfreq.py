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

from ... import opcodes
from ...serialization.serializables import Float64Field, Int32Field
from ..core import TensorOrder
from ..operators import TensorOperator, TensorOperatorMixin


class TensorRFFTFreq(TensorOperator, TensorOperatorMixin):
    _op_type_ = opcodes.RFFTFREQ

    n = Int32Field("n")
    d = Float64Field("d")

    def __call__(self, chunk_size=None):
        shape = (self.n // 2 + 1,)
        return self.new_tensor(
            None, shape, raw_chunk_size=chunk_size, order=TensorOrder.C_ORDER
        )


def rfftfreq(n, d=1.0, gpu=None, chunk_size=None):
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft).

    The returned float tensor `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
    the Nyquist frequency component is considered to be positive.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension

    Returns
    -------
    f : Tensor
        Tensor of length ``n//2 + 1`` containing the sample frequencies.

    Examples
    --------
    >>> import maxframe.tensor as mt

    >>> signal = mt.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
    >>> fourier = mt.fft.rfft(signal)
    >>> n = signal.size
    >>> sample_rate = 100
    >>> freq = mt.fft.fftfreq(n, d=1./sample_rate)
    >>> freq.execute()
    array([  0.,  10.,  20.,  30.,  40., -50., -40., -30., -20., -10.])
    >>> freq = mt.fft.rfftfreq(n, d=1./sample_rate)
    >>> freq.execute()
    array([  0.,  10.,  20.,  30.,  40.,  50.])

    """
    n, d = int(n), float(d)
    op = TensorRFFTFreq(n=n, d=d, dtype=np.dtype(float), gpu=gpu)
    return op(chunk_size=chunk_size)
