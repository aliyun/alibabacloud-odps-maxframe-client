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

from numbers import Integral
from typing import List

import numpy as np

from ... import opcodes
from ...core import EntityData
from ...serialization.serializables import (
    AnyField,
    BoolField,
    FieldTypes,
    KeyField,
    TupleField,
)
from ..core import TENSOR_TYPE, TensorOrder
from ..datasource import array
from ..operators import TensorOperatorMixin
from ..utils import gen_random_seeds
from .core import TensorRandomOperator


class TensorChoice(TensorRandomOperator, TensorOperatorMixin):
    _op_type_ = opcodes.RAND_CHOICE

    a = AnyField("a")
    size = TupleField("size", FieldTypes.int64)
    replace = BoolField("replace")
    p = KeyField("p")

    @classmethod
    def _set_inputs(cls, op: "TensorChoice", inputs: List[EntityData]):
        super()._set_inputs(op, inputs)
        if isinstance(op.a, TENSOR_TYPE):
            op.a = op._inputs[0]
        if isinstance(op.p, TENSOR_TYPE):
            op.p = op._inputs[-1]

    def __call__(self, a, p, chunk_size=None):
        inputs = []
        if isinstance(a, TENSOR_TYPE):
            inputs.append(a)
        if isinstance(p, TENSOR_TYPE):
            inputs.append(p)
        return self.new_tensor(
            inputs,
            shape=self.size,
            raw_chunk_size=chunk_size,
            order=TensorOrder.C_ORDER,
        )


def choice(random_state, a, size=None, replace=True, p=None, chunk_size=None, gpu=None):
    """
    Generates a random sample from a given 1-D array

    Parameters
    -----------
    a : 1-D array-like or int
        If a tensor, a random sample is generated from its elements.
        If an int, the random sample is generated as if a were mt.arange(a)
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement
    p : 1-D array-like, optional
        The probabilities associated with each entry in a.
        If not given the sample assumes a uniform distribution over all
        entries in a.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default

    Returns
    --------
    samples : single item or tensor
        The generated random samples

    Raises
    -------
    ValueError
        If a is an int and less than zero, if a or p are not 1-dimensional,
        if a is an array-like of size 0, if p is not a vector of
        probabilities, if a and p have different lengths, or if
        replace=False and the sample size is greater than the population
        size

    See Also
    ---------
    randint, shuffle, permutation

    Examples
    ---------
    Generate a uniform random sample from mt.arange(5) of size 3:

    >>> import maxframe.tensor as mt

    >>> mt.random.choice(5, 3).execute()
    array([0, 3, 4])
    >>> #This is equivalent to mt.random.randint(0,5,3)

    Generate a non-uniform random sample from np.arange(5) of size 3:

    >>> mt.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0]).execute()
    array([3, 3, 0])

    Generate a uniform random sample from mt.arange(5) of size 3 without
    replacement:

    >>> mt.random.choice(5, 3, replace=False).execute()
    array([3,1,0])
    >>> #This is equivalent to np.random.permutation(np.arange(5))[:3]

    Generate a non-uniform random sample from mt.arange(5) of size
    3 without replacement:

    >>> mt.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0]).execute()
    array([2, 3, 0])

    Any of the above can be repeated with an arbitrary array-like
    instead of just integers. For instance:

    >>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
    >>> np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
    array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'],
          dtype='|S11')
    """

    if isinstance(a, Integral):
        if a <= 0:
            raise ValueError("a must be greater than 0")
        a_size = a
        dtype = np.random.choice(
            1, size=(), p=np.array([1]) if p is not None else p
        ).dtype
    else:
        a = array(a)
        if a.ndim != 1:
            raise ValueError("a must be one dimensional")
        a_size = a.size
        dtype = a.dtype

    if p is not None:
        if not isinstance(p, TENSOR_TYPE):
            p = np.asarray(p)
            if not np.isclose(p.sum(), 1, rtol=1e-7, atol=0):
                raise ValueError("probabilities do not sum to 1")
            p = array(p, chunk_size=p.size)
        if p.ndim != 1:
            raise ValueError("p must be one dimensional")

    if size is None:
        size = ()
        length = 1
    else:
        try:
            tuple(size)
            length = np.prod(size)
        except TypeError:
            length = size
    if replace is False and length > a_size:
        raise ValueError(
            "Cannot take a larger sample than population when 'replace=False'"
        )

    size = random_state._handle_size(size)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorChoice(
        a=a, p=p, seed=seed, replace=replace, size=size, dtype=dtype, gpu=gpu
    )
    return op(a, p, chunk_size=chunk_size)
