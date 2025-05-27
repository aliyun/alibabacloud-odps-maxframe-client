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

from ...config import options


def arithmetic_operator(cls=None, init=True, sparse_mode=None):
    def _decorator(cls):
        def __init__(self, casting="same_kind", err=None, **kw):
            err = err if err is not None else np.geterr()
            super(cls, self).__init__(casting=casting, err=err, **kw)

        def _is_sparse_binary_and_const(x1, x2):
            if all(np.isscalar(x) for x in [x1, x2]):
                return False
            if all(hasattr(x, "issparse") and x.issparse() for x in [x1, x2]):
                return True
            if np.isscalar(x1):
                return cls._is_sparse_with_scalar(x1, True)
            if np.isscalar(x2):
                return cls._is_sparse_with_scalar(x2, False)
            return False

        def _is_sparse_binary_or_const(x1, x2):
            if (hasattr(x1, "issparse") and x1.issparse()) or (
                hasattr(x2, "issparse") and x2.issparse()
            ):
                return True
            return False

        _is_sparse_dict = dict(
            always_false=lambda *_: False,
            unary=lambda x: x.issparse(),
            binary_and=_is_sparse_binary_and_const,
            binary_or=_is_sparse_binary_or_const,
        )
        for v in _is_sparse_dict.values():
            v.__name__ = "_is_sparse"

        if init:
            cls.__init__ = __init__

        if sparse_mode in _is_sparse_dict:
            cls._is_sparse = staticmethod(_is_sparse_dict[sparse_mode])
        elif sparse_mode is not None:  # pragma: no cover
            raise ValueError(f"Unsupported sparse mode: {sparse_mode}")

        return cls

    if cls is not None:
        return _decorator(cls)
    else:
        return _decorator


class TreeReductionBuilder:
    def __init__(self, combine_size=None):
        self._combine_size = combine_size or options.dpe.reduction.combine_size

    def _build_reduction(self, inputs, final=False):
        raise NotImplementedError

    def build(self, inputs):
        combine_size = self._combine_size
        while len(inputs) > self._combine_size:
            new_inputs = []
            for i in range(0, len(inputs), combine_size):
                objs = inputs[i : i + combine_size]
                if len(objs) == 1:
                    obj = objs[0]
                else:
                    obj = self._build_reduction(objs, final=False)
                new_inputs.append(obj)
            inputs = new_inputs

        if len(inputs) == 1:
            return inputs[0]
        return self._build_reduction(inputs, final=True)
