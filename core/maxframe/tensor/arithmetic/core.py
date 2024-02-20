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

from ...core import ExecutableTuple
from ...serialization.serializables import (
    AnyField,
    DictField,
    FieldTypes,
    KeyField,
    ListField,
    StringField,
)
from ..core import Tensor, TensorOrder
from ..datasource import tensor as astensor
from ..operators import TensorOperator, TensorOperatorMixin
from ..utils import broadcast_shape, check_order, check_out_param, filter_inputs


class TensorElementWise(TensorOperatorMixin):
    __slots__ = ()


class TensorElementWiseWithInputs(TensorElementWise):
    def _set_sparse(self, inputs):
        raise NotImplementedError

    def _new_tileables(self, inputs, kws=None, **kw):
        self._set_sparse(inputs)
        return super()._new_tileables(inputs, kws=kws, **kw)


def _handle_out_dtype(val, dtype):
    if val.dtype != dtype:
        return val.astype(dtype)
    return val


class TensorBinOpMixin(TensorElementWiseWithInputs):
    __slots__ = ()

    def check_inputs(self, inputs):
        if len(inputs) > 4:
            raise ValueError(
                f"Binary operator's inputs should less than or equal 4, got {len(inputs)}"
            )

    @classmethod
    def _get_func(cls, xp):
        func_name = getattr(cls, "_func_name")
        return getattr(xp, func_name)


class TensorBinOp(TensorOperator, TensorBinOpMixin):
    lhs = AnyField("lhs", default=None)
    rhs = AnyField("rhs", default=None)
    out = KeyField("out", default=None)
    where = KeyField("where", default=None)
    casting = StringField("casting", default=None)
    order = StringField("order", default=None)
    err = DictField("err", FieldTypes.string, FieldTypes.string, default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.order is None:
            self.order = "K"
        check_order(self.order)

    @classmethod
    def _is_sparse(cls, x1, x2):
        return False

    def _set_sparse(self, inputs):
        inputs_iter = iter(inputs)
        x1 = self.lhs if np.isscalar(self.lhs) else next(inputs_iter)
        x2 = self.rhs if np.isscalar(self.rhs) else next(inputs_iter)
        setattr(self, "sparse", self._is_sparse(x1, x2))

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)

        self.lhs = self.lhs if np.isscalar(self.lhs) else next(inputs_iter)
        self.rhs = self.rhs if np.isscalar(self.rhs) else next(inputs_iter)
        if getattr(self, "out", None) is not None:
            self.out = next(inputs_iter)
        if getattr(self, "where", None) is not None:
            self.where = next(inputs_iter)

    def _process_inputs(self, x1, x2, out, where):
        x1 = x1 if np.isscalar(x1) else astensor(x1)
        x2 = x2 if np.isscalar(x2) else astensor(x2)
        self.lhs = x1
        self.rhs = x2

        if out is not None:
            if isinstance(out, Tensor):
                self.out = out
            else:
                raise TypeError(f"out should be Tensor object, got {type(out)} instead")
        if where is True:
            where = None
        if where is not None:
            where = astensor(where)
            self.where = where

        return x1, x2, out, where

    def _calc_order(self, x1, x2, out):
        if out is not None:
            return out.order

        if self.order in "KA":
            orders = []
            if not np.isscalar(x1):
                orders.append(x1.order)
            if not np.isscalar(x2):
                orders.append(x2.order)

            if len(orders) == 0:
                return TensorOrder.C_ORDER
            elif any(order == TensorOrder.C_ORDER for order in orders):
                return TensorOrder.C_ORDER
            else:
                return TensorOrder.F_ORDER
        elif self.order == "C":
            return TensorOrder.C_ORDER
        else:
            return TensorOrder.F_ORDER

    @property
    def ufunc_extra_params(self):
        return dict()

    def _call_tensor_ufunc(self, x1, x2, out=None, where=None):
        if hasattr(x1, "__tensor_ufunc__") or hasattr(x2, "__tensor_ufunc__"):
            ufunc = (
                x1.__tensor_ufunc__
                if hasattr(x1, "__tensor_ufunc__")
                else x2.__tensor_ufunc__
            )
            ret = ufunc(type(self), [x1, x2], out, where, **self.ufunc_extra_params)
            if ret is NotImplemented:
                return
            return ret

    def _call(self, x1, x2, out=None, where=None):
        # check tensor ufunc, if x1 or x2 is not a tensor, e.g. MaxFrame DataFrame
        # which implements tensor ufunc, will delegate the computation
        # to it if possible
        ret = self._call_tensor_ufunc(x1, x2, out=out, where=where)
        if ret is not None:
            return ret

        x1, x2, out, where = self._process_inputs(x1, x2, out, where)
        # check broadcast
        x1_shape = () if np.isscalar(x1) else x1.shape
        x2_shape = () if np.isscalar(x2) else x2.shape
        shape = broadcast_shape(x1_shape, x2_shape)
        order = self._calc_order(x1, x2, out)

        inputs = filter_inputs([x1, x2, out, where])
        t = self.new_tensor(inputs, shape, order=order)

        if out is None:
            return t

        check_out_param(out, t, self.casting)
        out_shape, out_dtype = out.shape, out.dtype

        # if `out` is specified, use out's dtype and shape
        if t.shape != out_shape:
            t = self.new_tensor(inputs, out_shape, order=order)
        setattr(self, "dtype", out_dtype)

        out.data = t.data
        return out

    def __call__(self, x1, x2, out=None, where=None):
        return self._call(x1, x2, out=out, where=where)

    def rcall(self, x1, x2, out=None, where=None):
        return self._call(x2, x1, out=out, where=where)


class TensorUnaryOpMixin(TensorElementWiseWithInputs):
    __slots__ = ()

    def check_inputs(self, inputs):
        if len(inputs) > 3:
            raise ValueError(
                f"Binary operator's inputs should less than or equal 3, got {len(inputs)}"
            )

    @classmethod
    def _get_func(cls, xp):
        func_name = getattr(cls, "_func_name")
        return getattr(xp, func_name)


class TensorUnaryOp(TensorOperator, TensorUnaryOpMixin):
    _input = KeyField("input")
    out = KeyField("out", default=None)
    where = KeyField("where", default=None)
    casting = StringField("casting", default=None)
    order = StringField("order", default=None)
    err = DictField("err", FieldTypes.string, FieldTypes.string, default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.order is None:
            self.order = "K"
        check_order(self.order)

    @property
    def input(self):
        return self._input

    @classmethod
    def _is_sparse(cls, x):
        if hasattr(x, "issparse") and x.issparse():
            return True
        else:
            return False

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)

        self._input = next(inputs_iter)
        if getattr(self, "out", None) is not None:
            self.out = next(inputs_iter)
        if getattr(self, "where", None) is not None:
            self.where = next(inputs_iter)

    def _process_inputs(self, x, out, where):
        x = astensor(x)

        if out is not None:
            if isinstance(out, Tensor):
                self.out = out
            else:
                raise TypeError(f"out should be Tensor object, got {type(out)} instead")
        if where is True:
            where = None
        if where is not None:
            where = astensor(where)
            self.where = where

        return x, out, where

    def _set_sparse(self, inputs):
        setattr(self, "sparse", self._is_sparse(inputs[0]))

    def _calc_order(self, x, out):
        if out is not None:
            return out.order

        if self.order in "KA":
            return x.order
        elif self.order == "C":
            return TensorOrder.C_ORDER
        else:
            return TensorOrder.F_ORDER

    @property
    def ufunc_extra_params(self):
        return dict()

    def _call_tensor_ufunc(self, x, out=None, where=None):
        if hasattr(x, "__tensor_ufunc__"):
            ret = x.__tensor_ufunc__(
                type(self), [x], out, where, **self.ufunc_extra_params
            )
            if ret is NotImplemented:
                return
            return ret

    def _call(self, x, out=None, where=None):
        # check tensor ufunc, if x is not a tensor, e.g. MaxFrame DataFrame
        # which implements tensor ufunc, will delegate the computation
        # to it if possible
        ret = self._call_tensor_ufunc(x, out=out, where=where)
        if ret is not None:
            return ret

        x, out, where = self._process_inputs(x, out, where)
        shape = x.shape
        order = self._calc_order(x, out)

        inputs = filter_inputs([x, out, where])
        t = self.new_tensor(inputs, shape, order=order)

        if out is None:
            return t

        check_out_param(out, t, getattr(self, "casting"))
        out_shape, out_dtype = out.shape, out.dtype

        # if `out` is specified, use out's dtype and shape
        if t.shape != out_shape:
            t = self.new_tensor(inputs, out_shape, order=order)
        setattr(self, "dtype", out_dtype)

        out.data = t.data
        return out

    def __call__(self, x, out=None, where=None):
        return self._call(x, out=out, where=where)


class TensorOutBinOp(TensorOperator, TensorElementWiseWithInputs):
    _input = KeyField("input")
    out1 = KeyField("out1", default=None)
    out2 = KeyField("out2", default=None)
    where = KeyField("where", default=None)
    order = StringField("order", default=None)
    casting = StringField("casting", default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.order is None:
            self.order = "K"
        check_order(self.order)

    @property
    def output_limit(self):
        return 2

    @property
    def input(self):
        return self._input

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)

        self._input = next(inputs_iter)
        if getattr(self, "out1", None) is not None:
            self.out1 = next(inputs_iter)
        if getattr(self, "out2", None) is not None:
            self.out2 = next(inputs_iter)
        if getattr(self, "where", None) is not None:
            self.where = next(inputs_iter)

    def _process_inputs(self, x, out1, out2, where):
        x = astensor(x)

        if out1 is not None:
            if isinstance(out1, Tensor):
                self.out1 = out1
            else:
                raise TypeError(
                    f"out1 should be Tensor object, got {type(out1)} instead"
                )
        if out2 is not None:
            if isinstance(out2, Tensor):
                self.out2 = out2
            else:
                raise TypeError(
                    f"out2 should be Tensor object, got {type(out2)} instead"
                )
        if where is True:
            where = None
        if where is not None:
            where = astensor(where)
            self.where = where

        return x, out1, out2, where

    @classmethod
    def _is_sparse(cls, x):
        return False

    def _set_sparse(self, inputs):
        setattr(self, "sparse", self._is_sparse(inputs[0]))

    @property
    def _fun(self):
        raise NotImplementedError

    def _calc_order(self, x, out):
        if out is not None:
            return out.order

        if self.order in "KA":
            return x.order
        elif self.order == "C":
            return TensorOrder.C_ORDER
        else:
            return TensorOrder.F_ORDER

    def _call(self, x, out1=None, out2=None, out=None, where=None):
        dtype = [r.dtype for r in self._fun(np.empty(1, dtype=x.dtype))]

        out = out or (None, None)
        out1 = out1 or out[0]
        out2 = out2 or out[1]
        x, out1, out2, where = self._process_inputs(x, out1, out2, where)
        shape = x.shape
        order1 = self._calc_order(x, out1)
        order2 = self._calc_order(x, out2)

        inputs = filter_inputs([x, out1, out2, where])
        t1, t2 = self.new_tensors(
            inputs,
            shape,
            kws=[
                {"order": order1, "dtype": dtype[0], "side": "left"},
                {"order": order2, "dtype": dtype[1], "side": "right"},
            ],
        )

        if out1 is None and out2 is None:
            return ExecutableTuple([t1, t2])

        if out1 is not None:
            check_out_param(out1, t1, self.casting)
            out1_shape, out1_dtype = out1.shape, out1.dtype
        else:
            out1_shape, out1_dtype = t1.shape, t1.dtype
        if out2 is not None:
            check_out_param(out2, t2, self.casting)
            out2_shape, out2_dtype = out2.shape, out2.dtype
        else:
            out2_shape, out2_dtype = t2.shape, t2.dtype
        # if `out` is specified, use out's dtype and shape
        if t1.shape != out1_shape or t2.shape != out2_shape:
            t1, t2 = self.new_tensor(
                inputs,
                [out1_shape, out2_shape],
                kws=[
                    {"order": order1, "dtype": out1_dtype},
                    {"order": order2, "dtype": out2_dtype},
                ],
            )

        if out1 is not None:
            out1.data = t1.data
        else:
            out1 = t1
        if out2 is not None:
            out2.data = t2.data
        else:
            out2 = t2
        return ExecutableTuple([out1, out2])

    def __call__(self, x, out1=None, out2=None, out=None, where=None):
        return self._call(x, out1=out1, out2=out2, out=out, where=where)


class TensorMultiOp(TensorElementWiseWithInputs, TensorOperator):
    args = ListField("args", default=None)
    out = KeyField("out", default=None)
    where = KeyField("where", default=None)
    casting = StringField("casting", default=None)
    order = StringField("order", default=None)
    err = DictField("err", FieldTypes.string, FieldTypes.string, default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.casting is None:
            self.casting = "same_kind"
        if self.order is None:
            self.order = "K"
        check_order(self.order)

    @classmethod
    def _is_sparse(cls, *args):
        return False

    def _set_sparse(self, inputs):
        inputs_iter = iter(inputs or ())
        args = list(self.args)
        for idx in range(len(self.args)):
            if not np.isscalar(self.args[idx]):
                args[idx] = next(inputs_iter)
        setattr(self, "sparse", self._is_sparse(*args))

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(inputs or ())

        args = list(self.args)
        for idx in range(len(args)):
            if not np.isscalar(args[idx]):
                args[idx] = next(inputs_iter)
        self.args = args

        if getattr(self, "out", None) is not None:
            self.out = next(inputs_iter)
        if getattr(self, "where", None) is not None:
            self.where = next(inputs_iter)

    def _process_inputs(self, *args, out=None):
        self.args = [a if np.isscalar(a) else astensor(a) for a in args]

        if out is not None:
            if isinstance(out, Tensor):
                self.out = out
            else:
                raise TypeError(f"out should be Tensor object, got {type(out)} instead")

        return args + (out,)

    def __call__(self, *args, out=None):
        proc_inputs_results = self._process_inputs(*args, out=out)
        args = proc_inputs_results[:-1]
        (out,) = proc_inputs_results[-1:]
        # check broadcast
        shapes = [() if np.isscalar(a) else a.shape for a in self.args]
        shape = broadcast_shape(*shapes)
        order = out.order if out is not None else None

        inputs = filter_inputs(list(args) + [out])
        t = self.new_tensor(inputs, shape, order=order)

        if out is None:
            return t

        check_out_param(out, t, self.casting)
        out_shape, out_dtype = out.shape, out.dtype

        # if `out` is specified, use out's dtype and shape
        if t.shape != out_shape:
            t = self.new_tensor(inputs, out_shape, order=order)
        setattr(self, "dtype", out_dtype)

        out.data = t.data
        return out
