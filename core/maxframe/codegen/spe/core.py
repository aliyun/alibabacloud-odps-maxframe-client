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

import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

import pandas as pd
from numpy import dtype as npdtype
from numpy import ndarray
from pandas.tseries.offsets import BaseOffset

from ...core import TILEABLE_TYPE, OperatorType, TileableGraph
from ...core.operator.base import Operator
from ...serialization import PickleContainer
from ...utils import TypeDispatcher, no_default
from ..core import (
    BUILTIN_ENGINE_SPE,
    DAGCodeContext,
    DAGCodeGenerator,
    DagOperatorAdapter,
    register_engine_codegen,
)
from .dataframe.udf import SpeUDF

_spe_op_adapter = TypeDispatcher()


def register_op_adapter(op_cls: Union[Type[OperatorType], List[Type[OperatorType]]]):
    def wrapper(cls: Type[DagOperatorAdapter]):
        op_classes = op_cls
        if not isinstance(op_cls, list):
            op_classes = [op_cls]
        for op_class in op_classes:
            _spe_op_adapter.register(op_class, cls)
        return cls

    return wrapper


class SPECodeContext(DAGCodeContext, ABC):
    logger_var: str = "logger"

    def __init__(self, session_id: str = None, subdag_id: str = None):
        super().__init__(session_id, subdag_id)
        self.imports = defaultdict(set)
        self.import_froms = defaultdict(set)

    def register_import(
        self, mod: str, alias: Optional[str] = None, from_item: Optional[str] = None
    ):
        if from_item is None:
            self.imports[mod].add(alias)
        else:
            self.import_froms[mod].add((from_item, alias))


class SPEOperatorAdapter(DagOperatorAdapter, ABC):
    @abstractmethod
    def generate_code(self, op: OperatorType, context: SPECodeContext) -> List[str]:
        raise NotImplementedError

    @classmethod
    def generate_call_args_with_attributes(
        cls,
        op: OperatorType,
        context: SPECodeContext,
        *args: str,
        skip_none: bool = False,
        kw_keys: Optional[Iterable[str]] = None,
        **kwargs: Optional[str],
    ) -> str:
        """
        Generate the codes with simple copying attributes from the input operator to the
        output Python call parameters. The absence of the attribute value will be
        fallback to None.

        Parameters
        ----------
        op: OperatorType
            The operator instance.
        context: SPECodeContext
            The SPECodeContext instance.
        *args:
            The attribute names, which will remain the same name in the call parameter.
        skip_none: bool
            If True, will skip attributes with None value.
        **kwargs:
            The keyword arguments with string values, which will map to the name of the
            call parameter.

            e.g. old_name="new_name" will be __call__(new_name=getattr(obj, old_name))

        Returns
        -------
        str :
            The call args codes.
        """
        kw = {key: kwargs.get(key) for key in kw_keys or ()}
        kw.update(kwargs)
        str_args = [a for a in args if isinstance(a, str)]
        all_args_dict = cls._collect_op_kwargs(
            op, str_args + list(kw.keys()), skip_none=skip_none
        )
        args_list = [
            all_args_dict.get(k) if isinstance(k, str) else op.inputs[k] for k in args
        ]
        kwargs_dict = {
            kw.get(k) or k: all_args_dict.get(k)
            for k in kw.keys()
            if k in all_args_dict
        }
        return ", ".join(cls._translate_call_args(context, *args_list, **kwargs_dict))

    @classmethod
    def _collect_op_kwargs(
        cls, op: OperatorType, args: List[str], skip_none: bool = True
    ) -> Dict[str, str]:
        kw = {}
        for arg in args:
            attr_val = getattr(op, arg, None)
            if attr_val is not no_default and (not skip_none or attr_val is not None):
                kw[arg] = attr_val
        return kw

    @classmethod
    def _translate_call_args(
        cls, context: SPECodeContext, *args, **kwargs
    ) -> List[str]:
        return [cls.translate_var(context, v) for v in args] + [
            f"{k}={cls.translate_var(context, v)}" for k, v in kwargs.items()
        ]

    @classmethod
    def translate_var(cls, context: SPECodeContext, val: Any) -> str:
        """
        Translate the val to a Python variable.

        Parameters
        ----------
        context: SPECodeContext
            The SPECodeContext instance.
        val : Any
            The value to be translated.

        Returns
        -------
        str :
            The var name which can be used in generated Python code.
        """
        if isinstance(val, TILEABLE_TYPE):
            return context.get_input_tileable_variable(val)

        if isinstance(val, (Callable, PickleContainer)) and not isinstance(
            val, BaseOffset
        ):
            # TODO: handle used resources here
            context.register_import("base64")
            context.register_import("cloudpickle")
            context.register_import("numpy", "np")
            udf = SpeUDF(val)
            context.register_udf(udf)
            return udf.name

        val_type = type(val)
        if val_type is list:
            vals_exp = [cls.translate_var(context, v) for v in val]
            return f"[{', '.join(vals_exp)}]"

        if val_type is tuple:
            vals_exp = [cls.translate_var(context, v) for v in val]
            if len(vals_exp) == 1:
                # need a trailing space to make a tuple
                vals_exp.append("")
            return f"({', '.join(vals_exp).rstrip()})"

        if val_type is slice:
            return f"slice({val.start}, {val.stop}, {val.step})"

        if val_type is ndarray:
            return cls.translate_var(context, val.tolist())

        if val_type is dict:
            kvs = list()
            for k, v in val.items():
                kvs.append(
                    f"{cls.translate_var(context, k)}: {cls.translate_var(context, v)}"
                )
            return f"{{{', '.join(kvs)}}}"

        if val_type is set:
            keys = [cls.translate_var(context, k) for k in val]
            return f"{{{', '.join(keys)}}}"

        if val_type is pd.Timestamp:
            context.register_import("pandas", "pd")
            return f"pd.Timestamp({str(val)!r})"

        if isinstance(val, npdtype):
            context.register_import("numpy", "np")
            return f"np.dtype({str(val)!r})"

        return context.register_operator_constants(val)

    def generate_pre_op_code(self, op: Operator, context: SPECodeContext) -> List[str]:
        context.register_operator_constants(True, "running")
        return ["if not running:\n    raise RuntimeError('CANCELLED')"]

    def gen_logging_code(
        self, context: SPECodeContext, message: str, expressions: List[str]
    ) -> str:
        exp_str = ", ".join(expressions)
        return f"{context.logger_var}.info('{message}', {exp_str})"

    def gen_timecost_code(
        self, context: SPECodeContext, phase_name: str, code_lines: List[str]
    ) -> List[str]:
        context.register_import("time")
        codes = ["start_time = time.time()"]
        codes.extend(code_lines)
        codes.append(
            f"{context.logger_var}.info('{phase_name} cost: %.2f s', time.time() - start_time)"
        )
        return codes


@register_engine_codegen
class SPECodeGenerator(DAGCodeGenerator):
    _context: SPECodeContext

    engine_type = BUILTIN_ENGINE_SPE
    engine_priority = 0

    def _init_context(self, session_id: str, subdag_id: str) -> SPECodeContext:
        return SPECodeContext(session_id, subdag_id)

    def _generate_delete_code(self, var_name: str) -> List[str]:
        return [f"del {var_name}"]

    @staticmethod
    def _generate_import_code(module: str, alias: Optional[str] = None):
        if alias is None:
            return f"import {module}"
        else:
            return f"import {module} as {alias}"

    @staticmethod
    def _generate_import_from_code(
        module: str, from_list: List[str], alias_list: Optional[List[str]] = None
    ):
        def build_from_str(from_str: str, alias: Optional[str]) -> str:
            if alias is not None:
                return f"{from_str} as {alias}"
            else:
                return from_str

        alias_list = alias_list or itertools.repeat(None)
        froms_str = ", ".join(
            build_from_str(f, a) for f, a in zip(from_list, alias_list)
        )
        return f"from {module} import {froms_str}"

    def get_op_adapter(self, op_type: Type[OperatorType]) -> Type[DagOperatorAdapter]:
        return get_op_adapter(op_type)

    def generate_code(self, dag: TileableGraph) -> List[str]:
        from . import dataframe, tensor

        del dataframe, tensor
        main_codes = super().generate_code(dag)
        import_codes = []
        for mod, aliases in self._context.imports.items():
            for alias in aliases:
                import_codes.append(self._generate_import_code(mod, alias))
        for mod, from_tuples in self._context.import_froms.items():
            from_mods, aliases = [], []
            for from_item, from_alias in from_tuples:
                from_mods.append(from_item)
                aliases.append(from_alias)
            import_codes.append(
                self._generate_import_from_code(mod, from_mods, aliases)
            )
        udf_codes = self._generate_udf_codes()
        return import_codes + udf_codes + main_codes

    def _generate_udf_codes(self) -> List[str]:
        udf_codes = list()
        for func in self.get_udfs():
            udf_codes.extend(func.encoded_content)
            udf_codes.append(f"{func.name} = udf_main_entry")
        return udf_codes


def get_op_adapter(op_type: Type[OperatorType]) -> Type[SPEOperatorAdapter]:
    return _spe_op_adapter.get_handler(op_type)
