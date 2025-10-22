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

import abc
import base64
import dataclasses
import logging
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

from odps.types import OdpsSchema
from odps.utils import camel_to_underline

from ..core import OperatorType, Tileable, TileableGraph, enter_mode
from ..core.operator import Fetch, Operator
from ..extension import iter_extensions
from ..io.odpsio import build_dataframe_table_meta
from ..io.odpsio.schema import pandas_to_odps_schema
from ..lib import wrapped_pickle as pickle
from ..protocol import DataFrameTableMeta, ResultInfo
from ..serialization import PickleContainer
from ..serialization.serializables import Serializable, StringField
from ..typing_ import PandasObjectTypes
from ..udf import MarkedFunction, PythonPackOptions

if TYPE_CHECKING:
    from odpsctx import ODPSSessionContext

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CodeGenResult:
    code: str
    input_key_to_variables: Dict[str, str]
    output_key_to_variables: Dict[str, str]
    output_key_to_result_infos: Dict[str, ResultInfo]
    constants: Dict[str, Any]


class AbstractUDF(Serializable):
    _session_id: str = StringField("session_id")

    def __init__(self, session_id: Optional[str] = None, **kw):
        super().__init__(_session_id=session_id, **kw)

    @property
    def name(self) -> str:
        return camel_to_underline(type(self).__name__)

    @property
    def session_id(self):
        return getattr(self, "_session_id", None)

    @session_id.setter
    def session_id(self, value: str):
        self._session_id = value

    @abc.abstractmethod
    def register(self, odps: "ODPSSessionContext", overwrite: bool = False):
        raise NotImplementedError

    @abc.abstractmethod
    def unregister(self, odps: "ODPSSessionContext"):
        raise NotImplementedError

    @abc.abstractmethod
    def collect_pythonpack(self) -> List[PythonPackOptions]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_pythonpack_resources(self, odps_ctx: "ODPSSessionContext") -> None:
        raise NotImplementedError


class UserCodeMixin:
    __slots__ = ()

    @classmethod
    def obj_to_python_expr(cls, obj: Any = None) -> str:
        """
        Parameters
        ----------
        obj
            The object to convert to python expr.
        Returns
        -------
        str :
            The str type content equals to the object when use in the python code directly.
        """
        if obj is None:
            return "None"

        if isinstance(obj, (int, float)):
            return repr(obj)

        if isinstance(obj, bool):
            return "True" if obj else "False"

        if isinstance(obj, bytes):
            base64_bytes = base64.b64encode(obj)
            return f"base64.b64decode({base64_bytes})"

        if isinstance(obj, str):
            return repr(obj)

        if isinstance(obj, list):
            return (
                f"[{', '.join([cls.obj_to_python_expr(element) for element in obj])}]"
            )

        if isinstance(obj, dict):
            items = (
                f"{repr(key)}: {cls.obj_to_python_expr(value)}"
                for key, value in obj.items()
            )
            return f"{{{', '.join(items)}}}"

        if isinstance(obj, tuple):
            obj_exprs = [cls.obj_to_python_expr(sub_obj) for sub_obj in obj]
            return f"({', '.join(obj_exprs)}{',' if len(obj) == 1 else ''})"

        if isinstance(obj, set):
            return (
                f"{{{', '.join([cls.obj_to_python_expr(sub_obj) for sub_obj in obj])}}}"
                if obj
                else "set()"
            )

        if isinstance(obj, PickleContainer):
            return UserCodeMixin.generate_pickled_codes(obj, None)

        raise ValueError(f"not support arg type {type(obj)}")

    @classmethod
    def generate_pickled_codes(
        cls,
        code_to_pickle: Any,
        main_entry_var_name: Union[str, None] = "udf_main_entry",
    ) -> str:
        """
        Generate pickled codes. The final pickled variable is called 'udf_main_entry'.

        Parameters
        ----------
        code_to_pickle: Any
            The code to be pickled.
        main_entry_var_name: str
            The variables in code used to hold the loads object from the cloudpickle

        Returns
        -------
        str :
            The code snippets of pickling, the final variable is called
            'udf_main_entry' by default.
        """
        pickled, buffers = cls.dump_pickled_data(code_to_pickle)
        pickle_loads_expr = (
            f"cloudpickle.loads({cls.obj_to_python_expr(pickled)}, "
            f"buffers={cls.obj_to_python_expr(buffers)})"
        )
        if main_entry_var_name:
            return f"{main_entry_var_name} = {pickle_loads_expr}"

        return pickle_loads_expr

    @staticmethod
    def dump_pickled_data(
        code_to_pickle: Any,
    ) -> Tuple[List[bytes], List[bytes]]:
        if isinstance(code_to_pickle, MarkedFunction):
            code_to_pickle = code_to_pickle.func
        if isinstance(code_to_pickle, PickleContainer):
            buffers = code_to_pickle.get_buffers()
            pickled = buffers[0]
            buffers = buffers[1:]
        else:
            pickled = pickle.dumps(code_to_pickle, protocol=pickle.DEFAULT_PROTOCOL)
            buffers = []
        return pickled, buffers


class DAGCodeContext(metaclass=abc.ABCMeta):
    def __init__(self, session_id: str = None, subdag_id: str = None):
        self._session_id = session_id
        self._subdag_id = subdag_id
        self._tileable_key_to_variables = dict()
        self.constants = dict()
        self._data_table_meta_cache = dict()
        self._odps_schema_cache = dict()
        self._udfs = dict()
        self._tileable_key_to_result_infos = dict()
        self._next_var_id = 0
        self._next_const_id = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def subdag_id(self) -> str:
        return self._subdag_id

    def register_udf(self, udf: AbstractUDF):
        udf.session_id = self._session_id
        self._udfs[udf.name] = udf

    def get_udfs(self) -> List[AbstractUDF]:
        return list(self._udfs.values())

    def get_input_tileable_variable(self, tileable: Tileable) -> str:
        """
        Get or create the variable name for an input tileable. It should be used on the
        RIGHT side of the assignment.
        """
        return self._get_tileable_variable(tileable)

    def get_output_tileable_variable(self, tileable: Tileable) -> str:
        """
        Get or create the variable name for an output tileable. It should be used on the
        LEFT side of the assignment.
        """
        return self._get_tileable_variable(tileable)

    def _get_tileable_variable(self, tileable: Tileable) -> str:
        try:
            return self._tileable_key_to_variables[tileable.key]
        except KeyError:
            var_name = self.next_var_name()
            self._tileable_key_to_variables[tileable.key] = var_name
            return var_name

    def next_var_name(self) -> str:
        var_name = f"var_{self._next_var_id}"
        self._next_var_id += 1
        return var_name

    def get_odps_schema(
        self, data: PandasObjectTypes, unknown_as_string: bool = False
    ) -> OdpsSchema:
        """
        Get the corresponding ODPS schema of the input df_obj.

        Parameters
        ----------
        data :
            The pandas data object.
        unknown_as_string :
            Whether mapping the unknown data type to a temp string value.

        Returns
        -------
        OdpsSchema :
            The OdpsSchema of df_obj.
        """
        if data.key not in self._odps_schema_cache:
            odps_schema, table_meta = pandas_to_odps_schema(data, unknown_as_string)
            self._data_table_meta_cache[data.key] = table_meta
            self._odps_schema_cache[data.key] = odps_schema
        return self._odps_schema_cache[data.key]

    def get_pandas_data_table_meta(self, data: PandasObjectTypes) -> DataFrameTableMeta:
        if data.key not in self._data_table_meta_cache:
            self._data_table_meta_cache[data.key] = build_dataframe_table_meta(data)
        return self._data_table_meta_cache[data.key]

    def register_operator_constants(self, const_val, var_name: str = None) -> str:
        if var_name is None:
            if (
                isinstance(const_val, (int, str, bytes, bool, float))
                or const_val is None
            ):
                return repr(const_val)
            var_name = f"const_{self._next_const_id}"
            self._next_const_id += 1

        self.constants[var_name] = const_val
        return var_name

    def put_tileable_result_info(
        self, tileable: Tileable, result_info: ResultInfo
    ) -> None:
        self._tileable_key_to_result_infos[tileable.key] = result_info

    def get_tileable_result_infos(self) -> Dict[str, ResultInfo]:
        return self._tileable_key_to_result_infos


class EngineAcceptance(Enum):
    """
    DENY: The operator is not accepted by the current engine.
    ACCEPT: The operator is accepted by the current engine, and doesn't break from here.
    BREAK_BEFORE: The operator is accepted by the current engine, but should break before
        its execution.
    BREAK_AFTER: The operator is accepted by the current engine, but should break after
        its execution.
    PREDECESSOR: The acceptance of the operator is decided by engines of its
        predecessors. If acceptance of all predecessors are SUCCESSOR, the acceptance
        of current operator is SUCCESSOR. Otherwise the engine selected in predecessors
        with highest priority is used.
    SUCCESSOR: The acceptance of the operator is decided by engines of its successors.
        If the operator has no successors, the acceptance will be treated as ACCEPT.
        Otherwise the engine selected in successors with highest priority is used.
    """

    DENY = 0
    ACCEPT = 1
    BREAK_AFTER = 2
    PREDECESSOR = 3
    SUCCESSOR = 4
    BREAK_BEFORE = 5

    @classmethod
    def _missing_(cls, pred: bool) -> "EngineAcceptance":
        """
        A convenience method to get ACCEPT or DENY result via the input predicate.

        Parameters
        ----------
        pred : bool
            The predicate variable.

        Returns
        -------
        EngineAcceptance :
            Returns ACCEPT if the predicate is true, otherwise returns DENY.
        """
        return cls.ACCEPT if pred else cls.DENY


class DagOperatorAdapter(metaclass=abc.ABCMeta):
    # todo handle refcount issue when generated code is being executed
    def accepts(self, op: Operator) -> EngineAcceptance:
        return EngineAcceptance.ACCEPT

    @abc.abstractmethod
    def generate_code(self, op: Operator, context: DAGCodeContext) -> List[str]:
        raise NotImplementedError

    def generate_comment(self, op: Operator, context: DAGCodeContext) -> List[str]:
        """
        Generate the comment codes before actual ones.

        Parameters
        ----------
        op : Operator
            The operator instance.
        context : DAGCodeContext
            The DagCodeContext instance.

        Returns
        -------
        result: List[str]
            The comment codes, one per line.
        """
        return list()

    def generate_pre_op_code(self, op: Operator, context: DAGCodeContext) -> List[str]:
        """
        Generate the codes before actually handling the operator.
        This method is usually implemented in the base class of each engine.

        Parameters
        ----------
        op : Operator
            The operator instance.
        context : DAGCodeContext
            The DagCodeContext instance.

        Returns
        -------
        result: List[str]
            The codes generated before one operator actually handled, one per line.
        """
        return list()

    def generate_post_op_code(self, op: Operator, context: DAGCodeContext) -> List[str]:
        """
        Generate the codes after actually handling the operator.
        This method is usually implemented in the base class of each engine.

        Parameters
        ----------
        op : Operator
            The operator instance.
        context : DAGCodeContext
            The DagCodeContext instance.

        Returns
        -------
        result: List[str]
            The codes generated after one operator actually handled, one per line.
        """
        return list()


_engine_to_codegen: Dict[str, Type["DAGCodeGenerator"]] = dict()


def register_engine_codegen(type_: Type["DAGCodeGenerator"]):
    _engine_to_codegen[type_.engine_type] = type_
    return type_


BUILTIN_ENGINE_DPE = "DPE"
BUILTIN_ENGINE_SPE = "SPE"
BUILTIN_ENGINE_MCSQL = "MCSQL"


class DAGCodeGenerator(metaclass=abc.ABCMeta):
    _context: DAGCodeContext

    engine_type: Optional[str] = None
    engine_priority: int = 0
    _extension_loaded = False
    _generate_comments_enabled: bool = True

    def __init__(self, session_id: str, subdag_id: str = None):
        self._session_id = session_id
        self._subdag_id = subdag_id
        self._context = self._init_context(session_id, subdag_id)
        self._generate_comments_enabled = True

    @classmethod
    def _load_engine_extensions(cls):
        if cls._extension_loaded:
            return
        for name, ep in iter_extensions():
            _engine_to_codegen[name.upper()] = ep.get_codegen()
        cls._extension_loaded = True

    @classmethod
    def get_engine_types(cls) -> List[str]:
        cls._load_engine_extensions()
        engines = sorted(
            _engine_to_codegen.values(), key=lambda x: x.engine_priority, reverse=True
        )
        return [e.engine_type for e in engines]

    @classmethod
    def get_by_engine_type(cls, engine_type: str) -> Type["DAGCodeGenerator"]:
        cls._load_engine_extensions()
        return _engine_to_codegen[engine_type]

    @abc.abstractmethod
    def get_op_adapter(self, op_type: Type[OperatorType]) -> Type[DagOperatorAdapter]:
        raise NotImplementedError

    @abc.abstractmethod
    def _init_context(self, session_id: str, subdag_id: str) -> DAGCodeContext:
        raise NotImplementedError

    def _generate_delete_code(self, var_name: str) -> List[str]:
        return []

    @enter_mode(build=True)
    def generate_code(self, dag: TileableGraph) -> List[str]:
        """
        Generate the code of the input dag.

        Parameters
        ----------
        dag : TileableGraph
            The input DAG instance.

        Returns
        -------
        List[str] :
            The code lines.
        """
        code_lines = []
        visited_op_key = set()
        result_key_set = set(t.key for t in dag.result_tileables)
        out_refcounts = dict()
        for tileable in dag.topological_iter():
            op: OperatorType = tileable.op
            if op.key in visited_op_key or isinstance(op, Fetch):
                continue

            visited_op_key.add(op.key)

            adapter = self.get_op_adapter(type(op))()
            code_lines.extend(adapter.generate_pre_op_code(op, self._context))
            if self._generate_comments_enabled:
                code_lines.extend(adapter.generate_comment(op, self._context))
            code_lines.extend(adapter.generate_code(op, self._context) or [])
            code_lines.extend(adapter.generate_post_op_code(op, self._context))
            code_lines.append("")  # Append an empty line to separate operators

            # record refcounts
            for out_t in op.outputs:
                if out_t.key in result_key_set:
                    continue
                if dag.count_successors(out_t) == 0:
                    delete_code = self._generate_delete_code(
                        self._context.get_input_tileable_variable(out_t)
                    )
                    code_lines.extend(delete_code)
                else:
                    out_refcounts[out_t.key] = dag.count_successors(out_t)

            # check if refs of inputs are no longer needed
            for inp_t in op.inputs:
                if inp_t.key not in out_refcounts:
                    continue
                out_refcounts[inp_t.key] -= 1
                if out_refcounts[inp_t.key] == 0:
                    delete_code = self._generate_delete_code(
                        self._context.get_input_tileable_variable(inp_t)
                    )
                    code_lines.extend(delete_code)
                    out_refcounts.pop(inp_t.key)

        return code_lines

    def generate(self, dag: TileableGraph) -> CodeGenResult:
        code_lines = self.generate_code(dag)
        input_key_to_vars = dict()
        for tileable in dag.topological_iter():
            op: OperatorType = tileable.op
            if isinstance(op, Fetch):
                fetch_tileable = self._context.get_input_tileable_variable(tileable)
                input_key_to_vars[op.outputs[0].key] = fetch_tileable

        result_variables = {
            t.key: self._context.get_input_tileable_variable(t) for t in dag.results
        }

        return CodeGenResult(
            code="\n".join(code_lines),
            input_key_to_variables=input_key_to_vars,
            output_key_to_variables=result_variables,
            constants=self._context.constants,
            output_key_to_result_infos=self._context.get_tileable_result_infos(),
        )

    def run_pythonpacks(
        self,
        odps_ctx: "ODPSSessionContext",
        python_tag: str,
        is_production: bool = False,
        schedule_id: Optional[str] = None,
        hints: Optional[dict] = None,
        priority: Optional[int] = None,
    ) -> Dict[str, PythonPackOptions]:
        key_to_packs = defaultdict(list)
        for udf in self._context.get_udfs():
            for pack in udf.collect_pythonpack():
                key_to_packs[pack.key].append(pack)
        distinct_packs = []
        for packs in key_to_packs.values():
            distinct_packs.append(packs[0])

        inst_id_to_req = {}
        for pack in distinct_packs:
            inst = odps_ctx.run_pythonpack(
                requirements=pack.requirements,
                prefer_binary=pack.prefer_binary,
                pre_release=pack.pre_release,
                force_rebuild=pack.force_rebuild,
                no_audit_wheel=pack.no_audit_wheel,
                python_tag=python_tag,
                is_production=is_production,
                schedule_id=schedule_id,
                hints=hints,
                priority=priority,
            )
            # fulfill instance id of pythonpacks with same keys
            for same_pack in key_to_packs[pack.key]:
                same_pack.pack_instance_id = inst.id
            inst_id_to_req[inst.id] = pack
        return inst_id_to_req

    def register_udfs(self, odps_ctx: "ODPSSessionContext"):
        for udf in self._context.get_udfs():
            logger.info("[Session=%s] Registering UDF %s", self._session_id, udf.name)
            udf.register(odps_ctx, True)

    def unregister_udfs(self, odps_ctx: "ODPSSessionContext"):
        for udf in self._context.get_udfs():
            logger.info("[Session=%s] Unregistering UDF %s", self._session_id, udf.name)
            udf.unregister(odps_ctx)

    def get_udfs(self) -> List[AbstractUDF]:
        return self._context.get_udfs()
