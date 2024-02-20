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

import abc
import base64
import dataclasses
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

from odps.types import OdpsSchema
from odps.utils import camel_to_underline

from .core import OperatorType, Tileable, TileableGraph
from .core.operator import Fetch
from .extension import iter_extensions
from .lib import wrapped_pickle as pickle
from .odpsio import build_dataframe_table_meta
from .odpsio.schema import pandas_to_odps_schema
from .protocol import DataFrameTableMeta, ResultInfo
from .serialization import PickleContainer
from .typing_ import PandasObjectTypes
from .udf import MarkedFunction

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


class AbstractUDF(abc.ABC):
    _session_id: str

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


class UserCodeMixin:
    @classmethod
    def generate_pickled_codes(cls, code_to_pickle: Any) -> List[str]:
        """
        Generate pickled codes. The final pickled variable is called 'pickled_data'.

        Parameters
        ----------
        code_to_pickle: Any
            The code to be pickled.

        Returns
        -------
        List[str] :
            The code snippets of pickling, the final variable is called 'pickled_data'.
        """
        pickled, buffers = cls.dump_pickled_data(code_to_pickle)
        pickled = base64.b64encode(pickled)
        buffers = [base64.b64encode(b) for b in buffers]
        buffers_str = ", ".join(f"base64.b64decode(b'{b.decode()}')" for b in buffers)
        return [
            f"base64_data = base64.b64decode(b'{pickled.decode()}')",
            f"pickled_data = cloudpickle.loads(base64_data, buffers=[{buffers_str}])",
        ]

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


class BigDagCodeContext(metaclass=abc.ABCMeta):
    def __init__(self, session_id: str = None):
        self._session_id = session_id
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

    def register_udf(self, udf: AbstractUDF):
        udf.session_id = self._session_id
        self._udfs[udf.name] = udf

    def get_udfs(self) -> List[AbstractUDF]:
        return list(self._udfs.values())

    def get_tileable_variable(self, tileable: Tileable) -> str:
        try:
            return self._tileable_key_to_variables[tileable.key]
        except KeyError:
            var_name = self._tileable_key_to_variables[
                tileable.key
            ] = f"var_{self._next_var_id}"
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
    BREAK: The operator is accepted by the current engine, but should break from here.
    """

    DENY = 0
    ACCEPT = 1
    BREAK = 2

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


class BigDagOperatorAdapter(metaclass=abc.ABCMeta):
    # todo handle refcount issue when generated code is being executed
    def accepts(self, op: OperatorType) -> EngineAcceptance:
        return EngineAcceptance.ACCEPT

    @abc.abstractmethod
    def generate_code(self, op: OperatorType, context: BigDagCodeContext) -> List[str]:
        raise NotImplementedError

    def generate_comment(
        self, op: OperatorType, context: BigDagCodeContext
    ) -> List[str]:
        """
        Generate the comment codes before actual ones.

        Parameters
        ----------
        op : OperatorType
            The operator instance.
        context : BigDagCodeContext
            The BigDagCodeContext instance.

        Returns
        -------
        result: List[str]
            The comment codes, one per line.
        """
        return list()


_engine_to_codegen: Dict[str, Type["BigDagCodeGenerator"]] = dict()


def register_engine_codegen(type_: Type["BigDagCodeGenerator"]):
    _engine_to_codegen[type_.engine_type] = type_
    return type_


BUILTIN_ENGINE_SPE = "SPE"
BUILTIN_ENGINE_MCSQL = "MCSQL"


class BigDagCodeGenerator(metaclass=abc.ABCMeta):
    _context: BigDagCodeContext

    engine_type: Optional[str] = None
    engine_priority: int = 0
    _extension_loaded = False

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._context = self._init_context(session_id)

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
    def get_by_engine_type(cls, engine_type: str) -> Type["BigDagCodeGenerator"]:
        cls._load_engine_extensions()
        return _engine_to_codegen[engine_type]

    @abc.abstractmethod
    def get_op_adapter(
        self, op_type: Type[OperatorType]
    ) -> Type[BigDagOperatorAdapter]:
        raise NotImplementedError

    @abc.abstractmethod
    def _init_context(self, session_id: str) -> BigDagCodeContext:
        raise NotImplementedError

    def _generate_comments(
        self, op: OperatorType, adapter: BigDagOperatorAdapter
    ) -> List[str]:
        return adapter.generate_comment(op, self._context)

    def _generate_pre_op_code(self, op: OperatorType) -> List[str]:
        return []

    def _generate_delete_code(self, var_name: str) -> List[str]:
        return []

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
            code_lines.extend(self._generate_pre_op_code(op))
            code_lines.extend(self._generate_comments(op, adapter))
            code_lines.extend(adapter.generate_code(op, self._context))
            code_lines.append("")  # Append an empty line to separate operators

            # record refcounts
            for out_t in op.outputs:
                if out_t.key in result_key_set:
                    continue
                if dag.count_successors(out_t) == 0:
                    delete_code = self._generate_delete_code(
                        self._context.get_tileable_variable(out_t)
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
                        self._context.get_tileable_variable(inp_t)
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
                input_key_to_vars[
                    op.outputs[0].key
                ] = self._context.get_tileable_variable(tileable)

        result_variables = {
            t.key: self._context.get_tileable_variable(t) for t in dag.results
        }

        return CodeGenResult(
            code="\n".join(code_lines),
            input_key_to_variables=input_key_to_vars,
            output_key_to_variables=result_variables,
            constants=self._context.constants,
            output_key_to_result_infos=self._context.get_tileable_result_infos(),
        )

    def register_udfs(self, odps_ctx: "ODPSSessionContext"):
        for udf in self._context.get_udfs():
            logger.info("[Session %s] Registering UDF %s", self._session_id, udf.name)
            udf.register(odps_ctx, True)

    def unregister_udfs(self, odps_ctx: "ODPSSessionContext"):
        for udf in self._context.get_udfs():
            logger.info("[Session %s] Unregistering UDF %s", self._session_id, udf.name)
            udf.unregister(odps_ctx)

    def get_udfs(self) -> List[AbstractUDF]:
        return self._context.get_udfs()
