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

from typing import Any, Dict

import numpy as np
import pandas as pd

from ....core.entity.output_types import OutputType
from ....core.operator.base import Operator
from ....core.operator.core import TileableOperatorMixin
from ....dataframe.operators import DataFrameOperatorMixin
from ....dataframe.utils import parse_index
from ....serialization.serializables import Int32Field
from ....serialization.serializables.core import Serializable
from ....serialization.serializables.field import (
    AnyField,
    BoolField,
    DictField,
    StringField,
)


class LLM(Serializable):
    name = StringField("name", default=None)

    def validate_params(self, params: Dict[str, Any]):
        pass


class LLMTaskOperator(Operator, DataFrameOperatorMixin):
    task = AnyField("task", default=None)
    model = AnyField("model", default=None)
    params = DictField("params", default=None)
    running_options: Dict[str, Any] = DictField("running_options", default=None)
    timeout = Int32Field("timeout", default=None)

    def __init__(self, output_types=None, **kw):
        if output_types is None:
            output_types = [OutputType.dataframe]
        super().__init__(_output_types=output_types, **kw)

    def get_output_dtypes(self) -> Dict[str, np.dtype]:
        raise NotImplementedError

    def __call__(self, data, index=None):
        outputs = self.get_output_dtypes()
        col_name = list(outputs.keys())
        columns = parse_index(pd.Index(col_name), store_data=True)
        out_dtypes = pd.Series(list(outputs.values()), index=col_name)
        index_value = index or data.index_value

        return self.new_dataframe(
            inputs=[data],
            shape=(np.nan, len(col_name)),
            index_value=index_value,
            columns_value=columns,
            dtypes=out_dtypes,
        )


class LLMTextGenOperator(LLMTaskOperator, TileableOperatorMixin):
    prompt_template = AnyField("prompt_template", default=None)

    def get_output_dtypes(self) -> Dict[str, np.dtype]:
        return {"response": np.dtype("O"), "success": np.dtype("bool")}


class LLMTextEmbeddingOp(LLMTaskOperator, TileableOperatorMixin):
    dimensions = Int32Field("dimensions", default=None)
    encoding_format = StringField("encoding_format", default=None)
    simple_output = BoolField("simple_output", default=False)

    def get_output_dtypes(self) -> Dict[str, np.dtype]:
        return {"response": np.dtype("O"), "success": np.dtype("bool")}
