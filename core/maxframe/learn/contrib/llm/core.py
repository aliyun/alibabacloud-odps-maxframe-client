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
from typing import Any, Dict

import numpy as np
import pandas as pd

from ....core.entity.output_types import OutputType
from ....core.operator.base import Operator
from ....core.operator.core import TileableOperatorMixin
from ....dataframe.utils import parse_index
from ....serialization.serializables.core import Serializable
from ....serialization.serializables.field import AnyField, DictField, StringField


class LLM(Serializable):
    name = StringField("name", default=None)

    def validate_params(self, params: Dict[str, Any]):
        pass


class LLMOperator(Operator, TileableOperatorMixin):
    model = AnyField("model", default=None)
    prompt_template = AnyField("prompt_template", default=None)
    params = DictField("params", default=None)

    def __init__(self, output_types=None, **kw):
        if output_types is None:
            output_types = [OutputType.dataframe]
        super().__init__(_output_types=output_types, **kw)

    def __call__(self, data):
        col_names = ["response", "success"]
        columns = parse_index(pd.Index(col_names), store_data=True)
        out_dtypes = pd.Series([np.dtype("O"), np.dtype("bool")], index=col_names)
        return self.new_tileable(
            inputs=[data],
            dtypes=out_dtypes,
            shape=(data.shape[0], len(col_names)),
            index_value=data.index_value,
            columns_value=columns,
        )
