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
import pandas as pd
import pyarrow as pa

from ... import opcodes
from ...core import ENTITY_TYPE, OutputType
from ...io.odpsio.schema import (
    pandas_dtype_to_arrow_type,
    pandas_dtypes_to_arrow_schema,
)
from ...lib.dtypes_extension import ArrowDtype
from ...serialization.serializables import BoolField
from ...tensor.core import TensorOrder
from ...utils import lazy_import
from ..core import DATAFRAME_TYPE
from ..initializer import Series as asseries
from .core import (
    CustomReduction,
    DataFrameReduction,
    DataFrameReductionMixin,
    ReductionCallable,
)

cudf = lazy_import("cudf")


class UniqueReduction(CustomReduction):
    _func_name = "unique"

    def agg(self, data):  # noqa: W0221  # pylint: disable=arguments-differ
        xdf = cudf if self.is_gpu() else pd
        # convert to series data
        return xdf.Series(data.unique())

    def post(self, data):  # noqa: W0221  # pylint: disable=arguments-differ
        return data.unique()


class UniqueReductionCallable(ReductionCallable):
    def __call__(self, value):
        return UniqueReduction(name="unique", is_gpu=self.kwargs["is_gpu"])(value)


class DataFrameUnique(DataFrameReduction, DataFrameReductionMixin):
    _op_type_ = opcodes.UNIQUE
    _func_name = "unique"

    output_list_scalar = BoolField("output_list_scalar", default=False)

    @property
    def is_atomic(self):
        return True

    def get_reduction_args(self, axis=None):
        return {}

    @classmethod
    def get_reduction_callable(cls, op):
        return UniqueReductionCallable(
            func_name=cls._func_name, kwargs=dict(is_gpu=op.is_gpu())
        )

    def __call__(self, a):
        if not isinstance(a, ENTITY_TYPE):
            a = asseries(a)
        self.axis = 0
        if isinstance(a, DATAFRAME_TYPE):
            assert self.output_list_scalar and self.axis == 0
            pa_schema = pandas_dtypes_to_arrow_schema(a.dtypes, unknown_as_string=True)
            if len(set(pa_schema.types)) == 1:
                out_dtype = ArrowDtype(pa.list_(pa_schema.types[0]))
            else:
                out_dtype = np.dtype("O")
            kw = {
                "dtype": out_dtype,
                "index_value": a.columns_value,
                "shape": (a.shape[1],),
            }
            self.output_types = [OutputType.series]
            return self.new_tileables([a], **kw)[0]
        else:
            if self.output_list_scalar:
                arrow_type = pa.list_(
                    pandas_dtype_to_arrow_type(a.dtype, unknown_as_string=True)
                )
                kw = {
                    "dtype": ArrowDtype(arrow_type),
                    "shape": (),
                }
                self.output_types = [OutputType.scalar]
            else:
                kw = {
                    "dtype": a.dtype,
                    "shape": (np.nan,),
                }
                self.output_types = [OutputType.tensor]
            return self.new_tileables([a], order=TensorOrder.C_ORDER, **kw)[0]


def _unique(values, method="tree", **kwargs):
    op = DataFrameUnique(method=method, **kwargs)
    return op(values)


def unique(values, method="tree"):
    """
    Uniques are returned in order of appearance. This does NOT sort.

    Parameters
    ----------
    values : 1d array-like
    method : 'shuffle' or 'tree', 'tree' method provide a better performance, 'shuffle'
    is recommended if the number of unique values is very large.

    See Also
    --------
    Index.unique
    Series.unique

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> import pandas as pd
    >>> md.unique(md.Series([2, 1, 3, 3])).execute()
    array([2, 1, 3])

    >>> md.unique(md.Series([2] + [1] * 5)).execute()
    array([2, 1])

    >>> md.unique(md.Series([pd.Timestamp('20160101'),
    ...                     pd.Timestamp('20160101')])).execute()
    array(['2016-01-01T00:00:00.000000000'], dtype='datetime64[ns]')

    >>> md.unique(md.Series([pd.Timestamp('20160101', tz='US/Eastern'),
    ...                      pd.Timestamp('20160101', tz='US/Eastern')])).execute()
    array([Timestamp('2016-01-01 00:00:00-0500', tz='US/Eastern')],
          dtype=object)
    """
    return _unique(values, method=method)
