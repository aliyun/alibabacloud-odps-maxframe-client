# Copyright 1999-2026 Alibaba Group Holding Ltd.
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


def _install():
    import pandas as pd

    from maxframe.dataframe.accessors.plotting.core import (
        DataFramePlotAccessor,
        PlotAccessor,
        SeriesPlotAccessor,
    )
    from maxframe.dataframe.core import (
        register_dataframe_accessor,
        register_series_accessor,
    )
    from maxframe.utils import adapt_docstring

    for method in dir(pd.DataFrame.plot):
        if not method.startswith("_"):
            DataFramePlotAccessor._register(method)
    register_dataframe_accessor("plot")(DataFramePlotAccessor)

    for method in dir(pd.Series.plot):
        if not method.startswith("_"):
            SeriesPlotAccessor._register(method)
    register_series_accessor("plot")(SeriesPlotAccessor)

    PlotAccessor.__doc__ = adapt_docstring(pd.DataFrame.plot.__doc__)


_install()
del _install
