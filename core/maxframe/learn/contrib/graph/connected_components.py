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

from maxframe import opcodes

from ....core import OutputType
from ....dataframe.operators import DataFrameOperator, DataFrameOperatorMixin
from ....dataframe.utils import parse_index
from ....serialization.serializables import Int32Field, StringField
from ....utils import make_dtypes


class DataFrameConnectedComponentsOperator(DataFrameOperator, DataFrameOperatorMixin):
    _op_type_ = opcodes.CONNECTED_COMPONENTS

    vertex_col1 = StringField("vertex_col1", default=None)
    vertex_col2 = StringField("vertex_col2", default=None)
    max_iter = Int32Field("max_iter", default=6)

    def __call__(self, df):
        node_id_dtype = df.dtypes[self.vertex_col1]
        dtypes = make_dtypes({"id": node_id_dtype, "component": node_id_dtype})
        # this will return a dataframe and a bool flag
        new_dataframe_tileable_kw = {
            "shape": (np.nan, 2),
            "index_value": parse_index(pd.RangeIndex(0)),
            "columns_value": parse_index(dtypes.index, store_data=True),
            "dtypes": dtypes,
        }
        new_scalar_tileable_kw = {"dtype": np.dtype(np.bool_), "shape": ()}
        return self.new_tileables(
            [df],
            kws=[new_dataframe_tileable_kw, new_scalar_tileable_kw],
        )

    @property
    def output_limit(self):
        return 2


def connected_components(
    dataframe, vertex_col1: str, vertex_col2: str, max_iter: int = 6
):
    """
    The connected components algorithm labels each node as belonging to a specific connected component with the ID of
    its lowest-numbered vertex.

    Parameters
    ----------
    dataframe : DataFrame
        A DataFrame containing the edges of the graph.

    vertex_col1 : str
        The name of the column in `dataframe` that contains the one of edge vertices. The column value must be an
        integer.

    vertex_col2 : str
        The name of the column in `dataframe` that contains the other one of edge vertices. The column value must be an
        integer.

    max_iter : int
        The algorithm use large and small star transformation to find all connected components, `max_iter`
        controls the max round of the iterations before finds all edges. Default is 6.


    Returns
    -------
    DataFrame
        Return dataFrame contains all connected component edges by two columns `id` and `component`. `component` is
        the lowest-numbered vertex in the connected components.

    Notes
    -------
    After `execute()`, the dataframe has a bool member `flag` to indicate if the `connected_components` already
    converged in `max_iter` rounds. `True` means the dataframe already contains all edges of the connected components.
    If `False` you can run `connected_components` more times to reach the converged state.

    Examples
    --------
    >>> import numpy as np
    >>> import maxframe.dataframe as md
    >>> import maxframe.learn.contrib.graph.connected_components
    >>> df = md.DataFrame({'x': [4, 1], 'y': [0, 4]})
    >>> df.execute()
       x  y
    0  4  1
    1  0  4

    Get connected components with 1 round iteration.

    >>> components, converged = connected_components(df, "x", "y", 1)
    >>> session.execute(components, converged)
    >>> components
       A   B
    0  1   0
    1  4   0

    >>> converged
    True

    Sometimes, a single iteration may not be sufficient to propagate the connectivity of all edges.
    By default, `connected_components` performs 6 iterations of calculations.
    If you are unsure whether the connected components have converged, you can check the `flag` variable in
    the output DataFrame after calling `execute()`.

    >>> df = md.DataFrame({'x': [4, 1, 7, 5, 8, 11, 11], 'y': [0, 4, 4, 7, 7, 9, 13]})
    >>> df.execute()
        x   y
    0   4   0
    1   1   4
    2   7   4
    3   5   7
    4   8   7
    5  11   9
    6  11  13

    >>> components, converged = connected_components(df, "x", "y", 1)
    >>> session.execute(components, converged)
    >>> components
       id  component
    0   4          0
    1   7          0
    2   8          4
    3  13          9
    4   1          0
    5   5          0
    6  11          9

    If `flag` is True, it means convergence has been achieved.

    >>> converged
    False

    You can determine whether to continue iterating or to use a larger number of iterations
    (but not too large, which would result in wasted computational overhead).

    >>> components, converged = connected_components(components, "id", "component", 1)
    >>> session.execute(components, converged)
    >>> components
       id  component
    0   4          0
    1   7          0
    2  13          9
    3   1          0
    4   5          0
    5  11          9
    6   8          0

    >>> components, converged = connected_components(df, "x", "y")
    >>> session.execute(components, converged)
    >>> components
       id  component
    0   4          0
    1   7          0
    2  13          9
    3   1          0
    4   5          0
    5  11          9
    6   8          0
    """

    # Check if vertex columns are provided
    if not vertex_col1 or not vertex_col2:
        raise ValueError("Both vertex_col1 and vertex_col2 must be provided.")

    # Check if max_iter is provided and within the valid range
    if max_iter is None:
        raise ValueError("max_iter must be provided.")
    if not (1 <= max_iter <= 50):
        raise ValueError("max_iter must be an integer between 1 and 50.")

    # Verify that the vertex columns exist in the dataframe
    missing_cols = [
        col for col in (vertex_col1, vertex_col2) if col not in dataframe.dtypes
    ]
    if missing_cols:
        raise ValueError(
            f"The following required columns {missing_cols} are not in {list(dataframe.dtypes.index)}"
        )

    # Ensure that the vertex columns are of integer type
    # TODO support string dtype
    incorrect_dtypes = [
        col
        for col in (vertex_col1, vertex_col2)
        if dataframe[col].dtype != np.dtype("int")
    ]
    if incorrect_dtypes:
        dtypes_str = ", ".join(str(dataframe[col].dtype) for col in incorrect_dtypes)
        raise ValueError(
            f"Columns {incorrect_dtypes} should be of integer type, but found {dtypes_str}."
        )

    op = DataFrameConnectedComponentsOperator(
        vertex_col1=vertex_col1,
        vertex_col2=vertex_col2,
        _output_types=[OutputType.dataframe, OutputType.scalar],
        max_iter=max_iter,
    )
    return op(
        dataframe,
    )
