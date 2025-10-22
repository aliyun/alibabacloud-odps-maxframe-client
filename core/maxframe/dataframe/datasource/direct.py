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

import pandas as pd


def read_clipboard(sep=None, **kwargs):
    """
    Read text from clipboard and pass to :func:`~pandas.read_csv`.

    Parses clipboard contents similar to how CSV files are parsed
    using :func:`~pandas.read_csv`.

    Parameters
    ----------
    sep : str, default '\\s+'
        A string or regex delimiter. The default of ``'\\s+'`` denotes
        one or more whitespace characters.

    **kwargs
        See :func:`~pandas.read_csv` for the full argument list.

    Returns
    -------
    DataFrame
        A parsed :class:`DataFrame` object.

    See Also
    --------
    DataFrame.to_clipboard : Copy object to the system clipboard.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_fwf : Read a table of fixed-width formatted lines into DataFrame.

    Examples
    --------
    >>> import maxframe.dataframe as md
    >>> df = md.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])
    >>> df.to_clipboard()  # doctest: +SKIP
    >>> md.read_clipboard()  # doctest: +SKIP.execute()
         A  B  C
    0    1  2  3
    1    4  5  6
    """
    from ..initializer import DataFrame

    return DataFrame(pd.read_clipboard(sep=sep, **kwargs))
