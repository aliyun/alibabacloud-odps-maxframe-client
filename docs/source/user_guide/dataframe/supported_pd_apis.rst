Supported pandas APIs
---------------------
The table below shows implementation of pandas APIs on MaxFrame on certain engines.
If the API is not fully supported, unsupported item will be shown in the detail
column.

Series API
~~~~~~~~~~
.. currentmodule:: maxframe.dataframe.Series

.. list-table::
   :header-rows: 1

   * - API
     - SQL Engine
     - SPE
     - Details
   * - :func:`add`, :func:`radd`
     - P
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`all`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`any`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`apply`
     - N
     - Y
     -
   * - :func:`astype`
     - P
     - Y
     - SQL engine: converting to categorical types not supported.
   * - :func:`count`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`div`, :func:`rdiv`
     - P
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`drop_duplicates`
     - P
     - Y
     - SQL engine: maintaining original order of data not supported.
   * - :func:`eq`, :func:`ne`
     - P
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`fillna`
     - P
     - Y
     - SQL engine: argument ``downcast``, ``limit`` and ``method`` not supported.
   * - :func:`floordiv`, :func:`rfloordiv`
     - P
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`ge`, :func:`gt`
     - P
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - ``series[item]`` (or ```__getitem__``)
     - N
     - Y
     -
   * - :func:`iloc`
     - P
     - Y
     - SQL engine: Non-continuous indexes or negative indexes (for instance, ``df.iloc[[1, 3]]``, ``df.iloc[1:10:2]`` or ``df.iloc[-3:]``) not supported.
   * - :func:`isin`
     - P
     - Y
     - SQL engine: index input not supported.
   * - :func:`isna`
     - Y
     - Y
     -
   * - :func:`isnull`
     - Y
     - Y
     -
   * - :func:`le`, :func:`lt`
     - P
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`map`
     - P
     - Y
     - SQL engine: argument ``arg`` only supports functions and non-derivative dicts with simple scalars.
   * - :func:`max`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`mean`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`min`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`mod`, :func:`rmod`
     - P
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`mul`, :func:`rmul`
     - P
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`nunique`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``dropna==False`` not supported.
   * - :func:`pow`, :func:`rpow`
     - P
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`rename`
     - N
     - Y
     -
   * - :func:`replace`
     - P
     - Y
     - SQL engine: when the argument ``regex`` is True, ``list`` or ``dict`` typed ``to_replace`` not supported. ``list`` or ``dict`` typed ``regex`` argument not supported.
   * - :func:`sample`
     - P
     - Y
     - SQL engine: argument ``replace`` and ``weights`` not supported. ``frac>1`` not supported.
   * - :func:`sem`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`set_axis`
     - N
     - Y
     -
   * - ``series[item] = value`` (or ```__setitem__``)
     - P
     - Y
     - SQL engine: not supported when ``item`` is callable or DataFrame / Series or index of ``series`` and ``item`` are different.
   * - :func:`sort_index`
     - P
     - Y
     - SQL engine: ``na_position=='last'`` not supported.
   * - :func:`sort_values`
     - P
     - Y
     - SQL engine: ``na_position=='last'`` not supported.
   * - :func:`sub`, :func:`rsub`
     - P
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`sum`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.
   * - :func:`transform`
     - P
     - Y
     - SQL engine: not supported when ``func`` is dict-like or list-like.
   * - :func:`truediv`, :func:`rtruediv`
     - P
     - Y
     - SQL engine: argument ``level`` and ``fill_value`` not supported.
   * - :func:`value_counts`
     - N
     - Y
     -
   * - :func:`var`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported.

DataFrame API
~~~~~~~~~~~~~
.. currentmodule:: maxframe.dataframe.DataFrame

.. list-table::
   :header-rows: 1

   * - API
     - SQL Engine
     - SPE
     - Details
   * - :func:`add`, :func:`radd`
     - P
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`all`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`any`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`apply`
     - P
     - Y
     - SQL engine: ``axis==0`` not supported. Series output for ``axis==1`` not supported.
   * - :func:`astype`
     - P
     - Y
     - SQL engine: converting to categorical types not supported.
   * - :func:`count`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`div`, :func:`rdiv`
     - P
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`drop`
     - Y
     - Y
     -
   * - :func:`drop_duplicates`
     - P
     - Y
     - SQL engine: maintaining original order of data not supported.
   * - :func:`eq`, :func:`ne`
     - P
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`fillna`
     - P
     - Y
     - SQL engine: argument ``downcast``, ``limit`` and ``method`` not supported. ``axis==1`` not supported.
   * - :func:`floordiv`, :func:`rfloordiv`
     - P
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`ge`, :func:`gt`
     - P
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - ``df[item]`` (or ```__getitem__``)
     - P
     - Y
     - SQL engine: when index of ``item`` and ``df`` are different, boolean indexing not supported.
   * - :func:`iloc`
     - P
     - Y
     - SQL engine: single row selection with ``df.iloc[1]`` not supported. Non-continuous indexes or negative indexes (for instance, ``df.iloc[[1, 3]]``, ``df.iloc[1:10:2]`` or ``df.iloc[-3:]``) not supported.
   * - :func:`isin`
     - P
     - Y
     - SQL engine: index input not supported.
   * - :func:`isna`
     - Y
     - Y
     -
   * - :func:`isnull`
     - Y
     - Y
     -
   * - ``__invert__``
     - Y
     - Y
     -
   * - :func:`le`, :func:`lt`
     - P
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`max`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`mean`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`merge`
     - P
     - Y
     - SQL engine: argument ``indicator`` and ``validate`` not supported.
   * - :func:`min`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`mod`, :func:`rmod`
     - P
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`mul`, :func:`rmul`
     - P
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`notna`
     - Y
     - Y
     -
   * - :func:`notnull`
     - Y
     - Y
     -
   * - :func:`nunique`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` or ``dropna==False`` not supported.
   * - :func:`pow`, :func:`rpow`
     - P
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`rename`
     - P
     - Y
     - SQL engine: argument ``index`` not supported.
   * - :func:`replace`
     - P
     - Y
     - SQL engine: when the argument ``regex`` is True, ``list`` or ``dict`` typed ``to_replace`` not supported. ``list`` or ``dict`` typed ``regex`` argument not supported.
   * - :func:`sample`
     - P
     - Y
     - SQL engine: argument ``replace`` and ``weights`` not supported. ``axis==1`` or ``frac>1`` not supported.
   * - :func:`sem`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`set_axis`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==0`` not supported.
   * - ``df[item] = value`` (or ```__setitem__``)
     - P
     - Y
     - SQL engine: not supported when ``item`` is callable or DataFrame / Series or index of ``df`` and ``item`` are different.
   * - :func:`sort_index`
     - P
     - Y
     - SQL engine: ``axis==1`` or ``na_position=='last'`` not supported.
   * - :func:`sort_values`
     - P
     - Y
     - SQL engine: ``axis==1`` or ``na_position=='last'`` not supported.
   * - :func:`sub`, :func:`rsub`
     - P
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`sum`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.
   * - :func:`transform`
     - P
     - Y
     - SQL engine: not supported when ``axis==1`` and ``func`` is dict-like or list-like.
   * - :func:`truediv`, :func:`rtruediv`
     - P
     - Y
     - SQL engine: argument ``axis``, ``level`` and ``fill_value`` not supported.
   * - :func:`value_counts`
     - N
     - Y
     -
   * - :func:`var`
     - P
     - Y
     - SQL engine: argument ``skipna``, ``level`` and ``min_count`` not supported. ``axis==1`` not supported.


Index API
~~~~~~~~~~
.. currentmodule:: maxframe.dataframe.Index

.. list-table::
   :header-rows: 1

   * - API
     - SQL Engine
     - SPE
     - Details
   * - :func:`all`
     - N
     - Y
     -
   * - :func:`any`
     - N
     - Y
     -
   * - :func:`astype`
     - P
     - Y
     - SQL engine: converting to categorical types not supported.
   * - :func:`count`
     - N
     - Y
     -
   * - :func:`drop_duplicates`
     - P
     - Y
     - SQL engine: maintaining original order of data not supported.
   * - ``series[item]`` (or ```__getitem__``)
     - N
     - Y
     -
   * - :func:`iloc`
     - P
     - Y
     - SQL engine: Non-continuous indexes or negative indexes (for instance, ``df.iloc[[1, 3]]``, ``df.iloc[1:10:2]`` or ``df.iloc[-3:]``) not supported.
   * - :func:`isna`
     - Y
     - Y
     -
   * - :func:`isnull`
     - Y
     - Y
     -
   * - :func:`max`
     - N
     - Y
     -
   * - :func:`mean`
     - N
     - Y
     -
   * - :func:`min`
     - N
     - Y
     -
   * - :func:`nunique`
     - N
     - Y
     -
   * - :func:`sort_values`
     - N
     - Y
     -
   * - :func:`value_counts`
     - N
     - Y
     -
