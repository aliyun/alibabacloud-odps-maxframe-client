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


from maxframe.liteframe.misc.eval import lf_eval, lf_query
from maxframe.liteframe.misc.rechunk import rebalance, rechunk


def _install():
    from maxframe.liteframe.core import LITEFRAME_TYPE
    from maxframe.liteframe.misc.astype import astype
    from maxframe.liteframe.misc.drop_duplicates import drop_duplicates, duplicated
    from maxframe.liteframe.misc.explode import explode

    for cls in LITEFRAME_TYPE:
        setattr(cls, "astype", astype)
        setattr(cls, "explode", explode)
        setattr(cls, "eval", lf_eval)
        setattr(cls, "query", lf_query)
        setattr(cls, "drop_duplicates", drop_duplicates)
        setattr(cls, "duplicated", duplicated)
        setattr(cls, "rechunk", rechunk)
        setattr(cls, "rebalance", rebalance)


_install()
del _install
