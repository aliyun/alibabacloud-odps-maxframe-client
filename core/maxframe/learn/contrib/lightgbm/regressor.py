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

from ..utils import make_import_error_func
from .core import LGBMScikitLearnBase

try:
    import lightgbm
except ImportError:
    lightgbm = None


if not lightgbm:
    LGBMRegressor = make_import_error_func("lightgbm")
else:

    class LGBMRegressor(LGBMScikitLearnBase, lightgbm.LGBMRegressor):
        _default_objective = "regression"
