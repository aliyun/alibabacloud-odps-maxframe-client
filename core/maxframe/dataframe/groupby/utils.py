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

import warnings

from ...utils import add_survey_log


def warn_prepend_index_group_keys(in_groupby):
    groupby_obj = in_groupby.op.groupby_params
    if groupby_obj.get("group_keys", True):
        from ... import __version__

        warnings.warn(
            "Group keys will be prepended automatically for returned "
            "indexes by default in future versions of MaxFrame. Requirement to "
            "pass group keys to index when skip_infer=True is deprecated. To "
            "avoid potential incompatibility, please specify "
            "prepend_index_group_keys=True to enable this behavior and silence "
            "this warning.",
            FutureWarning,
        )
        add_survey_log(
            {
                "Method": "apply_chunk",
                "Key": "prepend_index_group_keys",
                "Message": "prepend_index_group_keys==False with group_keys==True",
                "ClientVersion": __version__,
            }
        )


def warn_axis_argument(method_name, kwargs):
    if "axis" in kwargs:
        warnings.warn(
            f"The 'axis' argument takes no effect on the behavior of {method_name} "
            "and will be passed to the function directly. If you are trying to call "
            f"{method_name} with axis=1, try {method_name} without groupby as grouping "
            "with axis=1 makes no difference to the final result.",
            UserWarning,
        )
