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

from typing import Any, Dict, Optional


class MaxFrameError(Exception):
    pass


class MaxFrameIntentionalError(MaxFrameError):
    pass


class MaxFrameUserError(MaxFrameError):
    pass


class MaxFrameDeprecationError(MaxFrameUserError):
    pass


class TileableNotExecutedError(MaxFrameError):
    pass


class NoTaskServerResponseError(MaxFrameError):
    pass


class SessionAlreadyClosedError(MaxFrameError):
    def __init__(self, session_id: str):
        super().__init__(f"Session {session_id} is already closed")


class SessionNotFoundError(MaxFrameIntentionalError):
    def __init__(self, session_id: str):
        super().__init__(f"Session {session_id} not found")


class EngineUnavailableError(MaxFrameIntentionalError):
    def __init__(self, msg: str):
        super().__init__(msg)


class TypeInferWarning(UserWarning):
    pass


class OutputDtypeMismatchError(MaxFrameUserError, ValueError):
    """Raised when UDF output dtype doesn't match expected dtype"""

    def __init__(
        self,
        msg: Optional[str] = None,
        column: str = None,
        actual_dtype=None,
        expected_dtype=None,
        can_cast: bool = None,
        extra_msg: Optional[str] = None,
    ):
        if msg is not None:
            # If msg is provided, use it directly
            message = msg
        else:
            # Build message from parameters
            message = (
                f"Output dtype mismatch for column '{column}': "
                f"expected {expected_dtype}, got {actual_dtype}."
            )
            if can_cast:
                message += " Dtype will be automatically cast."
            else:
                message += " Cannot cast to expected dtype."
            if extra_msg:
                message += f" {extra_msg}"

        super().__init__(message)


class OutputColumnMismatchError(MaxFrameUserError, ValueError):
    """Raised when UDF output columns don't match expected columns"""

    def __init__(self, msg: Optional[str] = None, missing_cols=None, extra_cols=None):
        if msg is not None:
            # If msg is provided, use it directly
            message = msg
        else:
            # Build message from parameters
            parts = []
            if missing_cols:
                parts.append(f"Missing columns: {missing_cols}")
            if extra_cols:
                parts.append(f"Unexpected columns: {extra_cols}")
            message = "; ".join(parts)

        super().__init__(message)


def get_failure_info_from_exception(
    exception: BaseException, *, max_depth: int = 20
) -> Optional[Dict[str, Any]]:
    cur: Optional[BaseException] = exception
    depth = 0
    while cur is not None and depth < max_depth:
        fi = getattr(cur, "_failure_info", None)
        if fi:
            return fi
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        depth += 1
    return None
