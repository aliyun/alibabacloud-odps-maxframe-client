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

from ..functional import deprecate_positional_args


def test_deprecate_positional_args():
    """Test function with many parameters."""

    @deprecate_positional_args
    def func(a, b=1, c=2, d=3, e=4):
        return a, b, c, d, e

    # Test that calling with many positional args generates warning for all after first
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # b, c, d, e passed positionally
        result = func(10, 20, 30, 40, 50)
        assert len(w) == 0  # No warnings should be raised
        assert result == (10, 20, 30, 40, 50)

    @deprecate_positional_args
    def func(a):
        return a

    # Test that calling with single positional arg works without warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = func(10)
        assert len(w) == 0  # No warnings should be raised
        assert result == 10

    @deprecate_positional_args
    def func(a=1, *, b=2, c=3):
        return a, b, c

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = func(10, 20, 30)  # b and c passed positionally
        assert len(w) == 1  # One warning for b and c
        assert "'b'" in str(w[0].message) and "'c'" in str(w[0].message)
        assert result == (10, 20, 30)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = func(10)  # no args passed positionally
        assert len(w) == 0  # No warning for b and c
        assert result == (10, 2, 3)
