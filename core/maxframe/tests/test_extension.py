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

import threading

import pytest
from mock import MagicMock, patch

import maxframe.extension as extension
from maxframe.extension import iter_extensions, register_extension_hooks_once
from maxframe.tests.utils import reset_extension_hooks


@pytest.fixture
def reset_hook_groups():
    # The once-per-process guards are module state; isolate them per test.
    with reset_extension_hooks():
        yield


def _provider(name, **hooks):
    """Extension stub with a provider-stable ``(module, qualname)`` identity."""
    return type(name, (), hooks)


@pytest.mark.parametrize(
    "groups, expected",
    [
        (
            [("register_operators", "register_dpe_tilers")],
            ["register_operators", "register_dpe_tilers"],
        ),
        ([("register_dpe_executors",)], ["register_dpe_executors"]),
        (
            [
                ("register_operators",),
                ("register_operators",),
                ("register_dpe_tilers",),
            ],
            ["register_operators", "register_dpe_tilers"],
        ),
    ],
    ids=["operators-and-tilers", "executors", "once-per-group"],
)
def test_register_extension_hooks_once(groups, expected, reset_hook_groups):
    ext = MagicMock()
    with patch.object(extension, "iter_extensions", return_value=[("app", ext)]):
        for group in groups:
            register_extension_hooks_once(*group)

    assert [call[0] for call in ext.method_calls] == expected


def test_hook_group_retried_after_failure(reset_hook_groups):
    # Two providers sharing an entry-point name, with A succeeding before B
    # fails.  A failing hook must not complete the group; the retry replays
    # only what still needs it, so A runs once and both run on the first pass.
    ext_a = _provider("ExtensionA", register_operators=MagicMock())
    ext_b = _provider(
        "ExtensionB",
        register_operators=MagicMock(side_effect=[RuntimeError("boom"), None]),
    )
    exts = [("dup", ext_a), ("dup", ext_b)]
    with patch.object(extension, "iter_extensions", return_value=exts):
        with pytest.raises(RuntimeError):
            register_extension_hooks_once("register_operators")
        register_extension_hooks_once("register_operators")

    assert ext_a.register_operators.call_count == 1
    assert ext_b.register_operators.call_count == 2


def test_discovered_extensions_support_registration_hooks():
    # The DPE tilers/executors dispatch paths invoke these hooks on every
    # discovered extension, so all of them must provide the defaults.
    hooks = ("register_operators", "register_dpe_tilers", "register_dpe_executors")
    for _, ext in iter_extensions():
        assert all(callable(getattr(ext, hook)) for hook in hooks)


def test_competing_thread_waits_for_registration(reset_hook_groups):
    # DAG analysis runs in a worker thread, so two first DAGs can race into the
    # same group.  The second caller must not read the registry mid-registration
    # (it used to return early with a KeyError from the dispatcher).
    entered, release = threading.Event(), threading.Event()
    registered, observed = [], []

    def register_dpe_tilers():
        entered.set()
        assert release.wait(10)
        registered.append("tiler")

    def register():
        register_extension_hooks_once("register_dpe_tilers")
        observed.append(list(registered))

    provider = _provider("BlockingProvider", register_dpe_tilers=register_dpe_tilers)
    with patch.object(extension, "iter_extensions", return_value=[("app", provider)]):
        first = threading.Thread(target=register)
        first.start()
        assert entered.wait(10)

        second = threading.Thread(target=register)
        second.start()
        second.join(0.5)
        assert second.is_alive(), "competing lookup returned during registration"

        release.set()
        first.join(10)
        second.join(10)

    # both callers observed the fully registered provider, registered once
    assert observed == [["tiler"], ["tiler"]]


def test_reentrant_lookup_during_registration_does_not_recurse(reset_hook_groups):
    # An extension may look a registry up while registering; the same thread
    # must short-circuit instead of replaying (or deadlocking on) the group.
    calls = []

    def register_dpe_tilers():
        calls.append(1)
        register_extension_hooks_once("register_dpe_tilers")

    provider = _provider("ReentrantProvider", register_dpe_tilers=register_dpe_tilers)
    with patch.object(extension, "iter_extensions", return_value=[("app", provider)]):
        register_extension_hooks_once("register_dpe_tilers")

    assert calls == [1]
