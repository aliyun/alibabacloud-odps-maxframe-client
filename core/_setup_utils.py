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

import os
import platform
import re
import shutil
import subprocess
import sys
from sysconfig import get_config_vars

from setuptools import Extension

try:
    from packaging.version import parse as parse_version
except ImportError:
    from pkg_resources import parse_version

_release_prefix = "release/v"
_dev_pattern = re.compile(r"^0\.1\.0\.dev(?P<dev>\d+)$")
_fallback_version = "0.1.0.dev0"


def _get_gcc_version():
    gcc_executable = os.getenv("CC", "gcc")
    try:
        proc = subprocess.run([gcc_executable, "--version"], capture_output=True)
    except:
        return None

    gcc_ver_pattern = re.search(r"([\d\.]+)", proc.stdout.decode(), flags=re.I)
    if not gcc_ver_pattern:
        return None
    return parse_version(gcc_ver_pattern.group(1)).release


def cythonize_all_pyx(repo_root, source_to_ext_include=None):
    import numpy as np
    from Cython.Build import cythonize

    # From https://github.com/pandas-dev/pandas/pull/24274:
    # For mac, ensure extensions are built for macos 10.9 when compiling on a
    # 10.9 system or above, overriding distuitls behaviour which is to target
    # the version that python was built for. This may be overridden by setting
    # MACOSX_DEPLOYMENT_TARGET before calling setup.py
    if sys.platform == "darwin":
        if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
            current_system = platform.mac_ver()[0]
            python_target = get_config_vars().get(
                "MACOSX_DEPLOYMENT_TARGET", current_system
            )
            target_macos_version = "10.9"

            parsed_python_target = parse_version(python_target)
            parsed_current_system = parse_version(current_system)
            parsed_macos_version = parse_version(target_macos_version)
            if parsed_python_target <= parsed_macos_version <= parsed_current_system:
                os.environ["MACOSX_DEPLOYMENT_TARGET"] = target_macos_version

    cythonize_kw = dict(language_level=sys.version_info[0])
    cy_extension_kw = dict()
    if os.environ.get("CYTHON_TRACE"):
        cy_extension_kw["define_macros"] = [
            ("CYTHON_TRACE_NOGIL", "1"),
            ("CYTHON_TRACE", "1"),
        ]
        cythonize_kw["compiler_directives"] = {"linetrace": True}

    if "MSC" in sys.version:
        extra_compile_args = ["/std:c11", "/Ot"]
        cy_extension_kw["extra_compile_args"] = extra_compile_args
    else:
        extra_compile_args = ["-O3"]
        extra_link_args = []
        if sys.platform != "darwin":
            # for macOS, we assume that C++ 11 is enabled by default
            gcc_ver = _get_gcc_version()
            gcc_loc = shutil.which("gcc")
            try:
                gcc_loc = os.readlink(gcc_loc) if gcc_loc else None
            except:
                pass

            if gcc_ver:
                if gcc_ver[0] < 5:
                    # need flag for old GCC release
                    extra_compile_args.append("-std=c++0x")
                elif gcc_loc and all(
                    not gcc_loc.startswith(prefix)
                    for prefix in [
                        "/bin",
                        "/usr/bin",
                        "/usr/local/bin",
                        "/usr/share/bin",
                    ]
                ):
                    # installation of GCC with symbolic links need to
                    #  link to libstdc++ manually
                    extra_link_args.append("-lstdc++")
        cy_extension_kw["extra_compile_args"] = extra_compile_args
        cy_extension_kw["extra_link_args"] = extra_link_args

    # need to use c99 to compile python headers >= 3.12
    if sys.version_info[:2] >= (3, 12):
        os.environ["CFLAGS"] = "-std=c99"

    extensions_dict = dict()
    source_to_ext_include = source_to_ext_include or dict()
    for root, _, files in os.walk(repo_root):
        # skip build dir
        # only scan for /build/ in relative path as the root dir
        #  might contain /build/
        rel_dir = os.path.relpath(root, repo_root)
        if "/build/" in ("./" + rel_dir.replace("\\", "/")):
            continue

        for fn in files:
            if not fn.endswith(".pyx"):
                continue
            full_fn = os.path.relpath(os.path.join(root, fn), repo_root)
            include_dirs, source = source_to_ext_include.get(
                full_fn.replace(os.path.sep, "/"), [[], []]
            )
            mod_name = full_fn.replace(".pyx", "").replace(os.path.sep, ".")
            extensions_dict[mod_name] = Extension(
                mod_name,
                [full_fn] + source,
                include_dirs=[np.get_include()] + include_dirs,
                **cy_extension_kw,
            )

    cy_extensions = list(extensions_dict.values())
    return cythonize(cy_extensions, **cythonize_kw)


def _build_scm_main_version(version) -> str:
    try:
        from setuptools_scm.version import release_branch_semver_version
    except ImportError:
        from setuptools_scm.version import (
            release_branch_semver as release_branch_semver_version,
        )

    ver = release_branch_semver_version(version)
    if not ver.startswith("0.1.0.dev"):
        return ver

    dev_match = _dev_pattern.match(ver)
    if not dev_match:
        return ver
    dev_val = int(dev_match.group("dev"))
    cur_branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode()
        .strip()
    )
    if cur_branch.startswith(_release_prefix):
        branch_ver = cur_branch[len(_release_prefix) :]
        parsed_version = parse_version(branch_ver.lstrip("v"))
        branch_ver = (
            f"{parsed_version.major}"
            f".{parsed_version.minor}"
            f".{parsed_version.micro or 0}"
        )
    else:
        version_tags = (
            subprocess.check_output(["git", "tag", "--list", "--sort=-version:refname"])
            .decode()
            .splitlines()
        )
        branch_ver = next(
            (t for t in version_tags if "rc" not in t and "sov" not in t), None
        )
        if branch_ver is None:
            return ver
        parsed_version = parse_version(branch_ver.lstrip("v"))
        branch_ver = (
            f"{parsed_version.major}"
            f".{parsed_version.minor + 1}"
            f".{parsed_version.micro}"
        )
    return f"{branch_ver}.dev{dev_val}"


def _build_local_scheme(version) -> str:
    from setuptools_scm.version import get_local_node_and_date

    if version.distance == 0:
        return ""
    local_ver = get_local_node_and_date(version)
    if str(version.tag) == "0.0":
        return f"+br.{local_ver.lstrip('+')}"
    return local_ver


def make_scm_version(root=""):
    def func():
        # Check if .git folder exists in script directory or parent directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, root))

        has_git = os.path.exists(os.path.join(script_dir, ".git")) or os.path.exists(
            os.path.join(root_dir, ".git")
        )

        result = {
            "root": root,
            "version_scheme": _build_scm_main_version,
            "local_scheme": _build_local_scheme,
        }

        # Add fallback version if no .git folder is found
        if not has_git:
            result["fallback_version"] = _fallback_version
        return result

    return func
