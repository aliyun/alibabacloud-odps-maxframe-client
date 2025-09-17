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

import os
import platform
import sys
from sysconfig import get_config_vars

import numpy as np
from Cython.Build import cythonize
from pkg_resources import parse_version
from setuptools import Extension


def cythonize_all_pyx(repo_root, source_to_ext_include=None):
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
        if sys.platform != "darwin":
            # for macOS, we assume that C++ 11 is enabled by default
            extra_compile_args.append("-std=c++0x")
        cy_extension_kw["extra_compile_args"] = extra_compile_args

    # need to use c99 to compile python headers >= 3.12
    if sys.version_info[:2] >= (3, 12):
        os.environ["CFLAGS"] = "-std=c99"

    extensions_dict = dict()
    source_to_ext_include = source_to_ext_include or dict()
    for root, _, files in os.walk(repo_root):
        # skip build dir
        if "/build/" in root.replace("\\", "/"):
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
