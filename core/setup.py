# Copyright 1999-2024 Alibaba Group Holding Ltd.
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
from setuptools import Extension, setup

try:
    import distutils.ccompiler

    if sys.platform != "win32":
        from numpy.distutils.ccompiler import CCompiler_compile

        distutils.ccompiler.CCompiler.compile = CCompiler_compile
except ImportError:
    pass

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


repo_root = os.path.dirname(os.path.abspath(__file__))


cythonize_kw = dict(language_level=sys.version_info[0])
cy_extension_kw = dict()
if os.environ.get("CYTHON_TRACE"):
    cy_extension_kw["define_macros"] = [
        ("CYTHON_TRACE_NOGIL", "1"),
        ("CYTHON_TRACE", "1"),
    ]
    cythonize_kw["compiler_directives"] = {"linetrace": True}

if "MSC" in sys.version:
    extra_compile_args = ["/std:c11", "/Ot", "/I" + os.path.join(repo_root, "misc")]
    cy_extension_kw["extra_compile_args"] = extra_compile_args
else:
    extra_compile_args = ["-O3"]
    if sys.platform != "darwin":
        # for macOS, we assume that C++ 11 is enabled by default
        extra_compile_args.append("-std=c++0x")
    cy_extension_kw["extra_compile_args"] = extra_compile_args


# The pyx with C sources.
ext_include_source_map = {
    "maxframe/_utils.pyx": [
        ["maxframe/lib/mmh3_src"],
        ["maxframe/lib/mmh3_src/MurmurHash3.cpp"],
    ],
}


def _discover_pyx():
    exts = dict()
    for root, _, files in os.walk(os.path.join(repo_root, "maxframe")):
        for fn in files:
            if not fn.endswith(".pyx"):
                continue
            full_fn = os.path.relpath(os.path.join(root, fn), repo_root)
            include_dirs, source = ext_include_source_map.get(
                full_fn.replace(os.path.sep, "/"), [[], []]
            )
            mod_name = full_fn.replace(".pyx", "").replace(os.path.sep, ".")
            exts[mod_name] = Extension(
                mod_name,
                [full_fn] + source,
                include_dirs=[np.get_include()] + include_dirs,
                **cy_extension_kw,
            )
    return exts


extensions_dict = _discover_pyx()
cy_extensions = list(extensions_dict.values())

extensions = cythonize(cy_extensions, **cythonize_kw) + [
    Extension(
        "maxframe.lib.mmh3",
        [
            "maxframe/lib/mmh3_src/mmh3module.cpp",
            "maxframe/lib/mmh3_src/MurmurHash3.cpp",
        ],
    )
]

long_description = None
readme_files = [
    f"{repo_root}/../README.rst",
    f"{repo_root}/../README.md",
]
for readme_file in readme_files:
    if os.path.exists(readme_file):
        with open(readme_file) as f:
            long_description = f.read()
        break


setup_options = dict(
    long_description=long_description,
    ext_modules=extensions,
)
setup(**setup_options)
