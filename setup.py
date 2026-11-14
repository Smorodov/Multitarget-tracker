"""Build and package the pymtracking pybind11 module.

The python module is built by the regular CMake machinery of this
repository (MTRACKER_PYTHON=ON). The package metadata lives here, while
the version comes from the single source: the project() line of the root
CMakeLists.txt (roadmap H16).

Extra cmake cache variables can be passed through the CMAKE_ARGS
environment variable, e.g.:
    set CMAKE_ARGS=-DOpenCV_DIR=C:/libs/opencv/build
    python -m pip wheel .

Note: the wheel contains only the python module; the C++ runtime
dependencies (OpenCV etc.) must be available to the loader at runtime:
on Windows (Python 3.8+) via os.add_dll_directory()/system directories
(the module search PATH is not used for extension DLL dependencies),
on Linux the usual loader rules apply.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

PACKAGE_NAME = "pymtracking"


def get_version():
    """Parse the project VERSION from the root CMakeLists.txt."""
    cmake_lists = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CMakeLists.txt")
    with open(cmake_lists, "r", encoding="utf-8") as fh:
        text = fh.read()
    match = re.search(r"project\s*\(\s*MTTracking\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", text)
    if not match:
        raise RuntimeError("Can't parse the project VERSION from CMakeLists.txt")
    return match.group(1)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class BuildCMakeExt(build_ext):
    """Builds the extension using CMake instead of the setuptools implicit
    build."""

    def run(self):
        for extension in self.extensions:
            self.build_cmake(extension)
        super().run()

    def build_cmake(self, extension):
        self.announce("Preparing the build environment", level=3)
        build_dir = os.path.abspath(self.build_temp)
        os.makedirs(build_dir, exist_ok=True)
        extension_path = os.path.abspath(os.path.dirname(self.get_ext_fullpath(extension.name)))
        os.makedirs(extension_path, exist_ok=True)

        # Configure and build. Only the python binding stack is needed:
        # examples, the async detector and the CarsCounting example are
        # switched off, SILENT_WORK removes all imshow/waitKey calls.
        self.announce("Configuring cmake project", level=3)
        cmake_args = [
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DUSE_OCV_BGFG=ON",
            "-DUSE_OCV_KCF=ON",
            "-DSILENT_WORK=ON",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_ASYNC_DETECTOR=OFF",
            "-DBUILD_CARS_COUNTING=OFF",
            "-DMTRACKER_PYTHON=ON",
        ]
        extra_args = os.environ.get("CMAKE_ARGS", "")
        if extra_args:
            cmake_args += shlex.split(extra_args)

        configure_cmd = ["cmake", "-S" + extension.sourcedir, "-B" + build_dir] + cmake_args
        # MSBuild (Windows) understands /m, make/ninja (linux/mac) use -j.
        parallel_arg = "/m" if sys.platform == "win32" else "-j8"
        build_cmd = ["cmake", "--build", build_dir, "--config", "Release", "--", parallel_arg]
        for cmd in (configure_cmd, build_cmd):
            self.announce(" ".join(cmd), level=2)
            if subprocess.call(cmd) != 0:
                raise RuntimeError("CMake failed: " + " ".join(cmd))

        # Find the built module anywhere under the cmake build tree; fail
        # loudly if CMake did not produce it (an IndexError here was the old
        # behavior, roadmap H16).
        self.announce("Searching for the built python module", level=3)
        pyd_paths = []
        for root, _dirs, files in os.walk(build_dir):
            for name in files:
                base, ext = os.path.splitext(name)
                if base.startswith(PACKAGE_NAME) and ext in (".pyd", ".so"):
                    pyd_paths.append(os.path.join(root, name))
        if not pyd_paths:
            raise RuntimeError(
                "CMake did not produce the '%s' python module under '%s'. "
                "Check the cmake configure/build log above." % (PACKAGE_NAME, build_dir))

        self.announce("Moving the built python module", level=3)
        for pyd_path in pyd_paths:
            target = os.path.join(extension_path, os.path.basename(pyd_path))
            self.announce(pyd_path + " -> " + target, level=2)
            shutil.move(pyd_path, target)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=PACKAGE_NAME,
    version=get_version(),
    author="Nuzhny007",
    author_email="nuzhny@mail.ru",
    url="https://github.com/Smorodov/Multitarget-tracker",
    license="Apache 2.0",
    description="Official Python wrapper for Multitarget-tracker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension(name=PACKAGE_NAME, sourcedir=".")],
    cmdclass={
        "build_ext": BuildCMakeExt,
        },
    zip_safe=False,
    packages=find_packages(),
    keywords=["Multitarget-tracker", "Multiple Object Tracking", "Computer Vision", "Machine Learning"],
)
