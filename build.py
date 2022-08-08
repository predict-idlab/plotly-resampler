import os
import shutil
import sys

from distutils.command.build_ext import build_ext
from distutils.core import Distribution
from distutils.core import Extension
from distutils.errors import CCompilerError
from distutils.errors import DistutilsExecError
from distutils.errors import DistutilsPlatformError

import numpy as np

# C Extensions
with_extensions = True


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


extensions = []
if with_extensions:
    extensions = [
        Extension(
            name="plotly_resampler.aggregation.algorithms.lttbc",
            sources=["plotly_resampler/aggregation/algorithms/lttbc.c"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[np.get_include(), get_script_path()],
        ),
    ]


class BuildFailed(Exception):

    pass


class ExtBuilder(build_ext):
    # This class allows C extension building to fail.

    built_extensions = []

    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError) as e:
            print(
                "   Unable to build the C extensions, will use the slower python "
                "fallback for LTTB"
            )
            print(e)

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (
            DistutilsPlatformError,
            CCompilerError,
            DistutilsExecError,
            ValueError,
        ) as e:
            print(
                '   Unable to build the "{}" C extension; '.format(ext.name)
                + "will use the slower python fallback."
            )
            print(e)


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    distribution = Distribution({"name": "plotly_resampler", "ext_modules": extensions})
    distribution.package_dir = "plotly_resampler"

    cmd = ExtBuilder(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        if not os.path.exists(output):
            continue

        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)

    return setup_kwargs


if __name__ == "__main__":
    build({})
