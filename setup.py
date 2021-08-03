import os
import re
import sys
import platform
import subprocess
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        if platform.system() == "Windows":
            cmake_args = ['-DPYTHON_EXECUTABLE=' + sys.executable]
        else:
            cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir, '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]


        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        cmake_args += ['-DBUILD_YOLO_LIB=OFF']
        cmake_args += ['-DBUILD_YOLO_TENSORRT=OFF']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir + os.path.sep + 'build'] + cmake_args, cwd='.', env=env)
        subprocess.check_call(['cmake', '--build', 'build'] + build_args, cwd='.')


__metaclass__ = type
class InstallCMakeLibs(install_lib, object):
    """
    Get the libraries from the parent distribution, use those as the outfiles

    Skip building anything; everything is already built, forward libraries to
    the installation step
    """
    def run(self):
        """
        Copy libraries from the bin directory and place them as appropriate
        """
        self.announce("Moving library files", level=3)
        # We have already built the libraries in the previous build_ext step
        self.skip_build = True

        if platform.system() == "Windows":
            if platform.system() == "Windows":
                bin_dir = os.path.join('build', 'Release')
            else:
                bin_dir = 'build'
            print(bin_dir)
            libs = [os.path.join(bin_dir, _lib) for _lib in
                    os.listdir(bin_dir) if
                    os.path.isfile(os.path.join(bin_dir, _lib)) and
                    os.path.splitext(_lib)[1] in [".dll", ".so"]
                    and not (_lib.startswith("python"))]
        else:
            if hasattr(self.distribution, 'bin_dir'):
                bin_dir = self.distribution.bin_dir
            else:
                bin_dir = os.path.join(self.build_dir)

            # Depending on the files that are generated from your cmake
            # build chain, you may need to change the below code, such that
            # your files are moved to the appropriate location when the installation
            # is run
            libs = [os.path.join(bin_dir, _lib) for _lib in
                    os.listdir(bin_dir) if
                    os.path.isfile(os.path.join(bin_dir, _lib)) and
                    os.path.splitext(_lib)[1] in [".dll", ".so"]
                    and not (_lib.startswith("python"))]

        print('Install libs', libs, 'from', bin_dir, 'to', self.build_dir)
        for lib in libs:
            shutil.move(lib, os.path.join(self.build_dir, os.path.basename(lib)))
        # Mark the libs for installation, adding them to
        # distribution.data_files seems to ensure that setuptools' record
        # writer appends them to installed-files.txt in the package's egg-info
        #
        # Also tried adding the libraries to the distribution.libraries list,
        # but that never seemed to add them to the installed-files.txt in the
        # egg-info, and the online recommendation seems to be adding libraries
        # into eager_resources in the call to setup(), which I think puts them
        # in data_files anyways.
        #
        # What is the best way?
        # These are the additional installation files that should be
        # included in the package, but are resultant of the cmake build
        # step; depending on the files that are generated from your cmake
        # build chain, you may need to modify the below code
        self.distribution.data_files = [os.path.join(self.install_dir,
                                                     os.path.basename(lib))
                                        for lib in libs]
        # Must be forced to run after adding the libs to data_files
        self.distribution.run_command("install_data")
        super(InstallCMakeLibs, self).run()


setup(
    name='pymtracking',
    version='1.0.1',
    author='Nuzhny007',
    author_email='nuzhny@mail.ru',
    url='https://github.com/Nuzhny007',
    description='Multipe object tracking library',
    long_description='',
    ext_modules=[CMakeExtension('pymtracking')],
    #cmdclass=dict(build_ext=CMakeBuild, install_lib=InstallCMakeLibs),
    cmdclass={
        'build_ext': CMakeBuild,
        'install_lib': InstallCMakeLibs
    },
    zip_safe=False,
)
