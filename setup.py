import os, re, sys, shutil, platform, subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from setuptools.command.install_scripts import install_scripts
from distutils.command.install_data import install_data
from distutils.version import LooseVersion

PACKAGE_NAME = "pymtracking"

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class InstallCMakeLibsData(install_data):
    """
    Just a wrapper to get the install data into the egg-info
    Listing the installed files in the egg-info guarantees that
    all of the package files will be uninstalled when the user
    uninstalls your package through pip
    """
    def run(self):
        """
        Outfiles are the libraries that were built using cmake
        """
        # There seems to be no other way to do this; I tried listing the
        # libraries during the execution of the InstallCMakeLibs.run() but
        # setuptools never tracked them, seems like setuptools wants to
        # track the libraries through package data more than anything...
        # help would be appriciated
        self.outfiles = self.distribution.data_files

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
        if hasattr(self.distribution, 'bin_dir'):
            bin_dir = self.distribution.bin_dir
        else:
            bin_dir = os.path.join(self.build_dir, "Release")
            if not os.path.exists(bin_dir):
                bin_dir = "build/Release"
                self.build_dir = "build/Release"
        print("bin_dir:", bin_dir, "build_dir:", self.build_dir)
        # Depending on the files that are generated from your cmake
        # build chain, you may need to change the below code, such that
        # your files are moved to the appropriate location when the installation
        # is run
        libs = [os.path.join(bin_dir, _lib) for _lib in 
                os.listdir(bin_dir) if 
                os.path.isfile(os.path.join(bin_dir, _lib)) and 
                os.path.splitext(_lib)[1] in [".dll", ".so"]
                and not (_lib.startswith("python") or _lib.startswith(PACKAGE_NAME))]
        for lib in libs:
            shutil.move(lib, os.path.join(self.build_dir,
                                          os.path.basename(lib)))
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

__metaclass__ = type
class InstallCMakeScripts(install_scripts, object):
    """
    Install the scripts in the build dir
    """
    def run(self):
        """
        Copy the required directory to the build directory and super().run()
        """
        self.announce("Moving scripts files", level=3)
        # Scripts were already built in a previous step
        self.skip_build = True
        bin_dir = self.distribution.bin_dir
        scripts_dirs = [os.path.join(bin_dir, _dir) for _dir in
                        os.listdir(bin_dir) if
                        os.path.isdir(os.path.join(bin_dir, _dir))]
        for scripts_dir in scripts_dirs:
            shutil.move(scripts_dir,
                        os.path.join(self.build_dir,
                                     os.path.basename(scripts_dir)))
        # Mark the scripts for installation, adding them to 
        # distribution.scripts seems to ensure that the setuptools' record 
        # writer appends them to installed-files.txt in the package's egg-info
        self.distribution.scripts = scripts_dirs
        super(InstallCMakeScripts, self).run()

__metaclass__ = type
class BuildCMakeExt(build_ext, object):
    """
    Builds using cmake instead of the python setuptools implicit build
    """
    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """
        for extension in self.extensions:
            self.build_cmake(extension)
        super(BuildCMakeExt, self).run()

    def build_cmake(self, extension):
        """
        The steps required to build the extension
        """
        self.announce("Preparing the build environment", level=3)
        build_dir = os.path.join(self.build_temp)
        extension_path = os.path.abspath(os.path.dirname(self.get_ext_fullpath(extension.name)))
        os.makedirs(build_dir)
        os.makedirs(extension_path)
        python_version = str(sys.version_info[0]) + "." + str(sys.version_info[1])

        # Now that the necessary directories are created, build
        self.announce("Configuring cmake project", level=3)
        cmake_args = ['-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DUSE_OCV_BGFG=ON',
                      '-DUSE_OCV_KCF=ON',
                      '-DSILENT_WORK=ON',
                      '-DBUILD_EXAMPLES=OFF',
                      '-DBUILD_ASYNC_DETECTOR=OFF',
                      '-DBUILD_CARS_COUNTING=OFF',
                      '-DBUILD_YOLO_LIB=OFF',
                      '-DBUILD_YOLO_TENSORRT=OFF',
                      '-DMTRACKER_PYTHON=ON']
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        self.spawn(['cmake', '-H'+extension.sourcedir, '-B'+self.build_temp]+ cmake_args)

        self.announce("Building binaries", level=3)
        self.spawn(["cmake", "--build", self.build_temp, 
                    "--config", "Release", '--', '-j8'])

        # Build finished, now copy the files into the copy directory
        # The copy directory is the parent directory of the extension (.pyd)
        self.announce("Moving built python module", level=3)
        
        bin_dir = "build" # self.build_temp
        self.distribution.bin_dir = bin_dir
        list_bin = os.listdir(bin_dir)
        print("bin_dir:", bin_dir, ", extension_path:", extension_path, ", list_bin:", list_bin)
        pyd_path = []
        for _pyd in list_bin:
            print("_pyd:", _pyd)
            if os.path.isfile(os.path.join(bin_dir, _pyd)) and os.path.splitext(_pyd)[0].startswith(PACKAGE_NAME) and os.path.splitext(_pyd)[1] in [".pyd", ".so"]:
                   pyd_path.append(os.path.join(bin_dir, _pyd))
        print("pyd_path:", pyd_path)
        pyd_path = pyd_path[0]
        shutil.move(pyd_path, extension_path)

        # After build_ext is run, the following commands will run:
        # 
        # install_lib
        # install_scripts
        # 
        # These commands are subclassed above to avoid pitfalls that
        # setuptools tries to impose when installing these, as it usually
        # wants to build those libs and scripts as well or move them to a
        # different place. See comments above for additional information

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name=PACKAGE_NAME,
    version='1.0.1',
    author='Nuzhny007',
    author_email='nuzhny@mail.ru',
    url='https://github.com/Smorodov/Multitarget-tracker',
    license='Apache 2.0',
    description='Official Python wrapper for Multitarget-tracker',
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension(name=PACKAGE_NAME, sourcedir='.')],
    cmdclass={
        'build_ext': BuildCMakeExt,
        'install_data': InstallCMakeLibsData,
        'install_lib': InstallCMakeLibs,
        #'install_scripts': InstallCMakeScripts
        },
    zip_safe=False,
    packages=find_packages(),
    keywords=['Multitarget-tracker', 'Multiple Object Tracking', 'Computer Vision', 'Machine Learning'],
)

