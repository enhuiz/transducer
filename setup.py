import sys
from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

if sys.platform == "darwin":
    args = ["-DAPPLE"]
else:
    args = ["-fopenmp"]

setup(name='transducer',
      packages=['.'],
      ext_modules=[CppExtension('transducer_cpp', ['transducer.cpp'])],
      cmdclass={'build_ext': BuildExtension})
