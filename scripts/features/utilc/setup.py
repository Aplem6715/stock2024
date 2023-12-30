from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include  # cimport numpy を使うため

ext = Extension("scripts.features.utilc.zigzag",
                sources=["scripts/features/utilc/zigzag.pyx"],
                include_dirs=['.', get_include()],
                define_macros=[('CYTHON_TRACE', '1')])
setup(name="scripts.features.utilc.zigzag",
      ext_modules=cythonize([ext]),
      compiler_directives={'language_level': "3", 'profile': True, 'linetrace': True, 'binding': True})

ext = Extension("scripts.features.utilc.feat_util",
                sources=["scripts/features/utilc/feat_util.pyx"],
                include_dirs=['.', get_include()])
setup(name="scripts.features.utilc.feat_util",
      ext_modules=cythonize([ext]),
      compiler_directives={'language_level': "3"})
