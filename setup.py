#!/usr/bin/env python
import os
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

with open('requirements.txt') as f:
    REQUIRES = f.read().splitlines()

extras_require = {
   'jwst': ["jwst==1.3.3", "stcal", "asdf>=2.7.1,<2.11.0"],
   'hst': ["image_registration @ git+https://github.com/keflavich/image_registration.git"] # Need the GitHub version as 0.2.6 is required for python>=3.10, but 0.2.6 is not yet on PyPI
}

FILES = []
for root, _, files in os.walk("Eureka"):
    FILES += [os.path.join(root.replace("Eureka/", ""), fname) \
        for fname in files if not fname.endswith(".py") and not fname.endswith(".pyc")]

setup(name='Eureka',
      version='0.0.1',
      description='Lightcurve fitting package for time-series observations',
      packages=find_packages(".", exclude=["*.tests"]),
      package_data={'Eureka': FILES},
      install_requires=REQUIRES,
      author='Section 5',
      author_email='kbstevenson@gmail.com',
      license='MIT',
      url='https://github.com/kevin218/Eureka',
      long_description='',
      zip_safe=True,
      use_2to3=False,
      ext_modules = cythonize(["eureka/S3_data_reduction/niriss_cython.pyx"]),
      include_dirs = np.get_include(),
      extras_require=extras_require
)
