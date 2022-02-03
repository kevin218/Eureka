#!/usr/bin/env python
import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    REQUIRES = f.read().splitlines()

DEPENDENCY_LINKS = ['git+https://github.com/spacetelescope/jwst_gtvt.git@cd6bc76f66f478eafbcc71834d3e735c73e03ed5']

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
      dependency_links=DEPENDENCY_LINKS,
      author='Section 5',
      author_email='kbstevenson@gmail.com',
      license='MIT',
      url='https://github.com/kevin218/Eureka',
      long_description='',
      zip_safe=True,
      use_2to3=False
)
