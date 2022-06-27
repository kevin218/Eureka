#!/usr/bin/env python
from setuptools import setup
from Cython.Build import cythonize

if __name__ == "__main__":

    setup(
        ext_modules=cythonize(
            ["src/eureka/S3_data_reduction/niriss_cython.pyx"]),
    )
