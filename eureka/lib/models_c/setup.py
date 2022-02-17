from platform import system
from numpy import get_include
import os
import re
from distutils.core import setup, Extension

files = os.listdir('c_code/')
#this will filter the results for just the c files
#files = filter(lambda x: not re.search('.+[.]c[~]$',x),files)
#files = filter(lambda x: not re.search('[.#].+[.]c$',x),files)
#For python 3, also appears to work for Python 2
files = list(filter(lambda x: not re.search('.+[.]c[~]$',x),files))
files = list(filter(lambda x: not re.search('[.#].+[.]c$',x),files))
# files.remove(".svn")

ext_mod = []
#inc = [get_include(),'/Users/mayem1/Homebrew/opt/libomp/lib/','/Users/mayem1/Homebrew/opt/libomp/include/']
inc = [get_include(),'/usr/local/Cellar/libomp/11.0.0/lib/','/usr/local/Cellar/libomp/11.0.0/include/']

for idx,fname in enumerate(files):
    if system() == 'Linux':
        # LINUX
        # exec('mod'+str(idx)+'=Extension("'+fname.rstrip('.c')+'",sources=["c_code/'+fname+'"],include_dirs=inc,extra_compile_args=["-fopenmp"],extra_link_args=["-lgomp"])')
        exec('mod{}=Extension("{}",sources=["c_code/{}"],include_dirs=inc,extra_compile_args=["-fopenmp"],extra_link_args=["-lgomp"])'.format(idx, fname.rstrip('.c'), fname))
    elif system() == 'Darwin':
        # MAC
        exec('mod{}=Extension("{}",sources=["c_code/{}"],include_dirs=inc)'.format(idx, fname.rstrip('.c'), fname))
    elif system() == 'Windows':
        raise Exception('This package has not been tested on Windows; please edit the setup.py appropriately.')

    exec('ext_mod.append(mod'+str(idx)+')')

setup(name='models_c',version='1.0',description='Models in c for mcmc', ext_modules = ext_mod)
