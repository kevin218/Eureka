import os
import re
import sys

temp = str(__import__("models_c"))
dir  = re.findall('.+\'(.+)__init',temp)[0]

sys.path.append(dir+'ext_func/')
sys.path.append(dir+'py_func/')

ext = os.listdir(dir+'ext_func/')

namespace = {}
for mod in ext:
    if mod.endswith('.so'):
        try:
            #Python 2
            exec('from '+mod.partition('.')[0]+' import *')
            #Python 3, doesn't quite work
            #exec('from '+mod.partition('.')[0]+' import *', namespace)
        except:
            print('Warning: Could not import ' + mod)
            #print("Unexpected error:", sys.exc_info()[0])

pys = os.listdir(dir+'py_func/')
for mod in pys:
    if mod.endswith('.py'):
        try:
            exec('from '+mod.partition('.')[0]+' import *')
            #exec('reload(' + mod.partition('.')[0] + ')')
            #exec('from '+mod.partition('.')[0]+' import *', namespace)
        except:
            print('Warning: Could not import ' + mod)
            #print("Unexpected error:", sys.exc_info()[0])

#sys.path.remove(dir+'ext_func/')
#sys.path.remove(dir+'ext_func/')
