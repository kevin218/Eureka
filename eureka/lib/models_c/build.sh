#!/bin/bash
#this will remove any partial builds, if there was a failure

if [ "$1" == "clean" ]
then
    rm ext_func/*.so
    exit
fi

if [ "$1" == "python2" ]
then
    python2.7 setup.py build_ext --inplace
else
    python3 setup.py build_ext --inplace
    ./rename_so_files.sh
fi

mkdir -p ext_func/
mv -f *.so ext_func/
rm -rf build/
echo "~~You made it!~~"
