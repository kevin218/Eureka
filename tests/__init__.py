# Some bits of code to help VS Code run tests automatically

import os
# VS Code runs tests from the root directory, so need to chdir into
# the tests directory
if os.getcwd().split(os.sep)[-1] != "tests":
    os.chdir('tests')
