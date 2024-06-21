.. _faq:

Eureka! FAQ
===========

In this section you will find frequently asked questions about Eureka! as well as fixes for common problems

**Common Errors**
-----------------

Missing packages during installation
''''''''''''''''''''''''''''''''''''

If you are encountering errors when installing Eureka! like missing packages (e.g. extension_helpers), be sure
that you are following the instructions on the on the :ref:`installation` page. If you are trying to directly
call setup.py using the ``python setup.py install`` command, you should instead be using a pip or conda
installation command which helps to make sure that all required dependencies are installed in the right order
and checks for implicit dependencies. If you still encounter issues, you should be sure that you are using a
new conda environment as other packages you've previously installed could have conflicting requirements with Eureka!.

If you are following the installation instructions and still encounter an error, please open a new Issue on
`GitHub Issues <https://github.com/kevin218/Eureka/issues>`__ and paste the full error message you are getting along
with details about which python version and operating system you are using.

Installation issues with M1 processors
''''''''''''''''''''''''''''''''''''''

Note that if you are using a macOS device with an M1 processor, you will need to use the ``conda`` environment.yml file
installation instructions as we have had reports that the pip dependencies fail to build on the M1 processor.


Issues installing or importing batman
'''''''''''''''''''''''''''''''''''''

Be sure that you are installing (or have installed) batman-package (not batman) from pip. If you have accidentally
installed the wrong package you can try pip uninstalling it, but you may just need to make a whole new environment.
In general, we strongly recommend you closely follow the instructions on the :ref:`installation` page.

Issues running starry/theano code
'''''''''''''''''''''''''''''''''

Be sure that you have installed the ``pymc3`` optional dependencies when installing Eureka! as described on the
:ref:`installation` page. If you're getting an error that includes ``theano/tensor/elemwise.py:264: TypeError``,
you most likely need to make sure that you have the g++ compiler installed (i.e., do ``sudo apt install g++`` if you're
on Ubuntu). If you're getting issues stating something like numpy was compiled for a different version, then you likely
need to make sure you're starting with a new conda environment and use the ``pymc3`` optional dependencies which set a
more strict upper-limit on the numpy version permitted. If you're getting an error that includes ``ImportError: libGL.so.1``,
then you either need to start with a new conda environment and make sure that you only install ``opencv-python-headless``
and not ``opencv-python``, or you need to install libgl1 by doing ``sudo apt install libgl1``.


.. _faq-install:

Issues installing or importing jwst
'''''''''''''''''''''''''''''''''''

As a first step, make sure that you were following the instructions on the :ref:`installation` page and were
either using the conda environment.yml installation method or a pip installation command that included "[jwst]"
like ``pip install .[jwst]``. If you are getting error messages on linux that mention gcc, you likely need to
first install gcc using ``sudo apt install gcc``. If you are getting messages on macOS that mention clang,
X-Code, and/or CommandLineTools, you need to make sure that CommandLineTools is first manually installed.
CommandLineTools is installed whenever you first manually use a command that requires it like git, so one very
simple way to install it would be to run ``git status`` in a terminal which should give a pop-up saying that
command line developer tools must be installed first. Alternatively, you can instead run
``xcode-select --install`` which will install CommandLineTools.

If you were doing that and you are still receiving error messages, it is possible that something about your
installation environment does not play well with jwst. You should open or comment on an already open issue on the Eureka!
`GitHub Issues <https://github.com/kevin218/Eureka/issues>`__ page and tell us as many details as you can about every step you
took to finally get to your error message as well as details about your operating system, python version, and conda version.
You should also consider opening an issue on the `jwst GitHub <https://github.com/spacetelescope/jwst/issues>`__ page as
there may not be much we can do to help troubleshoot a package we have no control over.

Finally, if you simply cannot get jwst to install and still want to use later stages of the Eureka! pipeline, then you can
install Eureka! using ``pip install .`` instead of ``pip install .[jwst]`` which will not install the jwst package. Note,
however, that this means that Stages 1 and 2 will not work at all as Eureka's Stages 1 and 2 simply offer ways of editing
the behaviour of the jwst package's Stages 1 and 2.

CRDS server connection error
''''''''''''''''''''''''''''

If you correctly set the CRDS environment variables during Eureka! installation, but still get an error that the CRDS server connection and cache load failed, try reloading your shell configuration file with ``source ~/.zshrc`` or ``source ~/.bashrc``.


Matplotlib RuntimeError() whenever Eureka is imported and plt.show() is called
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

The importing of Eureka! sometimes causes a runtime error with Matplotlib. The error is related to latex
and reads as the following

``RuntimeError: Failed to process string with tex because latex could not be found``

There are several workarounds to this problem. The first is to insert these lines
prior to calling ``plt.show()``


.. code-block:: python

    from matplotlib import rc
    rc('text', usetex=False)


Another solution would be to catch it as an exception:

.. code-block:: python

    try:
      plt.show()
    except RuntimeError:
      from matplotlib import rc
      rc('text', usetex=False)


Some more permanent solutions would be to:

- Install the following ``sudo apt install cm-super``, although this won't always work

- Identify where your TeX installation is and manually add it to PATH in your bashrc or bash_profile.
  An example of this is to change ``export PATH="~/anaconda3/bin:$PATH"`` in your **~/.bashrc** file to ``export PATH="~/anaconda3/bin:~/Library/TeX/texbin:$PATH"``.
  For anyone using Ubuntu or an older version of Mac this might be found in /usr/bin instead. Make sure you run source ~/.bash_profile or source ~/.bashrc to apply the changes.

My question isn't listed here!
''''''''''''''''''''''''''''''

First check to see if your question/concern is already addressed in an open or closed issue on the Eureka!
`GitHub Issues <https://github.com/kevin218/Eureka/issues>`__ page. If not, please open a new issue and paste the
full error message you are getting along with details about which python version and operating system you
are using, and ideally the ecf you used to get your error (ideally copy-paste it into the issue in a
quote block).
