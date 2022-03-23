
Eureka! FAQ
============================

In this section you will find frequently asked questions about Eureka! as well as fixes for common problems

**Common Errors**
-----------------

Missing packages during installation
''''''''''''''''''''''''''''''''''''

If you are encountering errors when installing Eureka! like missing packages (e.g. extension_helpers), be sure
that you are following the instructions on the on the :ref:`installation` page. If you are trying to directly
call setup.py using the ``python setup.py install`` command, you should instead be using ``pip install .`` which
helps to make sure that all required dependencies are installed in the right order and checks for implicit
dependencies. If you still encounter issues, you should be sure that you are using a new conda environment as
other packages you've previously installed could have conflicting requirements with Eureka!.

If you are following the installation instructions and still encounter an error, please open a new Issue on
`GitHub <https://github.com/kevin218/Eureka/issues>`_ and paste the full error message you are getting along
with details about which python version and operating system you are using.

Issues installing or importing batman
'''''''''''''''''''''''''''''''''''''

Be sure that you are installing (or have installed) batman-package (not batman) from pip. If you have accidentally
installed the wrong package you can try pip uninstalling it, but you may just need to make a whole new environment.
In general, we strongly recommend you closely follow the instructions on the :ref:`installation` page.


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
`GitHub <https://github.com/kevin218/Eureka/issues>`_ page. If not, please open a new issue and paste the
full error message you are getting along with details about which python version and operating system you
are using, and ideally the ecf you used to get your error (ideally copy-paste it into the issue in a
quote block).

FAQ
--------------------------
