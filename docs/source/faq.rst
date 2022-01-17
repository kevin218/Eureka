
Eureka! FAQ
============================

In this section you will find frequently asked questions about Eureka! as well as fixes for common problems

**Common Errors**
-----------------

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


FAQ
--------------------------
