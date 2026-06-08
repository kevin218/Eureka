
Installation
============

Installation methods
--------------------

``Eureka!`` should currently be installed directly from the GitHub repository.
PyPI installation is not currently supported because some runtime dependencies
are still installed from GitHub or other pip-only sources.

There are two supported installation routes:

- Install directly from GitHub with ``pip``.
- Create the repository-managed conda environment from the generated
  ``environment.yml`` file, then install ``Eureka!`` into that environment.

The examples below use the ``v1.4`` tag. If you prefer a different branch or
tag, substitute that ref in the Git URL shown below.

If you are using a macOS device with an Apple Silicon processor (for example,
M1), you may need to use the ``environment.yml`` route below because some pip
dependencies have been reported to fail to build on that platform.


Initial environment preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is **strongly** recommended that you install ``Eureka!`` in a fresh
``conda`` environment, since previously installed packages can introduce
dependency conflicts. You can install a lightweight version of conda at
`this link <https://docs.conda.io/en/latest/miniconda.html>`_. Once conda is
installed, create a new environment with:

.. code-block:: bash

	conda create -n eureka python=3.12
	conda activate eureka

If you use the repository ``environment.yml`` route below, the environment
creation command will create the environment for you.

Option 1) Install directly from GitHub with ``pip``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once your environment is active, install ``Eureka!`` directly from GitHub with
``pip``:

.. code-block:: bash

	python -m pip install "eureka-bang[jwst] @ git+https://github.com/kevin218/Eureka.git@v1.4"

To install from a branch or a different tag, append the desired Git ref with
``@``. For example:

.. code-block:: bash

	python -m pip install "eureka-bang[jwst] @ git+https://github.com/kevin218/Eureka.git@mybranchname"
	python -m pip install "eureka-bang[jwst] @ git+https://github.com/kevin218/Eureka.git@v1.3"

In order to use any of the demo ECF files, follow the instructions in the
:ref:`Demos <demos>` section of the :ref:`Quickstart <quickstart>` page.


Optional dependency groups
~~~~~~~~~~~~~~~~~~~~~~~~~~

``Eureka!`` defines several optional dependency groups in ``pyproject.toml``:

- ``jwst`` installs the packages needed to run Stages 1 and 2 on JWST data.
- ``hst`` installs the packages needed to run Stage 3 on HST/WFC3 data.
- ``docs`` installs the dependencies needed to build the documentation.
- ``test`` installs the local test tooling.
- ``jupyter`` installs Jupyter and notebook-related tooling.
- ``dev`` installs packaging and environment-generation tools used for project
  maintenance.

The ``jwst`` group is strongly recommended for JWST users.

You can combine groups as needed. For example:

.. code-block:: bash

	python -m pip install "eureka-bang[jwst,hst] @ git+https://github.com/kevin218/Eureka.git@v1.4"
	python -m pip install "eureka-bang[jwst,test,docs] @ git+https://github.com/kevin218/Eureka.git@v1.4"

Installing with the repository ``environment.yml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository also includes an ``environment.yml`` file for users who want the
repository-managed conda environment. This file is generated from
``pyproject.toml`` and should not be edited manually. It currently constrains
the repository-managed conda environment to Python 3.11-3.12.

This route is especially useful for contributors and for users who want the
repository-managed environment. To install this way:

.. code-block:: bash

	git clone -b v1.4 https://github.com/kevin218/Eureka.git
	cd Eureka
	conda env create --file environment.yml --force
	conda activate eureka
	python -m pip install --no-deps .


Upgrading your Eureka! installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The safest and most reliable way to upgrade ``Eureka!`` is to create a fresh
environment and reinstall the desired version there. Trying to mutate an older
environment in place can lead to dependency mismatches, and we cannot provide
support for upgrades performed that way.


Additional ExoTiC-LD Downloads
------------------------------

If you wish to use the ExoTiC-LD package to compute model stellar
limb-darkening profile coefficients (computed in Eureka!'s Stage 4 and used in
Stage 5), you will need to download the ExoTiC-LD stellar models and
instrument throughputs. For details, please visit ExoTiC-LD's
`installation instructions <https://exotic-ld.readthedocs.io/en/latest/views/installation.html>`_,
making sure to download the files corresponding to your installed ExoTiC-LD
version. The first number in the version number should match. For example, you
can use the v3.1.2 files with the v3.0.0 ExoTiC-LD package version.


CRDS Environment Variables
--------------------------

If you install the ``jwst`` dependency group, you must also set CRDS
environment variables so the JWST Calibration Pipeline can download the
reference files it needs. For users not on the internal STScI network, set the
following variables in your ``~/.zshrc`` (for zsh users), ``~/.bashrc`` or
``~/.bash_profile`` (for bash users), or another shell initialization file.
Choose a cache location appropriate for your system, such as
``/Users/your_name/crds_cache`` on macOS or
``/home/your_name/crds_cache`` on Linux.

.. code-block:: bash

	export CRDS_PATH=/PATH/TO/FOLDER/crds_cache

	export CRDS_SERVER_URL=https://jwst-crds.stsci.edu

To apply the changes, close your current terminal and open a new one, or run
``source ~/.bashrc`` (changing ``.bashrc`` to the shell startup file you use)
in your current terminal session.

If these environment variables are not set, Stages 1-3 of the pipeline will
fail with an error message similar to ``No such file or directory:
'/grp/crds/cache/config/jwst/server_config'``.

Issues with installing the jwst dependency
------------------------------------------
If you have issues installing the ``jwst`` dependency, check the debugging
advice on the :ref:`FAQ page <faq-install>`.
