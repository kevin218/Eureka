
Installation
============

Installation methods
--------------------

In order to have consistent, repeatable results across the ``Eureka!`` user community, we recommend that all general users install
the most recent stable release of ``Eureka!``, v1.0. The following installation instructions are written with this in mind,
and the most recent stable release is also available as a zipped archive `here <https://github.com/kevin218/Eureka/releases/tag/v1.0>`_.
Also note that if you are using a macOS device with an Apple Silicon processor (e.g., M1), you may need to use the ``conda`` environment.yml file
installation instructions below as the ``pip`` dependencies have been reported to fail to build on Apple Silicon processors.


Initial environment preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is **strongly** recommended that you install ``Eureka!`` in a new ``conda`` environment as other packages you've previously
installed could have conflicting requirements with ``Eureka!``. You can install a lightweight version of conda at
`this link <https://docs.conda.io/en/latest/miniconda.html>`_. Once conda is installed, you can create a
new environment by doing:

.. code-block:: bash

	conda create -n eureka python==3.10.14
	conda activate eureka

Alternatively, if you are following the "Installing with a ``conda`` environment.yml file" instructions below,
you will not need to manually make a new ``conda`` environment as the ``conda env create --file environment.yml --force``
line will make a new one for you (or replace your old one if you already had one).

Option 1) With ``git`` and ``pip``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once in your new conda environment, you can install ``Eureka!`` directly from source on
`GitHub <http://github.com/kevin218/Eureka>`_ using ``git`` and ``pip`` by running:

.. code-block:: bash

	git clone -b v1.0 https://github.com/kevin218/Eureka.git
	cd Eureka
	pip install -e '.[jwst]'


Option 2) With ``pip`` only
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once in your new conda environment, you can install the ``Eureka!`` package with ``pip`` with the following command:

.. code-block:: bash

	pip install -e 'eureka[jwst]@git+https://github.com/kevin218/Eureka.git@v1.0'

Other specific branches can be installed using:

.. code-block:: bash

	pip install 'eureka[jwst]@git+https://github.com/kevin218/Eureka.git@mybranchname'

In order to use any of the demo ECF files, follow the instructions in the :ref:`Demos <demos>` section of the :ref:`Quickstart <quickstart>` page.


Including optional dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are also several optional dependency collections that can be installed with Eureka! to increase the flexibility of the software. These include:

- ``jwst`` which includes the necessary packages to run Stages 1-2 on JWST data.
- ``hst`` which includes the necessary packages to run Stage 3 on HST/WFC3 data.
- ``test`` which allows you to run our suite of pytest tests locally.
- ``pymc3`` which allows you to use the NUTS Hamiltonian Monte Carlo sampler implemented in PyMC3 as well as a gradient based optimizer which benefits from differentiable models. This also allows you to use the starry astrophysical model for modelling exoplanet transits, eclipses (including eclipse mapping signals), and phase curves.
- ``docs`` which allows you to build the documentation pages locally.
- ``jupyter`` which includes jupyter and ipykernel for convenience.

In the installation instructions above, the ``jwst`` optional dependency is used as we strongly recommend users run Stages 1 and 2 locally, but we wanted to give users the ability to opt-out of installing the dependencies installed with ``jwst`` if they didn't work on their system.

To install with one or more optional dependency collections, the above examples can be generalized upon. For example, to install with just the ``hst`` dependencies, one can replace ``[jwst]`` with ``[hst]``. Or if you want to install with multiple options, you can do things like ``[jwst,hst]``.

.. warning::
	To install the ``pymc3`` optional dependencies, you also need to install ``mkl-service`` which can only be installed from conda using ``conda install mkl-service``.

	In addition, attempting to specify ``[jwst,pymc3]`` when installing ``Eureka!`` will fail with a dependency conflict, as the newest version of the ``jwst`` pipeline is incompatible with ``pymc3``. Optional NUTS users should only specify ``[pymc3]`` in their installs, which will default to a slightly older version of the ``jwst`` pipeline. Other optional dependencies are currently compatible.

Installing with a ``conda`` environment.yml file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also download ``Eureka!`` using ``git`` and set up a ``conda`` environment directly from the ``git`` repository if
you'd prefer not to use ``pip`` to install dependencies. To use the ``pymc3`` optional dependencies, replace ``environment.yml`` with ``environmenmt_pymc3.yml`` in the steps below.

To install using conda:

.. code-block:: bash

	git clone -b v1.0 https://github.com/kevin218/Eureka.git
	cd Eureka
	conda env create --file environment.yml --force
	conda activate eureka
	pip install --no-deps .


Upgrading your Eureka! installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The safest and most reliable way of upgrading your Eurkea! installation from one version to another is to start from scratch by creating a new ``conda`` environment
and installing the new Eureka! version in that fresh environment. Trying to upgrade your Eureka! installation within an existing environment
(i.e., without first making a new conda environment) can lead to dependency mismatches, and we cannot provide support to users trying to upgrade Eureka! in this manner.


Additional ExoTiC-LD Downloads
------------------------------

If you wish to use the ExoTiC-LD package to compute model stellar limb-darkening profile coefficients (computed in Eureka!'s Stage 4 and used in Stage 5),
you will need to download the ExoTiC-LD stellar models and instrument throughputs. For details on how to do that, please visit ExoTiC-LD's
`installation instructions <https://exotic-ld.readthedocs.io/en/latest/views/installation.html>`_, making sure to download the files corresponding to your
installed ExoTiC-LD version (make sure the first number in the version number is the same, e.g. you can use the v3.1.2 files with the v3.0.0 ExoTiC-LD package version).


CRDS Environment Variables
--------------------------

``Eureka!`` installs the JWST Calibration Pipeline as part of its requirements, and this also requires users to set the proper environment
variables so that it can download the proper reference files needed to run the pipeline. For users not on the internal STScI network,
two environment variables need to be set to enable this functionality. In your ``~/.zshrc`` (for zsh users) or ``~/.bashrc`` or ``~/.bash_profile`` file (for bash
users), or other shell initialization file, add these two lines (specifying your desired location to cache the CRDS files,
e.g. ``/Users/your_name/crds_cache`` for Mac users or ``/home/your_name/crds_cache`` for Linux users):

	.. code-block:: bash

		export CRDS_PATH=/PATH/TO/FOLDER/crds_cache

		export CRDS_SERVER_URL=https://jwst-crds.stsci.edu

In order for your changes to apply, you must close your current terminal(s) and open a new terminal; alternatively, you can instead do ``source ~/.bashrc``
(changing .bashrc to whichever filename your system uses) within your currently open terminal(s).

If these environment variables are not set, then Stages 1-3 of the pipeline will fail with an error message that says something like ``No such file or directory: '/grp/crds/cache/config/jwst/server_config'``

Issues with installing the jwst dependency
------------------------------------------
If you have issues installing the jwst dependency, check out the debugging advice related to the jwst package on our
:ref:`FAQ page <faq-install>`.
