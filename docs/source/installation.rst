
Installation
============

Initial environment preparation
-------------------------------
It is **strongly** recommended that you install ``Eureka!`` in a new ``conda`` environment as other packages you've previously
installed could have conflicting requirements with ``Eureka!``. You can install a lightweight version of conda at `this link <https://docs.conda.io/en/latest/miniconda.html>`_. Once conda is installed, you can create a
new environment by doing:

.. code-block:: bash

	conda create -n eureka python==3.9.7
	conda activate eureka


Installation methods
--------------------

a) With ``git`` and ``pip``
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once in your new conda environment, you can install ``Eureka!`` directly from source on
`GitHub <http://github.com/kevin218/Eureka>`_ using ``git`` and ``pip`` by running:

.. code-block:: bash

	git clone https://github.com/kevin218/Eureka.git
	cd Eureka
	pip install -e '.[jwst]'

To update your ``Eureka!`` installation to the most recent version, you can do the following within that Eureka folder

.. code-block:: bash

	git pull
	pip install --upgrade '.[jwst]'

b) With ``git`` and ``conda``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once in your new conda environment, you can install ``Eureka!`` using ``git`` and ``conda``. This can be done following:

.. code-block:: bash

	git clone https://github.com/kevin218/Eureka.git
	cd Eureka
	conda env create --file environment.yml --force
	conda activate eureka
	pip install --no-deps .

To update your ``Eureka!`` installation to the most recent version, you can do the following within that Eureka folder

.. code-block:: bash

	git pull
	conda env update --file environment.yml --prune
	pip install --no-deps --upgrade .


c) With ``pip``
~~~~~~~~~~~~~~~

Once in your new conda environment, you can install the ``Eureka!`` package with ``pip`` with the following command:

.. code-block:: bash

	pip install -e git+https://github.com/kevin218/Eureka.git#egg=eureka[jwst]

where specific branches can be installed using:

.. code-block:: bash
	
	pip install -e git+https://github.com/kevin218/Eureka.git@mybranchname#egg=eureka[jwst]

If you desire any of the files in the `demos folder <https://github.com/kevin218/Eureka/tree/main/demos>`_, you will have to download these from
GitHub following the method described below.

To update your ``Eureka!`` installation to the most recent version, you can do then do the following

.. code-block:: bash

	pip install --upgrade git+https://github.com/kevin218/Eureka.git#egg=eureka[jwst]




CRDS Environment Variables
--------------------------

``Eureka!`` installs the JWST Calibration Pipeline as part of its requirements, and this also requires users to set the proper environment
variables so that it can download the proper reference files needed to run the pipeline. For users not on the internal STScI network,
two environment variables need to be set to enable this functionality. In your ``~/.zshrc`` (for Mac users) or ``~/.bashrc`` file (for bash
users), or other shell initialization file, add these two lines (specifying your desired location to cache the CRDS files,
e.g. ``/Users/your_name/crds_cache`` for Mac users or ``/home/your_name/crds_cache`` for Linux users):

	.. code-block:: bash

		export CRDS_PATH=/PATH/TO/FOLDER/crds_cache
		
		export CRDS_SERVER_URL=https://jwst-crds.stsci.edu

If these environment variables are not set, Stages 1-3 of the pipeline will fail.

Issues with installing the jwst dependency
------------------------------------------
If you have issues installing the jwst dependency, check out the debugging advice related to the jwst package on our
:ref:`FAQ page <faq-install>`.
