
Installation
=============================

It is recommended that you install Eureka! in a new conda environment as other packages you've previously
installed could have conflicting requirements with Eureka!.

With pip
---------

The simplest way to install the Eureka! package is with the following one-line command:

.. code-block:: bash

	pip install git+https://github.com/kevin218/Eureka.git#egg=eureka[jwst]

where specific branches can be installed using:

.. code-block:: bash
	
	pip install git+https://github.com/kevin218/Eureka.git@mybranchname#egg=eureka[jwst]

If you desire any of the files in the `demos folder <https://github.com/kevin218/Eureka/tree/main/demos>`_, you will have to download these from
GitHub following the method described below.

With Git/GitHub
----------------

1. You can install Eureka! directly from source on `GitHub <http://github.com/kevin218/Eureka>`_ in one of two ways:

	a. On the `GitHub website <http://github.com/kevin218/Eureka>`_, click on **Code** and **Download ZIP** followed by unpacking the distribution by opening up a terminal and typing:

		.. code-block:: bash

			unzip Eureka-main.zip

	b. OR, clone the repository using ``git`` by typing:

		.. code-block:: bash

			git clone https://github.com/kevin218/Eureka.git

2. Navigate into the newly created directory and **install** Eureka! by running the following:

	.. code-block:: bash

		pip install .[jwst]

CRDS Environment Variables
--------------------------

Eureka! installs the JWST Calibration Pipeline as part of its requirements, and this also requires users to set the proper environment variables so that it can download the proper reference files needed to run the pipeline. 
For users not on the internal STScI network, two environment variables need to be set to enable this functionality. In your ``.zshrc`` (for Mac users) or ``.bashrc`` file (for bash users), or other shell initialization file, add these two lines (specifying your desired location to cache the CRDS files, e.g. ``/Users/your_name/crds_cache`` for Mac users or ``/home/your_name/crds_cache`` for Linux users):

	.. code-block:: bash

		export CRDS_PATH=/PATH/TO/FOLDER/crds_cache
		
		export CRDS_SERVER_URL=https://jwst-crds.stsci.edu

If these environment variables are not set, Stages 1-3 of the pipeline will fail.

Issues with installing the jwst dependency
------------------------------------------

If you have problems installing Eureka! and it seems to be centred around the installation of the jwst package, you can also install Eureka without
this requirement by removing the "[jwst]" part from the pip install lines above. This will, however, result in you being unable to run Eureka's
Stages 1 and 2 which simply offer ways of editing the behaviour of the jwst package's Stages 1 and 2.

For the JWST ERS Pre-Launch Data Hackathon
-----------------------------------------------

Check out the install instructions on the `ERS GitHub <https://github.com/ers-transit/hackathon-2021-day2>`_ if you want to use Eureka! during the hackathon.
