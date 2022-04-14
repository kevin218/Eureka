
⚡️ Quickstart ⚡️
================

Want to get up and running with ``Eureka!``, but not really sure where to begin? Keep reading! 

1. Installation 📦
------------------

The first thing you need to do is install the package, so if you haven't already, take a break from this page and follow the :ref:`installation` instructions (if you have issues be sure to visit the `FAQ page <https://eurekadocs.readthedocs.io/en/latest/installation.html#issues-installing-or-importing-jwst>`_ first). 


2. Download the data 💾
-----------------------------------

With the installation complete, you'll need some data to run ``Eureka!`` on. For now let's use some simulated data that was produced for the Transiting Exoplanet Community ERS Data Challenge. Datasets for all four instruments are available on the `STScI Box site <https://stsci.app.box.com/s/tj1jnivn9ekiyhecl5up7mkg8xrd1htl/folder/154382715453>`_, however, for the rest of this quickstart guide the `NIRCam Tiny dataset <https://stsci.app.box.com/s/tj1jnivn9ekiyhecl5up7mkg8xrd1htl/folder/156846571847>`_ will be used. 

Now, I'm sure you wouldn't just leave the data in your Downloads folder, but if so, let's make a new directory to store things. For example:

.. code-block:: bash

	mkdir /User/Data/JWST-Sim/NIRCam/
	cd /User/Data/JWST-Sim/NIRCam/
	unzip ~/Downloads/Stage2.zip -d .

Note that for Eureka! you do *not* need to download any ancillary data - any additional files will be downloaded automatically (if you correctly set the CRDS environment variables during installation). 


3. Set up your run directory 🗂
-------------------------------

3.1 Gather the demo files
~~~~~~~~~~~~~~~~~~~~~~~~

We're almost there, but before you can get things running you need to set up a directory for ``Eureka!`` to store both input and output files. 

.. code-block:: bash
	
	mkdir /User/DataAnalysis/JWST/MyFirstEureka
	cd /User/DataAnalysis/JWST/MyFirstEureka

From here, the simplest way to set up all of the Eureka input files is to duplicate them from the JWST demos directory in ``Eureka!``. This can be done using your existing installation:

.. code-block:: bash
	
	mkdir demos
	cp -r /User/OpenSourceProjects/Eureka/demos/JWST/* ./demos

Or, if you're lost in the depths of your conda installation, you can also download the demos folder directly `here <https://downgit.github.io/#/home?url=https://github.com/kevin218/Eureka/tree/main/demos/JWST>`_:

.. code-block:: bash

	mkdir demos
	unzip -j ~/Downloads/JWST.zip -d ./demos

This demos directory contains a selection of template files to run ``Eureka!``. There are three different types of files:
    
    -  ``*.ecf``: These are ``Eureka!`` control files, and contain input parameters required to run each stage of the pipeline. For more detail on the ecf parameters for each stage, see :ref:`here<ecf>`.
    -  ``*.epf``: This is a ``Eureka!`` parameter file, and describes the initial guesses and priors to be used when performing light curve fitting (Stage 5).
    -  ``run_eureka.py``: A script to run the ``Eureka!`` pipeline. 

3.2 Customise the demo files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You might notice that not all of the demo files will be applicable for every dataset, either because they are tailored to a specific instrument, or because they are for a ``Eureka!`` pipeline stage that precedes the input data. This is the case for the NIRCam data being used here, which as a ``*calints.fits`` file (more information on JWST pipeline products `here <https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/product_types.html>`_) has already been processed through an equivalent to Stage 1 and 2 of ``Eureka!``.

So, let's copy over the specific files needed to process this NIRCam dataset further. Given that the dataset contains a transit for WASP-39b, let's also change some of the default filenames to something a little more informative:

.. code-block::

	cp demos/run_eureka.py .
	cp demos/S3_nircam_wfss_template.ecf S3_wasp39b.ecf
	cp demos/S4_template.ecf S4_wasp39b.ecf
	cp demos/S5_template.ecf S5_wasp39b.ecf
	cp demos/S5_fit_par_template.epf S5_fit_par_wasp39b.ecf
	cp demos/S6_template.ecf S6_wasp39b.ecf

Notice that all of the ``*.ecf`` files have a common "wasp39b" string. It's useful to keep this homogenous across files as it is what ``Eureka!`` understands as an "event label", and is used to locate specific input files when running the pipeline. To see this more clearly, open up the ``run_eureka.py`` file and modify the ``eventlabel`` string directly:

.. code-block:: bash

        eventlabel = 'wasp39b'


Finally, we need to connect everything together by opening up each ``.ecf`` file and updating the ``topdir``, ``inputdir``, and ``outputdir`` parameters within. For the ``S3_wasp39b.ecf``, we want something like:

.. code-block:: bash

	topdir		/User/
	inputdir	/Data/JWST-Sim/NIRCam/Stage2
	outputdir	/DataAnalysis/JWST/MyFirstEureka/Stage3

However, for the later stages we can use something simpler, e.g. for the ``S4_wasp39b.ecf``:

.. code-block:: bash

	topdir		/User/DataAnalysis/JWST/MyFirstEureka/
	inputdir	Stage3
	outputdir	Stage4

The explicit settings for the ``S5_wasp39b.ecf`` and ``S6_wasp39b.ecf`` will be skipped here for brevity (but you should still do them!), although it's important to notice that you must also assign the correct ``.epf`` file in the ``S5_wasp39b.ecf``:

.. code-block:: bash

	fit_par		./S5_fit_par_wasp39b.epf

Whilst editing those files you will have noticed that there are a whole range of other inputs that can be tweaked and adjusted at each different stage. For now, we can ignore these as the demo files we've selected have been specifically tailored to this simulated dataset of WASP-39b.

4. Run Eureka! 💡
-----------------------------------------------------------------

4.1.  Now you're ready to run ``Eureka!``

Enter ``python run_eureka.py`` at the command prompt to run each stage in sequence. To start at a later stage, simply edit the ``run_eureka.py`` script and comment out the earlier stages. 

Stages 3 and later use metadata from the previous stages. If you wish to run each stage individually rather than sequentially, comment out the metadata argument from the function calls (e.g. remove the `` s2_meta=s2_meta`` argument) and ``Eureka!`` will automatically search for the metadata.

4.2. The code will run and save data and plots in a new directory set by the ``outputdir`` parameter in each ``.ecf`` file.
Below you see an example for a simulated spectrum which you should get after running the script and having ``is_plotsS3 = 3``:

.. image:: ../media/fig3301-1-Image+Background.png

