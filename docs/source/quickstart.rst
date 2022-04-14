
⚡️ Quickstart ⚡️
================

Want to get up and running with ``Eureka!``, but not really sure where to begin? Keep reading! 

1. Installation 📦
------------------

The first thing you need to do is install the package, so if you haven't already, take a break from this page and follow the :ref:`installation` instructions (if you have issues be sure to visit the `FAQ page <https://eurekadocs.readthedocs.io/en/latest/installation.html#issues-installing-or-importing-jwst>`_ first). 


2. Download the data 💾
-----------------------------------

With the installation complete, you'll need some data to run ``Eureka!`` on. For now let's use some simulated data that was produced for the Transiting Exoplanet Community ERS Data Challenge. Datasets for all four instruments are available on the `STScI Box site <https://stsci.app.box.com/s/tj1jnivn9ekiyhecl5up7mkg8xrd1htl/folder/154382715453>`_, however, for the rest of this quickstart guide the `NIRCam Tiny dataset <https://stsci.app.box.com/s/tj1jnivn9ekiyhecl5up7mkg8xrd1htl/folder/156846571847>`_ will be used. 

Now, I'm sure you didn't just leave the data in your Downloads folder, but if so, let's make a new directory to store things. For example:

.. code-block:: bash

	mkdir /User/Data/JWST-Sim/NIRCam/
	cd /User/Data/JWST-Sim/NIRCam/
	mv ~/Downloads/Stage2 . 

Note that for Eureka! you do *not* need to download any ancillary data --- any additional files will be downloaded automatically (if you remembered to set the CRDS environment variables during installation!). 


3. Set up your run directory 
-----------------------------------------------------------------

3.1   Copy the JWST demos directory from your ``Eureka!`` installation to wherever you want to run your analysis. For example,

.. code-block:: bash

	cp -r /User/OpenSourceProjects/Eureka/demos/JWST /User/DataAnalysis/JWST/ERSDataChallenge

The JWST demos directory contains template files to run ``Eureka!``. There are three different types of files:
    
    -  ``Eureka!`` control files (with a .ecf extension). These files contain input parameters required to run each stage of the pipeline. For more detail on the ecf parameters for each stage, see :ref:`here<ecf>`.
    -  A ``fit_par`` file with initial guesses and priors for the light curve fits (Stage 5).
    -  A script to run the pipeline, ``run_eureka.py``. 


3.2  To set up a run, first decide which stage you want to start with. For example, if you download Stage 2 data, you will start with ``Eureka!`` Stage 3. 

Next, add an event label to the ecf filenames. If you're working with data for WASP-39b, the event label could be ``wasp39b``. Modify the file names of the templates that are appropriate for your stage and instrument mode. For example, if you're starting with Stage 2 NIRSPec data, you would update the following filenames: 

.. code-block::

	mv S3_nirspec_fs_template.ecf S3_wasp39b.ecf
	mv S4_template.ecf S4_wasp39b.ecf
	mv S5_template.ecf S5_wasp39b.ecf
	mv S6_template.ecf S6_wasp39b.ecf

Next, update the ``topdir``, ``inputdir``, and ``outputdir`` in each ``.ecf`` file. For each stage, the ``inputdir`` is the previous stage, e.g.:

.. code-block:: bash

	topdir /User/Data/JWST-Sim/NIRSpec/
	inputdir Stage2
	outputdir Stage3

Finally, update the event label in ``run_eureka.py``:

.. code-block:: bash

        eventlabel = 'wasp39b'

4. Run Eureka!
-----------------------------------------------------------------

4.1.  Now you're ready to run ``Eureka!``

Enter ``python run_eureka.py`` at the command prompt to run each stage in sequence. To start at a later stage, simply edit the ``run_eureka.py`` script and comment out the earlier stages. 

Stages 3 and later use metadata from the previous stages. If you wish to run each stage individually rather than sequentially, comment out the metadata argument from the function calls (e.g. remove the `` s2_meta=s2_meta`` argument) and ``Eureka!`` will automatically search for the metadata.

4.2. The code will run and save data and plots in a new directory set by the ``outputdir`` parameter in each ``.ecf`` file.
Below you see an example for a simulated spectrum which you should get after running the script and having ``is_plotsS3 = 3``:

.. image:: ../media/fig3301-1-Image+Background.png

