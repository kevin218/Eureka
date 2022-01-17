
Quickstart
============

1. Installation and requirements
-----------------------------------

Follow the instructions on the :ref:`installation` page.


2. Download data
-----------------------------------

2.1. Make a new directory on your computer to store the **simulated data and ancillary files**. E.g.:

.. code-block:: bash

	mkdir /User/Data/JWST-Sim/NIRCam/
	cd /User/Data/JWST-Sim/NIRCam/
	mkdir Stage2
	mkdir ancil

2.2. Download the `simulated NIRCam data <https://stsci.app.box.com/s/8r6kqh9m53jkwkff0scmed6zx42g307e/folder/136379342485>`_ from the STScI Box site and save the files in the `Stage2` directory.
You only need the fits files ending with the suffix ``*calints.fits``. The files are large (5 GB total) so the download may take a while.
If your internet connection is slow, download the `smallest file only <https://stsci.app.box.com/s/8r6kqh9m53jkwkff0scmed6zx42g307e/file/809097167084>`_  and the tutorial will still work.

2.3. Save the `NIRCam calibration data <https://github.com/ers-transit/hackathon-2021-day2/tree/main/ancil_files/NIRCam>`_ in the `ancil` directory.

3. Setup the Eureka! control file (.ecf) and run_eureka.py
-----------------------------------------------------------------

3.1 Go into the downloaded ``Eureka!`` directory (it is likely called Eureka-main if you downloaded it from GitHub) and open the file ``Eureka-main/demos/S3/S3_template.ecf``.
Update "topdir + inputdir" and "topdir + ancildir" to the location of your Stage2 data and the ancil data, respectively.

You can get more information about the ecf (``Eureka!`` control file) :ref:`here<ecf>`.

3.2 As the simulated data is containing a transit of WASP-43b, let's call the event 'wasp43b'.
If you have a look into the ``run_eureka.py`` script, this has been already been set for you (``eventlabel = 'wasp43b'``).
In order for Eureka! to find the control files, you have to change their names:

.. code-block::

	S3_template.ecf --> S3_wasp43b.ecf
	S4_template.ecf --> S4_wasp43b.ecf



1. Run Eureka!
-----------------------------------------------------------------

4.1. Now execute the ``run_eureka.py`` script by typing:

.. code-block:: bash

	python run_eureka.py


If your data directory contains all 21 simulated Stage 2 NIRCam Data segments, this can take around 25 minutes. For a quick look set ``testing`` in the S3 ecf to ``True``.
This will only reduce and analyze the last segment (=smallest file) in your Stage 2 Data directory.

Note: If you run into a ``matplotlib`` error, you might want to install ``sudo apt install cm-super`` and try it again.

4.2. The code will run and save data and plots in a new directory in ``demos/S3/``.
Below you see an example for a simulated spectrum which you should get after running the script and having ``is_plotsS3 = 3``:

.. image:: ../media/fig3301-1-Image+Background.png


