# Eureka!

**Welcome to Eureka!**

ALERT: Project Eureka! is currently under heavy development. Use at your own risk.

## Installation

### With pip

The simplest way to install the Eureka! package is with the following one-line command:

```bash
pip install git+git://github.com/kevin218/Eureka.git
```

where specific branches can be installed using:

```bash
pip install git+git://github.com/kevin218/Eureka.git@mybranchname
```

If you desire any of the files in the [demos folder](https://github.com/kevin218/Eureka/tree/main/demos), you will have to download these from
GitHub following the method described below.

### With Git/GitHub

1. You can install Eureka! directly from source on [GitHub](http://github.com/kevin218/Eureka) in one of two ways:
	- On the [GitHub website](http://github.com/kevin218/Eureka), click on **Code** and **Download ZIP** followed by unpacking the distribution by opening up a terminal and typing:

		```bash
		unzip Eureka-main.zip
		```

	- OR, clone the repository using ``git`` by typing:

		```bash
		git clone https://github.com/kevin218/Eureka.git
		```

2. Navigate into the newly created directory and **install** Eureka! by running the following:

	```bash
	pip install .
	```

3. Set JWST CRDS environment variables. Eureka! installs the JWST Calibration Pipeline, and these variables need to be set for the calibration steps to properly run. The best practice for doing this is as follows: 

In your ``.zshrc`` (for Mac users) or ``.bashrc`` file (for bash users), or other shell initialization file, add these two lines (specifying your desired location to cache the CRDS files, e.g. ``/Users/your_name/crds_cache`` for Mac users or ``/home/your_name/crds_cache`` for Linux users):

	```bash
	export CRDS_PATH=/PATH/TO/FOLDER/crds_cache	
	export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
	```

## Documentation

Check out the docs at [https://eurekadocs.readthedocs.io](https://eurekadocs.readthedocs.io).
