#! /usr/bin env python
#Converts UTC Julian dates to Terrestrial Time and Barycentric Dynamical Time Julian dates
#Author: Ryan A. Hardy, hardy.r@gmail.com
#Last update: 2018-12
import numpy as np
import urllib
import os
import re
import time
#from univ import c
from .splinterp import splinterp
import scipy.interpolate as si
ftpurl  = 'ftp://ftp.nist.gov/pub/time/leap-seconds.list'
ftpurl2 = 'ftp://ftp.boulder.nist.gov/pub/time/leap-seconds.list'

def leapdates(rundir):
	'''Generates an array of leap second dates which
	are automatically updated every six months.
	Uses local leap second file, but retrieves a leap
	second file from NIST if the current file is out of date.
	'''
	try:
		files = os.listdir(rundir)
		recent = np.sort(files)[-1]
		nist = open(rundir+recent, 'r')
		doc = nist.read()
		nist.close()
		table = doc.split('#@')[1].split('\n#\n')[1].split('\n')
		expiration = np.float(doc.split('#@')[1].split('\n')[0][1:])
		ntpepoch = 2208988800
		if time.time()+ ntpepoch > expiration:
			print("Leap-second file expired.	Retrieving new file.")
			try:
				nist = urllib.request.urlopen(ftpurl)
				#print('Leap-second ftp 1 worked')
			except:
				try:
					nist = urllib.request.urlopen(ftpurl2)
					#print('Leap-second ftp 2 worked')
				except:
					print('NIST leap-second file not available.	Using stored table.')
			doc = nist.read().decode('utf-8')
			nist.close()
			newexp = doc.split('#@')[1].split('\r\n')[0][1:]
			newfile = open(rundir+"leap-seconds."+newexp, 'w')
			newfile.write(doc)
			newfile.close()
			table = doc.split('#@')[1].split('\r\n#\r\n')[1].split('\r\n')
			print("Leap second file updated.")
		else:
			print("Local leap second file retrieved.")
			print("Next update: "+time.asctime( time.localtime(expiration-ntpepoch)))
		ls = np.zeros(len(table))
		for i in range(len(table)):
			ls[i] = np.float(table[i].split('\t')[0])
		jd = ls/86400+2415020.5
		return jd
	except:
		print('NIST leap-second file not available.	Using stored table.')

		return np.array([2441316.5,
		2441682.5,
		2442047.5,
		2442412.5,
		2442777.5,
		2443143.5,
		2443508.5,
		2443873.5,
		2444238.5,
		2444785.5,
		2445150.5,
		2445515.5,
		2446246.5,
		2447160.5,
		2447891.5,
		2448256.5,
		2448803.5,
		2449168.5,
		2449533.5,
		2450082.5,
		2450629.5,
		2451178.5,
		2453735.5,
		2454831.5])+1

def leapseconds(jd_utc, dates):
		'''Computes the difference between UTC and TT for a given date.
		jd_utc	=	 (float) UTC Julian date
	dates	=	(array_like) an array of Julian dates on which leap seconds occur'''
		utc_tai = len(np.where(jd_utc > dates)[0])+10-1
		tt_tai = 32.184
		return tt_tai + utc_tai

def utc_tt(jd_utc,rundir):
		'''Converts UTC Julian dates to Terrestrial Time (TT).
		jd_utc	=	 (array-like) UTC Julian date'''
		dates = leapdates(rundir)
		if len(jd_utc) > 1:
				dt = np.zeros(len(jd_utc))
				for i in range(len(jd_utc)):
						 dt[i]	= leapseconds(jd_utc[i], dates)
		else:
				dt = leapseconds(jd_utc, dates)
		return jd_utc+dt/86400.

def utc_tdb(jd_utc,rundir):
	'''Converts UTC Julian dates to Barycentric Dynamical Time (TDB).
	Formula taken from USNO Circular 179, based on that found in Fairhead and Bretagnon (1990).	Accurate to 10 microseconds.
	jd_utc	=	 (array-like) UTC Julian date

	'''
	jd_tt = utc_tt(jd_utc,rundir)
	T =	(jd_tt-2451545.)/36525
	jd_tdb = jd_tt + (0.001657*np.sin(628.3076*T + 6.2401)
	+ 0.000022*np.sin(575.3385*T 	+	 4.2970)
	+ 0.000014*np.sin(1256.6152*T 	+	 6.1969)
	+ 0.000005*np.sin(606.9777*T 	+	 4.0212)
	+ 0.000005*np.sin(52.9691*T 	+	 0.4444)
	+ 0.000002*np.sin(21.3299*T 	+	 5.5431)
	+ 0.000010*T*np.sin(628.3076*T 	+ 	4.2490))/86400.
	return jd_tdb

def bjdcorr(date, ra, dec, location="s"):
	#Wrapper for suntimecorr
	horizonsdir = "/home/esp01/ancil/horizons/"
	if location == "s":
		tabfile = "all_spitzer.vec"
	elif location == "g":
		tabfile = "all_geo.vec"
	elif location == "h":
		tabfile = "all_sun.vec"
	bjd = suntimecorr(hms_rad(ra), dms_rad(dec), date, horizonsdir+tabfile)
	return bjd

def hms_rad(params):
    hour, minute, second = params
    #Converts right ascension from hh:mm:ss to radians
    return (hour + minute/60. + second/3600.)*np.pi/12

def dms_rad(params):
    degree, minute, second = params
    #Converts declination from dd:mm:ss to radians
    return (np.abs(degree) + minute/60. + second/3600.)*np.sign(degree)*np.pi/180


def suntimecorr(ra, dec, obst,	coordtable, verbose=False):
	#+
	# NAME:
	#			 SUNTIMECORR
	#
	# PURPOSE:

	#			 This function calculates the light-travel time correction from
	#			 observer to a standard location.	It uses the 2D coordinates
	#			 (RA and DEC) of the object being observed and the 3D position
	#			 of the observer relative to the standard location.	The latter
	#			 (and the former, for solar-system objects) may be gotten from
	#			 JPL's Horizons system.
	#
	# CATEGORY:
	#			 Astronomical data analysis
	#
	# CALLING SEQUENCE:
	#			 time = SUNTIMECORR(Ra, Dec, Obst, Coordtable)
	#
	# INPUTS:
	#			 Ra:		 Right ascension of target object in RADIANS
	#			 Dec:		Declination of target object in RADIANS
	#			 Obst:	 Time of observation in Julian Date (may be a vector)
	#			 Coordtable:		Filename of output table from JPL HORIZONS
	#											specifying the position of the observatory
	#											relative to the standard position.	The
	#											HORIZONS request should be in the form of the
	#											following example, with a subject line of JOB:
	#
	#!$$SOF
	#!
	#! Example e-mail command file. If mailed to "horizons@ssd.jpl.nasa.gov"
	#! with subject "JOB", results will be mailed back.
	#!
	#! This example demonstrates a subset of functions. See main doc for
	#! full explanation. Send blank e-mail with subject "BATCH-LONG" to
	#! horizons@ssd.jpl.nasa.gov for complete example.
	#!
	# EMAIL_ADDR = 'shl35@cornell.edu'			! Send output to this address
	#																			 !	(can be blank for auto-reply)
	# COMMAND		= '-79'									! Target body, closest apparition
	#
	# OBJ_DATA	 = 'YES'										! No summary of target body data
	# MAKE_EPHEM = 'YES'										! Make an ephemeris
	#
	# START_TIME	= '2005-Aug-24 06:00'		 ! Start of table (UTC default)
	# STOP_TIME	 = '2005-Aug-25 02:00'		 ! End of table
	# STEP_SIZE	 = '1 hour'								 ! Table step-size
	#
	# TABLE_TYPE = 'VECTOR'						! Specify VECTOR ephemeris table type
	# CENTER		 = '@10'								 ! Set observer (coordinate center)
	# REF_PLANE	= 'FRAME'									! J2000 equatorial plane
	#
	# VECT_TABLE = '3'											! Selects output type (3=all).
	#
	# OUT_UNITS	= 'KM-S'									 ! Vector units# KM-S, AU-D, KM-D
	# CSV_FORMAT = 'NO'										 ! Comma-separated output (YES/NO)
	# VEC_LABELS = 'YES'										! Label vectors in output (YES/NO)
	# VECT_CORR	= 'NONE'									 ! Correct for light-time (LT),
	#																			 !	or lt + stellar aberration (LT+S),
	#																			 !	or (NONE) return geometric
	#																			 !	vectors only.
	#!$$EOF
	#
	# KEYWORD PARAMETERS:
	#
	#	All keywords are returned, modifying the named variable IN THE CALLER.
	#
	#			 X:			X component of position vectors (km) extracted from COORDTABLE
	#			 Y:			Y component of position vectors (km) extracted from COORDTABLE
	#			 Z:			Z component of position vectors (km) extracted from COORDTABLE
	#			 TIME:	 times (in Julian Date) extracted from COORDTABLE
	#			 OBSX:	 X component of position vector (km) at OBST, found
	#							 by spline interpolation of X and TIME, in shape of OBST.
	#			 OBSY:	 Y component of position vector (km) at OBST, found
	#							 by spline interpolation of Y and TIME, in shape of OBST.
	#			 OBSZ:	 Z component of position vector (km) at OBST, found
	#							 by spline interpolation of Z and TIME, in shape of OBST.
	#
	#	The position vectors are given in the following coordinate system:
	#		Reference epoch: J2000.0
	#		xy-plane: plane of the Earth's mean equator at the reference epoch
	#		x-axis	: out along ascending node of instantaneous plane of the Earth's
	#							orbit and the Earth's mean equator at the reference epoch
	#		z-axis	: along the Earth mean north pole at the reference epoch
	#
	# OUTPUTS:
	#			 This function returns the time correction in seconds to be
	#			 ADDED to the observation time to get the time when the
	#			 observed photons would have reached the plane perpendicular to
	#			 their travel and containing the reference position.
	#
	# SIDE EFFECTS:
	#	The keyword parameters change data IN THE CALLER.
	#
	# PROCEDURE:
	#			 Ephemerides are often calculated for BJD, barycentric Julian
	#			 date.	That is, they are correct for observations taken at the
	#			 solar system barycenter's distance from the target.	The BJD
	#			 of our observation is the time the photons we observe would
	#			 have crossed the sphere centered on the object and containing
	#			 the barycenter.	We must thus add the light-travel time from
	#			 our observatory to this sphere.	For non-solar-system
	#			 observations, we approximate the sphere as a plane, and
	#			 calculate the dot product of the vector from the barycenter to
	#			 the telescope and a unit vector to from the barycenter to the
	#			 target, and divide by the speed of light.
	#
	#			 Properly, the coordinates should point from the standard
	#			 location to the object.	Practically, for objects outside the
	#			 solar system, the adjustment from, e.g., geocentric (RA-DEC)
	#			 coordinates to barycentric coordinates has a negligible effect
	#			 on the trig functions used in the routine.
	#
	# EXAMPLE:
	#
	# Spitzer is in nearly the Earth's orbital plane.	Light coming from
	# the north ecliptic pole should hit the observatory and the sun at
	# about the same time.
	#
	# Ra	= 18d	 * !dpi /	12d # coordinates of ecliptic north pole in radians
	# Dec = 66.5d * !dpi / 180d # "
	# Obst = 2453607.078d			 # Julian date of 2005-08-24 14:00
	# print, SUNTIMECORR(Ra, Dec, Obst, 'cs41_spitzer.vec', $
	#								x = x, y = y, z = z, $
	#								obsX = obsX, obsY = obsY, obsz = obsZ, $
	#								time = time)
	# #			 1.0665891 # about 1 sec, close to zero
	#
	# # If the object has the RA and DEC of Spitzer, light time should be
	# # about 8 minutes to the sun.
	# obs	= [x[0], y[0], z[0]] # vector to the object
	# obst = time[0]
	# print, sqrt(total(obs^2))
	# #	 1.5330308e+08 # about 1 AU, good
	# raobs	= atan(obs[1], obs[0])
	# decobs = atan(obs[2], sqrt(obs[0]^2 + obs[1]^2))
	# print, raobs, decobs
	# #		 -0.65427333		 -0.25659940
	# print, 1d / 60d * SUNTIMECORR(Raobs, Decobs, Obst, 'cs41_spitzer.vec', $
	#								x = x, y = y, z = z, $
	#								obsX = obsX, obsY = obsY, obsz = obsZ, $
	#								time = time)
	##				8.5228630 # good, about 8 minutes light time to travel 1 AU
	#
	# MODIFICATION HISTORY:
	#			 Written by:		 Statia Luszcz 12/2005
	#	2006-03-09 jh	Corrected 90deg error in algorithm, renamed,
	#			updated header, made Coordtable a positional
	#			arg since it's required, switched to radians.
	#	2007-06-28 jh	Renamed to suntimecorr since we now use
	#			barycentric Julian date.
	#	2009-01-28 jh	 Change variables to long, use spline instead
	#			of linfit so we can use one HORIZONS file for
	#			the whole mission.
	#	2009-02-22 jh	 Reshape spline results to shape of obst.	Make
	#			it handle unsorted unput data properly.
	#			Header update.
	#	2011-12-26 rhardy Moved function to utc_tt.py

	start_data = '$$SOE'
	end_data	 = '$$EOE'

	# Read in whole table as an list of strings, one string per line
	ctable = open(coordtable, 'r')
	wholetable = ctable.readlines()
	ctable.close()

	# Find startline
	search = -1
	i = 0
	while search == -1:
		search = wholetable[i].find(start_data)
		i += 1

	# Find endline
	search = -1
	j = 0
	while search == -1:
		search = wholetable[j].find(end_data)
		j += 1

	# Chop table
	data = wholetable[i:j-2]
	datalen = len(data)
	n_entries = int(datalen / 4)

	# Times are entries 0, 4, 8 etc.
	# The first 'word' per line is the time in JD
	time = np.zeros(n_entries)
	for i in np.arange(n_entries):
		time[i] = np.double(data[i*4].split()[0])

		# FINDME: dont hardcode 22
	# Coords (X,Y,Z) are entries 1, 5, 9, etc.
	x = np.zeros(n_entries)
	y = np.zeros(n_entries)
	z = np.zeros(n_entries)
	leng = 22 # numbers length in the horizon file
	xstart = data[1].find('X') + 3
	ystart = data[1].find('Y') + 3
	zstart = data[1].find('Z') + 3
	for i in np.arange(n_entries):
		line = data[i*4+1]
		x[i] = np.double(line[xstart: xstart + leng])
		y[i] = np.double(line[ystart: ystart + leng])
		z[i] = np.double(line[zstart: zstart + leng])


	# interpolate to observing times
	# We must preserve the shape and order of obst.	Spline takes
	# monotonic input and produces linear output.	x, y, z, time are
	# sorted as HORIZONS produces them.

	# Get shape of obst
	tshape	= np.shape(obst)

	# Reshape to 1D and sort
	# FINDME: use .flat/.flatten
	obstime = obst.reshape(-1)
	ti = np.argsort(obstime)
	tsize = np.size(obstime)

	# Allocate output arrays
	obsx = np.zeros(tsize)
	obsy = np.zeros(tsize)
	obsz = np.zeros(tsize)

	# Interpolate sorted arrays
	obsx[ti] = splinterp(obstime[ti], time, x)
	obsy[ti] = splinterp(obstime[ti], time, y)
	obsz[ti] = splinterp(obstime[ti], time, z)

	if verbose:
		print( 'X, Y, Z = ', obsx, obsy, obsz)

	# Change ra and dec into unit vector n_hat
	object_unit_x = np.cos(dec) * np.cos(ra)
	object_unit_y = np.cos(dec) * np.sin(ra)
	object_unit_z = np.sin(dec)

	# Dot product the vectors with n_hat
	rdotnhat = ( obsx * object_unit_x +
							 obsy * object_unit_y +
							 obsz * object_unit_z	)

	# Reshape back to the original shape
	rdotnhat = rdotnhat.reshape(tshape)

	# Divide by the speed of light and return

	# FINDME check it works right.
	return rdotnhat / 299792.458

def splinterp(x2, x, y):

	""" This function implements the methods splrep and splev of the
	module scipy.interpolate


	Parameters
	----------
	X2: 1D array_like
	  array of points at which to return the value of the
	  smoothed spline or its derivatives

	X, Y: array_like
		The data points defining a curve y = f(x).

	Returns
	-------
	an array of values representing the spline function or curve.
	If tck was returned from splrep, then this is a list of arrays
	representing the curve in N-dimensional space.


	Examples
	--------
	>>> import numpy as np
	>>> import matplotlib.pyplot as plt

	>>> x = np.arange(21)/20.0 * 2.0 * np.pi
	>>> y = np.sin(x)
	>>> x2 = np.arange(41)/40.0 *2.0 * np.pi

	>>> y2 = splinterp(x2, x, y)
	>>> plt.plot(x2,y2)


	"""

	tck = si.splrep(x, y)
	y2  = si.splev(x2, tck)
	return y2
	