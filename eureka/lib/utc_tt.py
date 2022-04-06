#! /usr/bin env python
#Converts UTC Julian dates to Terrestrial Time and Barycentric Dynamical Time Julian dates
#Author: Ryan A. Hardy, hardy.r@gmail.com
#Last update: 2011-03-17
import numpy as np
import urllib
import os
import re
import time
import scipy.interpolate as si

def leapdates(rundir):
	'''Generates an array of leap second dates which
	are automatically updated every six months.	
	Uses local leap second file, but retrieves a leap 
	second file from NIST if the current file is out of date.
	Last update: 2011-03-17'''
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
			#nist = urllib.urlopen('ftp://utcnist.colorado.edu/pub/leap-seconds.list')
			nist = urllib.request.urlopen('ftp://ftp.boulder.nist.gov/pub/time/leap-seconds.list')
			doc = nist.read()
			nist.close()
			newexp = doc.split('#@')[1].split('\n')[0][1:]
			newfile = open(rundir+"leap-seconds."+newexp, 'w')
			newfile.write(doc)
			newfile.close()
			table = doc.split('#@')[1].split('\n#\n')[1].split('\n')
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

def utc_tt(jd_utc, leapdir):
	'''Converts UTC Julian dates to Terrestrial Time (TT).
	jd_utc	=	 (array-like) UTC Julian date'''
	dates = leapdates(leapdir)
	if len(jd_utc) > 1:
		dt = np.zeros(len(jd_utc))
		for i in range(len(jd_utc)):
			dt[i]	= leapseconds(jd_utc[i], dates)
	else:
		dt = leapseconds(jd_utc, dates)
	return jd_utc+dt/86400.

def utc_tdb(jd_utc, leapdir):
	'''Converts UTC Julian dates to Barycentric Dynamical Time (TDB).
	Formula taken from USNO Circular 179, based on that found in Fairhead and Bretagnon (1990).	Accurate to 10 microseconds.
	jd_utc	=	 (array-like) UTC Julian date
	
	'''
	jd_tt = utc_tt(jd_utc, leapdir)
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
