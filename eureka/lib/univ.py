# $Author: carthik $
# $Revision: 267 $
# $Date: 2010-06-08 22:33:22 -0400 (Tue, 08 Jun 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/univ.py $
# $Id: univ.py 267 2010-06-09 02:33:22Z carthik $

import numpy as np
from scipy import constants

class Univ:
  """  
  Set universal constants in info structure conversions.
  """

  def __init__(self):
      # steradians per square arcsecond
      self.srperas = constants.arcsec**2.0
      # from MDH 2.1, p. 25., sect. 3.7.1: 2.35044e-11

      # Micro Jansky to mks
      # 1d-6  converts uJy to Jy, 1d-26 converts Jy to W m^-2 Hz^-1
      self.ujy2mks = 1e-32

      # Time
      self.mjdoff = 2400000.5

      # Julian date of J2000.0 = 1.5 Jan 2000 (see ESAA ch 27)
      self.j2kjd  = 2451545.0 

      # Julian date of January first 1980
      self.jdjf80 = 2444239.49960294  

      # meters in a parsec
      self.mppc      = 3.0856776e16  # from AllenII, p.12

      # Units                             # Source
      # Solar luminosity in Watts
      self.lsun      =        3.827e+26   # Wikipedia
      # Solar radius in meters
      self.rsun      =     695508.0e+03   # +- 26 km, AllenII
      # Jupiter radius in meters
      self.rjup      =      71492.0e+03   # AllenII
      # astronomical unit in meters
      self.au        = 1.4959787066e+11   # m, AllenII
      # Stefan-Boltzmann's constant in J m^-2 s^-1 K^-4
      self.stefboltz = constants.sigma
      # speed of light in meters per second
      self.c         = constants.c
      # Plank's constant in Joule per second
      self.h         = constants.h
      # Bolzmann's constant in Joules per Kelvin
      self.k         = constants.k


