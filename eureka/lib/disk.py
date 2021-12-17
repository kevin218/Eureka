# $Author$
# $Revision$
# $Date$
# $HeadURL$
# $Id$

#+
# NAME:
#	DISK
#
# PURPOSE:
#	This function returns a byte array containing a disk.  The
#	disk is centered at x, y, and has radius r.  The array is nx,
#	ny in size and has byte type.  Pixel values of 1 indicate that
#	the center of a pixel is within r of x,y.  Pixel values of 0
#	indicate the opposite.  The center of each pixel is the
#	integer position of that pixel.
#
# CATEGORY:
#	Array creation.
#
# CALLING SEQUENCE:
#
#	diskarr = disk(R, X, Y, Nx, Ny)
#		or
#	diskarr = disk(R, Ctr, Sz)
#		which is the same as
#	diskarr = disk(R, [X, Y], [Nx, Ny])
#
# INPUTS:
#	There are two ways to enter the inputs.  If Nx is defined, then:
#	R:	radius of the disk, may be fractional.
#	X:	X position of the center of the disk, may be fractional.
#	Y:	Y position of the center of the disk, may be fractional.
#	Nx:	width of the output array.
#	Ny:	height of the output array.
#	Otherwise:
#	R:	Radius of the disk, may be fractional.
#	Ctr:	Position of the center of the disk, 2-element vector.
#	Sz:	Size of the output array, 2-element vector.
#
# KEYWORDS:
#	STATUS:	(returned) Set to 1 if any part of the disk is
#	outside the image boundaries.
#
# OUTPUTS:
#	This function returns a byte array as described above.
#
# PROCEDURE:
#	Each pixel is evaluated to see whether it is within r of
#	(x,y).  If so, the value is set to 1.  Otherwise it is 0.
#
# SIDE EFFECTS:
#	STATUS is set in caller.
#
# EXAMPLE:
#
#		tvscl, DISK(6.4, 9.9, 10.1, 30, 30)
#		tvscl, DISK(6.4, [9.9,  10.1], [30, 30], STATUS=STATUS)
#		print, STATUS
#		tvscl, DISK(6.4, [ 3,   10.1], [30, 30], STATUS=STATUS)
#		print, STATUS
#		tvscl, DISK(6.4, [27,   10.1], [30, 30], STATUS=STATUS)
#		print, STATUS
#		tvscl, DISK(6.4, [ 9.9,  3],   [30, 30], STATUS=STATUS)
#		print, STATUS
#		tvscl, DISK(6.4, [ 9.9, 27],   [30, 30], STATUS=STATUS)
#		print, STATUS
#
# MODIFICATION HISTORY:
# 	Written by:	Joseph Harrington, Cornell.  2003 April 4
#			jh@oobleck.astro.cornell.edu
#
#	2004 Feb 27	jh added alternate input method
#	2005 Nov 16	jh added STATUS, simplified disk calculation,
#				use double precision
#-

import numpy as np

def disk(r, ctr, size, status=False):

  # return status
  retstatus = status

  # check if disk is off image
  status = 0
  if (ctr[0] - r < 0 or ctr[0] + r > size[0]-1 or
      ctr[1] - r < 0 or ctr[1] + r > size[1]-1 ):
    status = 1

  # calculate pixel distance from center
  # print('disk size:',  size)
  ind = np.indices(size)
  fdisk = (ind[0]-ctr[0])**2.0 + (ind[1]-ctr[1])**2.0

  # return mask disk (and status if requested)
  ret = fdisk <= r**2.0
  if retstatus:
    ret = ret, status
  return ret
  