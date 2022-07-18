# $Author: patricio $
# $Revision: 304 $
# $Date: 2010-07-13 11:36:20 -0400 (Tue, 13 Jul 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/interp2d.py $
# $Id: interp2d.py 304 2010-07-13 15:36:20Z patricio $

# 2d interpolation using 1D interp twice

# must include or all or none of y,x,yi,xi

# iexpand*(np.array(sz)-1)+1 = np.array(sz)+(np.array(sz)-1)*(iexpand-1)

import numpy as np

def interp2d(image, expand=5, y=None, x=None, yi=None, xi=None):
  sz = np.shape(image)
  imagen = np.zeros( expand * (np.array(sz) - 1) + 1 )
  szi = np.shape(imagen)

  if y is None:
    y  = np.arange(sz[0])
    x  = np.arange(sz[1])
    yi = np.linspace(0, sz[0]-1, szi[0])
    xi = np.linspace(0, sz[1]-1, szi[1])

  for k in np.arange(sz[0]):
    imagen[k] = np.interp(xi, x, image[k])

  for k in np.arange(szi[1]):
    imagen[:, k] = np.interp(yi, y, imagen[0:sz[0], k])

  return imagen
