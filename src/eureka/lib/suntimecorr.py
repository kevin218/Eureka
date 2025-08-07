import numpy as np
import re
from scipy.constants import c
from .splinterp import splinterp


def getcoords(file):
    """Use regular expressions to extract X,Y,Z, and time values from the
    horizons file.

    Parameters
    ----------
    file : strs list
        A list containing the lines of a horizons file.

    Returns
    -------
    list
        A four elements list containing the X, Y, Z, and time arrays of
        values from file.

    Examples
    --------
    .. highlight:: python
    .. code-block:: python

        >>> start_data = '$$SOE'
        >>> end_data   = '$$EOE'

        >>> # Read in whole table as an list of strings, one string per line
        >>> ctable = open('/home/esp01/ancil/horizons/all_spitzer.vec', 'r')
        >>> wholetable = ctable.readlines()
        >>> ctable.close()

        >>> # Find start and end line
        >>> i = 0
        >>> while wholetable[i].find(end_data) == -1:
        >>>     if wholetable[i].find(start_data) != -1:
        >>>        start = i + 1
        >>>     i += 1

        >>> # Chop table
        >>> data = wholetable[start:i-2]

        >>> # Find values:
        >>> x, y, z, t = getcoords(data)

        >>> print(x, y, z, t)
    """
    x, y, z, time = [], [], [], []
    for i in np.arange(len(file)):
        # Use regular expressions to match strings enclosed between X,
        # Y, Z and end of line
        m = re.search(' X =(.*)Y =(.*) Z =(.*)\n', file[i])
        if m is not None:
            x.append(np.double(m.group(1)))
            y.append(np.double(m.group(2)))
            z.append(np.double(m.group(3)))
        # Match first word which is followed by ' = A'
        t = re.search('(.+) = A', file[i])
        if t is not None:
            time.append(np.double(t.group(1)))
    # return numpy arrays
    return np.array(x), np.array(y), np.array(z), np.array(time)


def suntimecorr(ra, dec, obst, coordtable, verbose=False):
    """This function calculates the light-travel time correction from
    observer to a standard location.  It uses the 2D coordinates (RA
    and DEC) of the object being observed and the 3D position of the
    observer relative to the standard location.  The latter (and the
    former, for solar-system objects) may be gotten from JPL's
    Horizons system.

    Parameters
    ----------
    ra : float
        Right ascension of target object in radians.
    dec : float
        Declination of target object in radians.
    obst : float or numpy float array
        Time of observation in Julian Date (may be a vector)
    coordtable : str
        Filename of output table from JPL HORIZONS specifying
        the position of the observatory relative to the
        standard position.
    verbose : bool
        If True, print X,Y,Z coordinates.

    Returns
    -------
    np.array
        This function returns the time correction in seconds to be ADDED
        to the observation time to get the time when the observed photons
        would have reached the plane perpendicular to their travel and
        containing the reference position.

    Examples
    --------
    .. highlight:: python
    .. code-block:: python

        >>> # Spitzer is in nearly the Earth's orbital plane.  Light coming
        >>> # from the north ecliptic pole should hit the observatory and
        >>> # the sun at about the same time.


        >>> import suntimecorr as sc
        >>> ra  = 18.0 * np.pi /  12 # ecliptic north pole coords in radians
        >>> dec = 66.5 * np.pi / 180 # "
        >>> obst = np.array([2453607.078])  # Julian date of 2005-08-24 14:00
        >>> print(sc.suntimecorr(
        >>>           ra, dec, obst,
        >>>           '/home/esp01/ancil/horizons/cs41_spitzer.vec'))
        1.00810877 # about 1 sec, close to zero

        >>> # If the object has the RA and DEC of Spitzer, light time should be
        >>> # about 8 minutes to the sun.
        >>> obs  = np.array([111093592.8346969, -97287023.315796047,
        >>>                  -42212080.826677799])
        >>> # vector to the object
        >>> obst = np.array([2453602.5])

        >>> print( np.sqrt(np.sum(obs**2.0)) )
        153585191.481 # about 1 AU, good
        >>> raobs  = np.arctan(obs[1]/ obs[0])
        >>> decobs = np.arctan(obs[2]/ np.sqrt(obs[0]**2 + obs[1]**2))
        >>> print(raobs, decobs)
        -0.7192383661, -0.2784282118
        >>> print(sc.suntimecorr(raobs, decobs, obst,
        >>>                      '/home/esp01/ancil/horizons/cs41_spitzer.vec')
        >>>       / 60.0)
        8.5228630 # good, about 8 minutes light time to travel 1 AU

    Notes
    -----
    The position vectors from coordtable are given in the following
    coordinate system:
    Reference epoch : J2000.0
    xy-plane : plane of the Earth's mean equator at the reference epoch
    x-axis : out along ascending node of instantaneous plane of the Earth's
    orbit and the Earth's mean equator at the reference epoch
    z-axis : along the Earth mean north pole at the reference epoch

    Ephemerides are often calculated for BJD, barycentric Julian date.
    That is, they are correct for observations taken at the solar
    system barycenter's distance from the target.  The BJD of our
    observation is the time the photons we observe would have crossed
    the sphere centered on the object and containing the barycenter.
    We must thus add the light-travel time from our observatory to
    this sphere.  For non-solar-system observations, we approximate
    the sphere as a plane, and calculate the dot product of the vector
    from the barycenter to the telescope and a unit vector to from the
    barycenter to the target, and divide by the speed of light.

    Properly, the coordinates should point from the standard location
    to the object.  Practically, for objects outside the solar system,
    the adjustment from, e.g., geocentric (RA-DEC) coordinates to
    barycentric coordinates has a negligible effect on the trig
    functions used in the routine.
    """
    start_data = '$$SOE'
    end_data = '$$EOE'

    # Read in whole table as an list of strings, one string per line
    ctable = open(coordtable, 'r')
    wholetable = ctable.readlines()
    ctable.close()

    # Find start and end line
    i = 0
    # while end has not been found:
    while wholetable[i].find(end_data) == -1:
        # if start is found get the index of next line:
        if wholetable[i].find(start_data) != -1:
            start = i + 1
        i += 1

    # Chop table
    data = wholetable[start:i-2]

    # Extract values:
    x, y, z, time = getcoords(data)

    # Interpolate to observing times:
    # We must preserve the shape and order of obst.  Spline takes
    # monotonic input and produces linear output.  x, y, z, time are
    # sorted as HORIZONS produces them.

    # Save shape of obst
    tshape = np.shape(obst)

    # Reshape to 1D and sort
    obstime = obst.flatten()
    ti = np.argsort(obstime)   # indexes of sorted array by time
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
        print('X, Y, Z = ', obsx, obsy, obsz)

    # Change ra and dec into unit vector n_hat
    object_unit_x = np.cos(dec) * np.cos(ra)
    object_unit_y = np.cos(dec) * np.sin(ra)
    object_unit_z = np.sin(dec)

    # Dot product the vectors with n_hat
    rdotnhat = (obsx * object_unit_x +
                obsy * object_unit_y +
                obsz * object_unit_z)

    # Reshape back to the original shape
    rdotnhat = rdotnhat.reshape(tshape)

    # Time correction is: dt = length/velocity
    # Divide by the speed of light and return
    return rdotnhat/(c/1000.0)
