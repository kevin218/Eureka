# $Author: patricio $
# $Revision: 288 $
# $Date: 2010-06-24 03:33:44 -0400 (Thu, 24 Jun 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/manageevent.py $
# $Id: manageevent.py 288 2010-06-24 07:33:44Z patricio $

import h5py as h5
try:
    import cPickle as pickle
except:
    import _pickle as pickle

"""
    Name
    ----
    Manage Event

    File
    ----
    manageevnet.py

    Description
    -----------
    Routines for handling events.

    Package Contents
    ----------------
    saveevent(event, filename, save=['event'], delete=[])
        Saves an event in .dat (using cpickle) and .h5 (using h5py) files.

    loadevent(filename, load):
        Loads an event stored in .dat and .h5 files.

    updateevent(event, filename, add):
        Adds parameters given by add from filename to event.


    Examples:
    ---------
    >>> from manageevent import *
    >>> # Save  hd209bs51_ini.dat and hd209bs51_ini.h5 files.

    >>> saveevent(event, 'd209bs51_ini', save=['data', 'head','uncd',
                                            'bdmskd'])

    >>> # Load the event and its data frames
    >>> event = loadevent('hd209bs51_ini', ['data'])

    >>> # Load uncd and bdmsk into event:
    >>> updateevent(event, 'hd209bs51_ini', ['uncd', 'bdmskd'])

    Revisions
    ---------
    2010-07-10  patricio  joined loadevent and       pcubillos@fulbrightmail.org
                          saveevent into this package.
                          updateevent added.
    2010-11-12  patricio  reimplemented using exec()
"""


def saveevent(event, filename, save=[], delete=[], protocol=3):
  """
    Saves an event in .dat (using cpickle) and .h5 (using h5py) files.

    Parameters
    ----------
    event    : An Event instance.
    filename : String
               The string contains the name of the event file.
    save     : String tuple
               The elements of this tuple contain the parameters to save.
               We usually use the values: 'data', 'uncd', 'head', 'bdmskd',
               'brmksd' or 'mask'.
    delete   : String tuple
               Parameters to be deleted.

    Notes
    -----
    The input filename should not have the .dat nor the .h5 extentions.
    Side effect: This routine deletes all parameters except 'event'
                 after saving it.

    Returns
    -------

    Examples
    --------
    See package example.

    Revisions
    ---------
    2010-07-10  patricio  Added documentation.     pcubillos@fulbrightmail.org
  """

  if save != []:
    handle = h5.File(filename + '.h5', 'w')
    for param in save:
      exec('handle["'  + param + '"] = event.' + param)
      exec('del(event.' + param + ')')
      # calibration data
      if event.havecalaor:
        exec('handle["pre'   + param + '"] = event.pre'  + param)
        exec('handle["post'  + param + '"] = event.post' + param)
        exec('del(event.pre' + param + ', event.post'    + param + ')')
    handle.close()

  # delete if requested
  for param in delete:
    exec('del(event.' + param + ')')
    if event.havecalaor:
      exec('del(event.pre' + param + ', event.post' + param + ')')

  # Pickle-Save the event
  handle = open(filename + '.dat', 'wb')
  pickle.dump(event, handle, protocol)
  handle.close()


def loadevent(filename, load=[], loadfilename=None):
  """
    Loads an event stored in .dat and .h5 files.

    Parameters
    ----------
    filename : String
               The string contains the name of the event file.
    load     : String tuple
               The elements of this tuple contain the parameters to read.
               We usually use the values: 'data', 'uncd', 'head', 'bdmskd',
               'brmskd' or 'mask'.

    Notes
    -----
    The input filename should not have the .dat nor the .h5 extentions.

    Returns
    -------
    This function return an Event instance.

    Examples
    --------
    See package example.

    Revisions
    ---------
    2010-07-10  patricio  Added documentation.     pcubillos@fulbrightmail.org
  """
  from astropy.io import fits
  handle = open(filename + '.dat', 'rb')
  event = pickle.load(handle, encoding='latin1')
  handle.close()

  if loadfilename == None:
    loadfilename = filename

  if load != []:
    handle = h5.File(loadfilename + '.h5', 'r')
    for param in load:
      exec('event.' + param + ' = handle["' + param + '"][:]')
      # calibration data:
      if event.havecalaor:
        exec('event.pre'  + param + ' = handle["pre'  + param + '"][:]')
        exec('event.post' + param + ' = handle["post' + param + '"][:]')

    handle.close()

  return event


def updateevent(event, filename, add):
  """
    Adds parameters given by add from filename to event.

    Parameters
    ----------
    event    : An Event instance.
    filename : String
               The string contains the name of the event file.
    add      : String tuple
               The elements of this tuple contain the parameters to
               add.  We usually use the values: 'data', 'uncd', 'head',
               'bdmskd', 'brmaskd' or 'mask'.

    Notes
    -----
    The input filename should not have the .dat nor the .h5 extentions.

    Returns
    -------
    This function return an Event instance.

    Examples
    --------
    See package example.

    Revisions
    ---------
    2010-07-10  patricio  Created.     pcubillos@fulbrightmail.org
  """
  event2 = loadevent(filename, load=add)

  for param in add:
    exec('event.' + param + ' = event2.' + param)
    # calibration data
    if event.havecalaor:
      exec('event.pre'  + param + ' = event2.pre'  + param)
      exec('event.post' + param + ' = event2.post' + param)

  return event
