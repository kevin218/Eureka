# $Author: patricio $
# $Revision: 267 $
# $Date: 2010-06-08 22:33:22 -0400 (Tue, 08 Jun 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/reader3.py $
# $Id: reader3.py 267 2010-06-09 02:33:22Z patricio $

import numpy as np

"""
    This class loads a control file (pcf) and lets you
    querry the parameters and values.

    Constructor Parameters:
    -----------------------
    file : A control file containing the parameters and values.

    Notes:
    ------
    A parameter can have one or more values, differet parameters can
    have different number of values.

    The function Param.get(index) automatically interprets the type of the
    values. If they can be cast into a numeric value retuns a numeric
    value, otherwise returns a string.

    Examples:
    --------
    >>> # Load a pcf file:
    >>> import reader3 as rd
    >>> reload(rd)
    >>> pcf = rd.Pcffile('/home/patricio/ast/esp01/anal/wa011bs11/run/wa011bs11.pcf')

    >>> Each parameter has the attribute value, wich is a ndarray:
    >>> pcf.planet.value
    array(['wa011b'],
          dtype='|S6')

    >>> # To get the n-th value of a parameter use pcffile.param.get(n):
    >>> # if it can't be converted to a number/bool/etc, it returns a string.
    >>> pcf.planet.get(0)
    'wa011b'
    >>> pcf.photchan.get(0)
    1
    >>> pcf.fluxunits.get(0)
    True

    >>> # Use pcffile.param.value[n] to get the n-th value as string:
    >>> pcf.aorname.get(0)
    38807808
    >>> pcf.aorname.value[0]
    '38807808'

    >>> # The function pcffile.param.getarr() returns the numeric/bool/etc
    >>> # values of a parameter as a nparray:
    >>> pcf.sigma.value
    array(['4.0', '4.0'],
          dtype='|S5')
    >>> pcf.sigma.getarr()
    array([4.0, 4.0], dtype=object)


    Modification History:
    --------------------
    2009-01-02 chris      Initial Version.
                          by Christopher Campo      ccampo@gmail.com
    2010-03-08 patricio   Modified from ccampo version.
                          by Patricio Cubillos      pcubillos@fulbrightmail.org
    2010-10-27 patricio   Docstring updated
    2011-02-12 patricio   Merged with ccampo's tepclass.py

"""


# each parameter is an instance of this class
class Param:
  # constructor
  def __init__(self, vals):
    self.value = vals

  def get(self, index=0):
    """
    Return a numeric/boolean/None/etc. value if possible, else return a string.
    """
    try:
      return eval(self.value[index])
    except:
      return self.value[index]

  def getarr(self):
    length = np.size(self.value)
    ret = np.zeros(length, dtype='object')
    for i in np.arange(length):
      ret[i] = self.get(i)
    return ret


class Pcf:

  def __init__(self, params):

  # :::: And now, Chris' code ::::
  # load all parameters
    for i, parname in enumerate(params):
      exec("self.{pname} = Param(parname[1:])".format(pname  = parname[0]))

  def make_file(self, name):

    file = open(name, 'w')

    attrib = vars(self)
    keys = attrib.keys()

    file.write("@ " + self.pcfname.get() + "\n")
    for key in keys:
      if key != "pcfname":
        file.write(key + " " + attrib.get(key).value[0] + "\n")
    file.close()

def read_pcf(file):
    """
    Function to read the file:
    """

    # List containing the set of parameters:
    pcfsets = []

    # Read the file
    file = open(file, 'r')
    lines = file.readlines()
    file.close()

    cleanlines = []   # list with only the important lines
    block      = []   # Blocks separator
    # Clean the lines:
    for i in np.arange(len(lines)):
      line = lines[i]
      # Strip off comments:
      try:
        line = line[0:line.index('#')].strip()
      except:
        line = line.strip()

      # Keep only useful lines:
      if len(line) > 0:
        cleanlines.append(line)
        # identify the separators:
        if line[0] == "@":
          block.append([len(cleanlines)-1, line[1:].strip()])

    # Append a line to mark the end of the block:
    block.append([len(cleanlines), "end"])


    if (len(block)-1) == 0:
      # do normal readpcfs
      params = []
      for line in cleanlines:
        params.append( np.array(line.split()) )
      return Pcf(params)
    else:
      # Loop over each block:
      for i in np.arange(len(block)-1):
        params    = []  # List for the parameters and values of the block
        multiples = []  # line position of multiple valued parameter
        nval      = []  # number of values

        for j in np.arange(block[i][0]+1, block[i+1][0]):
          params.append( np.array(cleanlines[j].split()) )
          # if the parameter has more than 1 value:
          if len(params[-1]) > 2:
            multiples.append(len(params)-1)
            nval.append(len(params[-1])-1)

        
        # number of parameters with multiple values:
        nmul = len(multiples)

        if nmul == 0:
          pcfsets.append(params)
          pcfsets[-1].append(["pcfname", str(block[i][1])])
        else:
          # calculate total number of sets
          nt = 1
          for j in np.arange(nmul):
            nt *= nval[j]
          ncurrent = nt

          # holder of the sets of params:
          parset = []
          # make nt copies of the original set:
          for j in np.arange(nt):
            parset.append(params[:])
            # and add the pcfname:
            parset[j].append(["pcfname", str(block[i][1])])

          # Loop over each multiple valued parameter:
          for j in np.arange(nmul):
            ncurrent /= nval[j]
            mpar = np.copy(params[multiples[j]][1:])
            # Edit the value in each set:
            for k in np.arange(nt):
              index = int((k/ncurrent) % nval[j])

              parset[k][multiples[j]] = np.array([params[multiples[j]][0],
                                                  mpar[index]])
          for ps in parset:
            pcfsets.append(ps)

      # return a List of Pcf objects (one for each set):
      pcf = []
      i = 0
      for pcfset in pcfsets:
        pcf.append(Pcf(pcfset))
        i += 1
      return pcf
