
import numpy as np
import os

"""
    This class loads a Eureka! Control File (ecf) and lets you
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
    >>> # Load a ecf file:
    >>> import reader3 as rd
    >>> reload(rd)
    >>> ecf = rd.ecffile('/home/patricio/ast/esp01/anal/wa011bs11/run/wa011bs11.ecf')

    >>> Each parameter has the attribute value, wich is a ndarray:
    >>> ecf.planet.value
    array(['wa011b'],
          dtype='|S6')

    >>> # To get the n-th value of a parameter use ecffile.param.get(n):
    >>> # if it can't be converted to a number/bool/etc, it returns a string.
    >>> ecf.planet.get(0)
    'wa011b'
    >>> ecf.photchan.get(0)
    1
    >>> ecf.fluxunits.get(0)
    True

    >>> # Use ecffile.param.value[n] to get the n-th value as string:
    >>> ecf.aorname.get(0)
    38807808
    >>> ecf.aorname.value[0]
    '38807808'

    >>> # The function ecffile.param.getarr() returns the numeric/bool/etc
    >>> # values of a parameter as a nparray:
    >>> ecf.sigma.value
    array(['4.0', '4.0'],
          dtype='|S5')
    >>> ecf.sigma.getarr()
    array([4.0, 4.0], dtype=object)


    Modification History:
    --------------------
    2009-01-02 chris      Initial Version.
                          by Christopher Campo      ccampo@gmail.com
    2010-03-08 patricio   Modified from ccampo version.
                          by Patricio Cubillos      pcubillos@fulbrightmail.org
    2010-10-27 patricio   Docstring updated
    2011-02-12 patricio   Merged with ccampo's tepclass.py
    2022-03-24 taylor     Significantly modified for Eureka
                          by Taylor J Bell          bell@baeri.org

"""


# each parameter is an instance of this class
class Param:
    # constructor
    def __init__(self, vals):
        self.value = vals

    def __str__(self):
        try:
            return str(self.get(0))
        except:
            try:
                return str(self.getarr(0))
            except:
                return "Unable to print parameter"

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

class Ecf:
    def __init__(self, params):
        # :::: And now, Chris' code ::::
        # load all parameters
        for i, parname in enumerate(params):
            setattr(self, parname[0], Param(parname[1:]))

    def __str__(self):
        output = ''
        for par in self.__dict__:
            output += par+': '+str(self.__dict__[par])+'\n'
        return output

    def make_file(self, name):
        with open(name, 'w') as file:
            attrib = vars(self)
            keys = attrib.keys()

            file.write("@ " + self.ecfname.get() + "\n")
            for key in keys:
                if key != "ecfname":
                    file.write(key + " " + attrib.get(key).value[0] + "\n")
        return

def read_ecf(folder, file):
    """
    Function to read ECF files:
    """

    # Read the file
    with open(os.path.join(folder, file), 'r') as file:
        lines = file.readlines()

    cleanlines = []   # list with only the important lines
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

    # do normal readecfs
    params = []
    for line in cleanlines:
        temp_line = line.split()
        if len(temp_line)>2:
            # Remove any spaces in the ecf value (otherwise only the values before the space would be used)
            temp_line = [temp_line[0], ''.join(temp_line[1:])]
        params.append( np.array(temp_line) )
    return Ecf(params)

def store_ecf(meta, ecf):
    '''
    Store values from Eureka control file as parameters in Meta object.
    '''
    for key in ecf.__dict__.keys():
        try:
            setattr(meta, key, getattr(ecf, key).get(0))
        except:
            try:
                setattr(meta, key, getattr(ecf, key).getarr(0))
            except:
                print("Unable to store parameter: " + key)
    return

def copy_ecf(meta, ecffolder, ecffile):
    # Copy ecf (and update inputdir to be precise which exact inputs were used)
    new_ecfname = meta.outputdir + ecffile.split('/')[-1]
    with open(new_ecfname, 'w') as new_file:
        with open(os.path.join(ecffolder, ecffile), 'r') as file:
            for line in file.readlines():
                if len(line.strip())==0 or line.strip()[0]=='#':
                    new_file.write(line)
                else:
                    line_segs = line.strip().split()
                    if line_segs[0]=='inputdir':
                        new_file.write(line_segs[0]+'\t\t/'+meta.inputdir+'\t'+' '.join(line_segs[2:])+'\n')
                    else:
                        new_file.write(line)
    return
