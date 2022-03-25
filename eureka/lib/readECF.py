
import numpy as np
import os

"""
    This class loads a Eureka! Control File (ecf) and lets you
    querry the parameters and values.

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

class MetaClass:
    '''A class to hold Eureka! metadata.
    '''

    def __init__(self, folder='./', file=None, **kwargs):
        self.params = {}
        if file is not None and folder is not None and os.path.exists(os.path.join(folder,file)):
            self.read(folder, file)

        if kwargs is not None:
            # Add any kwargs to the parameter dict
            self.params.update(kwargs)

            # Store each as an attribute
            for param, value in kwargs.items():
                setattr(self, param, value)

        return

    def __str__(self):
        output = ''
        for par in self.params:
            output += par+': '+str(getattr(self, par))+'\n'
        return output

    def __repr__(self):
        output = type(self).__module__+'.'+type(self).__qualname__+'('
        output += "folder='./', file=None, "
        output += "**"+str(self.params)
        output = output+')'
        return output
    
    def __setattr__(self, item, value):
        """Maps attributes to values

        Parameters
        ----------
        item: str
            The name for the attribute
        value: any
            The attribute value
        """
        if item=='lines' or item=='params' or item=='filename' or item=='folder':
            self.__dict__[item] = value
            return

        # Set the attribute
        self.__dict__[item] = value

        # Add it to the list of parameters
        self.__dict__['params'][item] = value

        return

    def read(self, folder, file):
        """
        Function to read the file:
        """
        self.filename = file
        self.folder = folder
        # Read the file
        with open(os.path.join(folder, file), 'r') as file:
            self.lines = file.readlines()

        cleanlines = []   # list with only the important lines
        # Clean the lines:
        for line in self.lines:
            # Strip off comments:
            if "#" in line:
                line = line[0:line.index('#')]
            line = line.strip()

            # Keep only useful lines:
            if len(line) > 0:
                cleanlines.append(line)

        for line in cleanlines:
            name = line.split()[0]
            val = ''.join(line.split()[1:])
            try:
                val = eval(val)
            except:
                pass
            self.params[name] = val

        # Store each as an attribute
        for param, value in self.params.items():
            setattr(self, param, value)

        return

    def write(self, folder):
        with open(os.path.join(folder, self.filename), 'w') as file:
            file.writelines(self.lines)
        return

    def copy_ecf(self):
        # Copy ecf (and update inputdir to be precise which exact inputs were used)
        new_ecfname = os.path.join(self.outputdir, self.filename)
        with open(new_ecfname, 'w') as new_file:
            for line in self.lines:
                if len(line.strip())==0 or line.strip()[0]=='#':
                    new_file.write(line)
                else:
                    line_segs = line.strip().split()
                    if line_segs[0]=='inputdir':
                        new_file.write(line_segs[0]+'\t\t/'+self.inputdir+'\t'+' '.join(line_segs[2:])+'\n')
                    else:
                        new_file.write(line)
        return
