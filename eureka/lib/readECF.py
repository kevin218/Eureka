
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
        '''Initialize the MetaClass object.

        Parameters
        ----------
        folder: str, optional
            The folder containing an ECF file to be read in. Defaults to './'.
        file:   str, optional
            The ECF filename to be read in. Defaults to None which results in an empty MetaClass object.
        **kwargs:   dict, optional
            Any additional parameters to be loaded into the MetaClass after the ECF has been read in

        Notes
        -----

        History:
        - Mar 2022 Taylor J Bell
            Initial Version based on old readECF code.
        '''
        self.params = {}
        if file is not None and folder is not None:
            if os.path.exists(os.path.join(folder,file)):
                self.read(folder, file)
            else:
                raise ValueError(f"The file {os.path.join(folder,file)} does not exist.")

        if kwargs is not None:
            # Add any kwargs to the parameter dict
            self.params.update(kwargs)

            # Store each as an attribute
            for param, value in kwargs.items():
                setattr(self, param, value)

        return

    def __str__(self):
        '''A function to nicely format some outputs when a MetaClass object is converted to a string.

        This function gets used if one does str(meta) or print(meta).

        Parameters
        ----------
        None

        Returns
        -------
        output: str
            A string representation of what is contained in the MetaClass object.

        Notes
        -----
        History:

        - Mar 2022 Taylor J Bell
            Initial version.
        '''
        output = ''
        for par in self.params:
            # For each parameter, format a line as "Name: Value"
            output += par+': '+str(getattr(self, par))+'\n'
        return output

    def __repr__(self):
        '''A function to nicely format some outputs when asked for a printable representation of the MetaClass object.

        This function gets used if one does repr(meta) or does just meta in an interactive shell.

        Parameters
        ----------
        None

        Returns
        -------
        output: str
            A string representation of what is contained in the MetaClass object in a manner that could reproduce a similar MetaClass object.

        Notes
        -----
        History:

        - Mar 2022 Taylor J Bell
            Initial version.
        '''
        # Get the fully qualified name of the class
        output = type(self).__module__+'.'+type(self).__qualname__+'('
        # Show what folder and file were used to read in an ECF
        output += f"folder='{self.folder}', file='{self.filename}', "
        # Show what values have been loaded into the params dictionary
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

        Returns
        -------
        None
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
        """A function to read ECF files

        Parameters
        ----------
        folder: str
            The folder containing an ECF file to be read in.
        file:   str
            The ECF filename to be read in.

        Returns
        -------
        None

        Notes
        -----

        History:
        - Mar 2022 Taylor J Bell
            Initial Version based on old readECF code.
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

        if self.inputdir[0]=='/':
            self.inputdir = self.inputdir[1:]
        if self.inputdir[-1]!='/':
            self.inputdir += '/'
        if self.outputdir[0]=='/':
            self.outputdir = self.outputdir[1:]
        if self.outputdir[-1]!='/':
            self.outputdir += '/'

        return

    def write(self, folder):
        """A function to write an ECF file based on the current MetaClass settings.

        NOTE: For now this only rewrites the input ECF file to a new ECF file in the requested folder.
        In the future this function should make a full ECF file based on any adjusted parameters.

        Parameters
        ----------
        folder: str
            The folder where the ECF file should be written.

        Returns
        -------
        None

        Notes
        -----

        History:
        - Mar 2022 Taylor J Bell
            Initial Version.
        """
        with open(os.path.join(folder, self.filename), 'w') as file:
            file.writelines(self.lines)
        return

    def copy_ecf(self):
        """Copy an ECF file to the output directory to ensure reproducibility.

        NOTE: This will update the inputdir of the ECF file to point to the exact inputdir
        used to avoid ambiguity later and ensure that the ECF could be used to make the
        same outputs.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----

        History:
        - Mar 2022 Taylor J Bell
            Initial Version based on old readECF code.
        """
        # Copy ecf (and update inputdir to be precise which exact inputs were used)
        new_ecfname = os.path.join(self.outputdir, self.filename)
        with open(new_ecfname, 'w') as new_file:
            for line in self.lines:
                if len(line.strip())==0 or line.strip()[0]=='#':
                    new_file.write(line)
                else:
                    line_segs = line.strip().split()
                    if line_segs[0]=='inputdir':
                        if self.topdir in self.inputdir:
                            inputdir_string = self.inputdir[len(self.topdir):]
                        else:
                            inputdir_string = self.inputdir
                        new_file.write(line_segs[0]+'\t\t'+inputdir_string+'\t'+' '.join(line_segs[2:])+'\n')
                    else:
                        new_file.write(line)
        return
