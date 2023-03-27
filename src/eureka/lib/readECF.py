import os

# Required in case user passes in a numpy object (e.g. np.inf)
import numpy as np


class MetaClass:
    '''A class to hold Eureka! metadata.

    This class loads a Eureka! Control File (ecf) and lets you
    query the parameters and values.

    Notes
    -----
    History:

    - 2009-01-02 Christopher Campo
        Initial Version.
    - 2010-03-08 Patricio Cubillos
        Modified from ccampo version.
    - 2010-10-27 Patricio Cubillos
        Docstring updated
    - 2011-02-12 Patricio Cubillos
        Merged with ccampo's tepclass.py
    - 2022-03-24 Taylor J Bell
        Significantly modified for Eureka
    '''

    def __init__(self, folder=None, file=None, **kwargs):
        '''Initialize the MetaClass object.

        Parameters
        ----------
        folder : str; optional
            The folder containing an ECF file to be read in. Defaults to None
            which resolves to './'.
        file : str; optional
            The ECF filename to be read in. Defaults to None which results
            in an empty MetaClass object.
        **kwargs : dict
            Any additional parameters to be loaded into the MetaClass after
            the ECF has been read in

        Notes
        -----
        History:

        - Mar 2022 Taylor J Bell
            Initial Version based on old readECF code.
        '''
        if folder is None:
            folder = '.'+os.sep

        self.params = {}
        if file is not None and folder is not None:
            if os.path.exists(os.path.join(folder, file)):
                self.read(folder, file)
            else:
                raise ValueError(f"The file {os.path.join(folder,file)} "
                                 f"does not exist.")

        if kwargs is not None:
            # Add any kwargs to the parameter dict
            self.params.update(kwargs)

            # Store each as an attribute
            for param, value in kwargs.items():
                setattr(self, param, value)

    def __str__(self):
        '''A function to nicely format some outputs when a MetaClass object is
        converted to a string.

        This function gets used if one does str(meta) or print(meta).

        Returns
        -------
        str
            A string representation of what is contained in the
            MetaClass object.

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
        '''A function to nicely format some outputs when asked for a printable
        representation of the MetaClass object.

        This function gets used if one does repr(meta) or does just meta in an
        interactive shell.

        Returns
        -------
        str
            A string representation of what is contained in the MetaClass
            object in a manner that could reproduce a similar MetaClass object.

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
        item : str
            The name for the attribute
        value : any
            The attribute value
        """
        if item in ['lines', 'params', 'filename', 'folder']:
            self.__dict__[item] = value
            return

        # Set the attribute
        self.__dict__[item] = value

        # Add it to the list of parameters
        self.__dict__['params'][item] = value

    def read(self, folder, file):
        """A function to read ECF files

        Parameters
        ----------
        folder : str
            The folder containing an ECF file to be read in.
        file : str
            The ECF filename to be read in.

        Notes
        -----
        History:

        - Mar 2022 Taylor J Bell
            Initial Version based on old readECF code.
        - April 25, 2022 Taylor J Bell
            Joining topdir and inputdir/outputdir here.
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
                # FINDME: Need to catch only the expected exception
                pass
            self.params[name] = val

        # Store each as an attribute
        for param, value in self.params.items():
            setattr(self, param, value)

        self.inputdir_raw = self.inputdir
        self.outputdir_raw = self.outputdir

        # Join inputdir_raw and outputdir_raw to topdir for convenience
        # Use split to avoid issues from beginning
        self.inputdir = os.path.join(self.topdir,
                                     *self.inputdir.split(os.sep))
        self.outputdir = os.path.join(self.topdir,
                                      *self.outputdir.split(os.sep))

        # Make sure there's a trailing slash at the end of the paths
        if self.inputdir[-1] != os.sep:
            self.inputdir += os.sep
        if self.outputdir[-1] != os.sep:
            self.outputdir += os.sep

    def write(self, folder):
        """Write an ECF file based on the current MetaClass settings.

        NOTE: For now this rewrites the input_meta data to a new ECF file
        in the requested folder. In the future this function should make a full
        ECF file based on all parameters in meta.

        Parameters
        ----------
        folder : str
            The folder where the ECF file should be written.

        Notes
        -----
        History:

        - Mar 2022 Taylor J Bell
            Initial Version.
        - Oct 2022 Eva-Maria Ahrer
            Update parameters and replace
        """
        
        for i in range(len(self.lines)):
            line = self.lines[i]
            # Strip off comments:
            if "#" in line:
                line = line[0:line.index('#')]
            line = line.strip()

            if len(line) > 0:
                name = line.split()[0]
                val = ''.join(line.split()[1:])
                new_val = self.params[name]
                # check if values have been updated
                if val != new_val:
                    self.lines[i] = self.lines[i].replace(str(val), 
                                                          str(new_val))
        
        with open(os.path.join(folder, self.filename), 'w') as file:
            file.writelines(self.lines)

    def copy_ecf(self):
        """Copy an ECF file to the output directory to ensure reproducibility.

        NOTE: This will update the inputdir of the ECF file to point to the
        exact inputdir used to avoid ambiguity later and ensure that the ECF
        could be used to make the same outputs.

        Notes
        -----
        History:

        - Mar 2022 Taylor J Bell
            Initial Version based on old readECF code.
        """
        # Copy ecf (and update inputdir to be precise which exact inputs
        # were used)
        new_ecfname = os.path.join(self.outputdir, self.filename)
        with open(new_ecfname, 'w') as new_file:
            for line in self.lines:
                if len(line.strip()) == 0 or line.strip()[0] == '#':
                    new_file.write(line)
                else:
                    line_segs = line.strip().split()
                    if line_segs[0] == 'inputdir':
                        new_file.write(line_segs[0]+'\t\t'+self.inputdir_raw +
                                       '\t'+' '.join(line_segs[2:])+'\n')
                    else:
                        new_file.write(line)
