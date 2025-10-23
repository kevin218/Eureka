import os
import time as time_pkg
import crds
import shlex
# Required in case user passes in a numpy object (e.g. np.inf)
import numpy as np

from ..version import version

# A boolean to track if a Eureka! version mis-match warning has been issued
warned = False


class MetaClass:
    '''A class to hold Eureka! metadata.

    This class loads a Eureka! Control File (ecf) and lets you
    query the parameters and values.
    '''

    def __init__(self, folder='.'+os.sep, file=None, eventlabel=None,
                 stage=None, **kwargs):
        '''Initialize the MetaClass object.

        Parameters
        ----------
        folder : str; optional
            The folder containing an ECF file to be read in. Defaults to
            '.'+os.sep.
        file : str; optional
            The ECF filename to be read in. Defaults to None which first tries
            to find the filename using eventlabel and stage, and if that fails
            results in an empty MetaClass object.
        eventlabel : str; optional
            The unique identifier for these data.
        stage : int; optional
            The current analysis stage number.
        **kwargs : dict
            Any additional parameters to be loaded into the MetaClass after
            the ECF has been read in
        '''
        if folder is None:
            folder = '.'+os.sep

        if file is None and eventlabel is not None and stage is not None:
            file = f'S{stage}_{eventlabel}.ecf'

        self.params = {}

        # Determine if a file should be read
        file_path = os.path.join(folder, file) if file is not None else None
        if file_path is not None and os.path.exists(file_path):
            self.read(folder, file)
        elif file_path is not None and not kwargs:
            raise ValueError(f"The file {file_path} does not exist and no "
                             "kwargs were provided.")
        # else: assume kwargs will populate everything

        self.version = version
        if stage is not None:
            self.stage = stage
        self.eventlabel = eventlabel
        self.datetime = time_pkg.strftime('%Y-%m-%d')

        # If the data format hasn't been specified, must be eureka output
        self.data_format = getattr(self, 'data_format', 'eureka')

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

        # Stage may not be set yet, if not set, default to 0
        if hasattr(self, 'stage'):
            stage = self.stage
        else:
            stage = 0

        if (item == 'inst' and value == 'wfc3' and stage != '4cal'
                and stage < 4):
            # Fix issues with CRDS server set for JWST
            if 'jwst-crds.stsci.edu' in os.environ['CRDS_SERVER_URL']:
                print('CRDS_SERVER_URL is set for JWST and not HST.'
                      ' Automatically adjusting it up for HST.')
                url = 'https://hst-crds.stsci.edu'
                os.environ['CRDS_SERVER_URL'] = url
                crds.client.api.set_crds_server(url)
                crds.client.api.get_server_info.cache.clear()

            # If a specific CRDS context is entered in the ECF, apply it.
            # Otherwise, log and fix the default CRDS context to make sure
            # it doesn't change between different segments.
            self.pmap = getattr(self, 'pmap',
                                crds.get_context_name('hst')[4:-5])
            os.environ['CRDS_CONTEXT'] = f'hst_{self.pmap}.pmap'
        elif (item == 'inst' and value is not None and stage != '4cal'
              and stage < 4):
            # Fix issues with CRDS server set for HST
            if 'hst-crds.stsci.edu' in os.environ['CRDS_SERVER_URL']:
                print('CRDS_SERVER_URL is set for HST and not JWST.'
                      ' Automatically adjusting it up for JWST.')
                url = 'https://jwst-crds.stsci.edu'
                os.environ['CRDS_SERVER_URL'] = url
                crds.client.api.set_crds_server(url)
                crds.client.api.get_server_info.cache.clear()

            # If a specific CRDS context is entered in the ECF, apply it.
            # Otherwise, log and fix the default CRDS context to make sure
            # it doesn't change between different segments.
            self.pmap = getattr(self, 'pmap', None)
            if self.pmap is None:
                # Only need an internet connection if pmap is None
                self.pmap = crds.get_context_name('jwst')[5:-5]
            os.environ['CRDS_CONTEXT'] = f'jwst_{self.pmap}.pmap'

        if ((item == 'pmap') and hasattr(self, 'pmap') and
                (self.pmap is not None) and (self.pmap != value) and
                (stage != '4cal') and (stage < 4)):
            print(f'WARNING: pmap was set to {self.pmap} in the previous stage'
                  f' but is now set to {value} in this stage. This may cause '
                  'unexpected or undesireable behaviors.')

        global warned
        if (not warned and (item == 'version') and hasattr(self, 'version') and
                (self.version is not None) and (self.version != value)):
            warned = True
            print(f'WARNING: The Eureka! version was {self.version} in the '
                  f'previous stage but is now {value} in this stage. This may '
                  'cause unexpected or undesireable behaviors.')

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
            name = shlex.split(line)[0]
            # Split off the name and remove all spaces except quoted substrings
            # Also keep quotation marks for things that need to be escaped
            # (e.g. max is a built-in funciton)
            val = ''.join(shlex.split(line, posix=False)[1:])
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
