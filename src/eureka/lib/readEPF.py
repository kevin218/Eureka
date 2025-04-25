import numpy as np
import os
import inspect


class Parameter:
    """A generic parameter class"""
    def __init__(self, name, value, ptype, priorpar1=None, priorpar2=None,
                 prior=None):
        """Instantiate a Parameter with a name and value at least.

        Parameters
        ----------
        name : str
            The name of the parameter.
        value : float, int, str, list, tuple
            The value of the parameter.
        ptype : str
            The parameter type from ['free','fixed','independent','shared',
            'white_free', 'white_fixed'].
        priorpar1 : float, int, str, list, tuple; optional
            The first prior input value: lower-bound for uniform/log uniform
            priors, or mean for normal priors. Defaults to None.
        priorpar2 : float, int, str, list, tuple; optional
            The second prior input value: upper-bound for uniform/log uniform
            priors, or std. dev. for normal priors. Defaults to None.
        prior : str; optional
            Type of prior, ['U','LU','N']. Defaults to None.
        """
        # Set the attributes
        self.name = name
        self.value = value
        self.priorpar1 = priorpar1
        self.priorpar2 = priorpar2
        self.ptype = ptype
        self.prior = prior

    def __str__(self):
        '''A function to nicely format some outputs when a Parameter object is
        converted to a string.

        This function gets used if one does str(param) or print(param).

        Returns
        -------
        str
            A string representation of what is contained in
            the Parameter object.

        Notes
        -----
        History:

        - Mar 2022 Taylor J Bell
            Initial version.
        '''
        # Return a string representation of the self.values list
        return str(self.values)

    def __repr__(self):
        '''A function to nicely format some outputs when asked for a printable
        representation of the Parameter object.

        This function gets used if one does repr(param) or does just param
        in an interactive shell.

        Returns
        -------
        str
            A string representation of what is contained in the Parameter
            object in a manner that could reproduce a similar Parameter object.

        Notes
        -----
        History:

        - Mar 2022 Taylor J Bell
            Initial version.
        '''
        # Get the fully qualified name of the class
        output = type(self).__module__+'.'+type(self).__qualname__+'('
        # Get the list of Parameter.__init__ arguments (excluding self)
        keys = inspect.getfullargspec(Parameter.__init__).args[1:]
        # Show how the Parameter could be initialized
        # (e.g. name='rp', val=0.01, ptype='free')
        for name in keys:
            val = getattr(self, name)
            if isinstance(val, str):
                val = "'"+val+"'"
            else:
                val = str(val)
            output += name+'='+val+', '
        # Remove the extra ', ' and close with a parenthesis
        output = output[:-2]+')'
        return output

    @property
    def ptype(self):
        """Getter for the ptype"""
        return self._ptype

    @ptype.setter
    def ptype(self, param_type):
        """Setter for ptype

        Parameters
        ----------
        param_type : str
            Parameter type, ['free','fixed','independent','shared',
            'white_free', 'white_fixed']
        """
        if param_type not in ['free', 'fixed', 'independent', 'shared',
                              'white_free', 'white_fixed']:
            raise ValueError("ptype must be 'free', 'fixed', 'independent', "
                             "'shared', 'white_free', or 'white_fixed'")

        self._ptype = param_type

    @property
    def values(self):
        """Return all values for this parameter

        Returns
        -------
        list
            [self.name, self.value, self.ptype, self.priorpar1,
             self.priorpar2, self.prior] excluding any values which
             are None.
        """
        vals = (self.name, self.value, self.ptype, self.priorpar1,
                self.priorpar2, self.prior)

        return list(filter(lambda x: x is not None, vals))


class Parameters:
    """A class to hold the Parameter instances

    This class loads a Eureka! Parameter File (epf) and lets you
    query the parameters and values.

    Notes
    -----
    History:

    - 2022-03-24 Taylor J Bell
        Based on readECF with significant edits for Eureka
    """
    def __init__(self, param_path=None, param_file=None, **kwargs):
        """Initialize the parameter object

        Parameters
        ----------
        param_path : str; optional
            The path to the parameters. Defaults to None.
        param_file : str; optional
            A text file of the parameters to parse. Defaults to None.
        **kwargs : dict
            Any additional settings to set in the Parameters object.
        """
        if param_path is None:
            param_path = '.'+os.sep

        # Make an empty params dict
        self.params = {}
        self.dict = {}

        # If a param_file is given, make sure it exists
        param_file_okay = (param_file is not None and
                           param_path is not None)
        if param_file_okay:
            if not os.path.exists(os.path.join(param_path, param_file)):
                raise FileNotFoundError(
                    f"The Eureka! Parameter File:\n"
                    f"{os.path.join(param_path, param_file)}\n"
                    f"does not exist. Make sure to update the fit_par setting"
                    f" in your Stage 5 ECF to point to the EPF file you've "
                    f"made.")
            elif param_file.endswith('.txt') or param_file.endswith('.json'):
                raise AssertionError(
                    'ERROR: S5 parameter files in txt or json file formats '
                    'have been deprecated.\n'
                    'Please change to using EPF (Eureka! Parameter File) '
                    'file formats.')
            elif param_file.endswith('.ecf'):
                print('WARNING, using ECF file formats for S5 parameter files '
                      'has been deprecated.')
                print('Please update the file format to an EPF (Eureka! '
                      'Parameter File; .epf).')

            self.read(param_path, param_file)

        # Add any kwargs to the parameter dict
        self.params.update(kwargs)

        # Try to store each as an attribute
        for param, value in self.params.items():
            setattr(self, param, value)

    def __str__(self):
        '''A function to nicely format some outputs when a Parameters object
        is converted to a string.

        This function gets used if one does str(params) or print(params).

        Returns
        -------
        str
            A string representation of what is contained in
            the Parameters object.

        Notes
        -----
        History:

        - Mar 2022 Taylor J Bell
            Initial version.
        '''
        output = ''
        for key in self.params:
            # For each parameter, format a line as "Name: Value"
            output += key+': '+str(getattr(self, key))+'\n'
        return output[:-1]

    def __repr__(self):
        '''A function to nicely format some outputs when asked for a printable
        representation of the Parameters object.

        This function gets used if one does repr(params) or does just params
        in an interactive shell.

        Returns
        -------
        str
            A string representation of what is contained in the Parameters
            object in a manner that could reproduce a similar Parameters
            object.

        Notes
        -----
        History:

        - Mar 2022 Taylor J Bell
            Initial version.
        '''
        # Get the fully qualified name of the class
        output = type(self).__module__+'.'+type(self).__qualname__+'('
        # Show what folder and file were used to read in an EPF
        output += f"param_path='{self.folder}', param_file='{self.filename}', "
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
        if item in ['epf', 'params', 'dict', 'filename', 'folder',
                    'lines', 'cleanlines']:
            self.__dict__[item] = value
            return

        if isinstance(value, (str, float, int, bool)):
            # Convert single items to list
            value = [value, ]
        elif isinstance(value, tuple):
            # Convert tuple to list
            value = list(value)
        elif not isinstance(value, list):
            raise TypeError("Cannot set {}={}.".format(item, value))

        # Set the attribute
        self.params[item] = value
        self.__dict__[item] = Parameter(item, *value)

        # Add it to the list of parameters
        self.__dict__['dict'][item] = self.__dict__[item].values[1:]

    def __add__(self, other):
        """Add parameters to make a combined model

        Parameters
        ----------
        other : Parameters
            The parameters object to combine

        Returns
        -------
        newParams : Parameters
            The combined model
        """
        # Make sure it is the right type
        if not isinstance(self, type(other)):
            raise TypeError('Only another Parameters instance may be added.')

        # Combine the model parameters too
        kwargs = self.dict
        kwargs.update(other.dict)
        newParams = Parameters(**kwargs)

        return newParams

    def read(self, folder, file):
        """A function to read EPF files

        Parameters
        ----------
        folder : str
            The folder containing an EPF file to be read in.
        file : str
            The EPF filename to be read in.

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

            line = ' '.join(line.split())

            # Keep only useful lines:
            if len(line) > 0:
                cleanlines.append(line)

        self.params = {}
        for line in cleanlines:
            par = np.array(line.split())
            name = par[0]
            vals = []
            for i in range(len(par[1:])):
                try:
                    vals.append(eval(par[i+1]))
                except:
                    # FINDME: Need to catch only the expected exception
                    vals.append(par[i+1])
            self.params[name] = vals

    def write(self, folder):
        """A function to write an EPF file based on the current Parameters
        settings.

        NOTE: For now this only rewrites the input EPF file to a new EPF file
        in the requested folder. In the future this function should make a
        full EPF file based on any adjusted parameters.

        Parameters
        ----------
        folder : str
            The folder where the EPF file should be written.

        Notes
        -----
        History:

        - Mar 2022 Taylor J Bell
            Initial Version.
        """
        with open(os.path.join(folder, self.filename), 'w') as file:
            file.writelines(self.lines)
