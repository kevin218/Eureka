"""Base and child classes to handle orbital parameters

Author: Joe Filippazzo
Email: jfilippazzo@stsci.edu
"""
import os
import json

import numpy as np
from ..lib import readECF as rd

class Parameter:
    """A generic parameter class"""
    def __init__(self, name, value, ptype='free', mn=None, mx=None):
        """Instantiate a Parameter with a name and value at least

        Parameters
        ----------
        name: str
            The name of the parameter
        value: float, int, str, list, tuple
            The value of the parameter
        ptype: str
            Parameter type, ['free','fixed','independent','shared']
        mn: float, int, str, list, tuple (optioal)
            The minimum value
        mx: float, int, str, list, tuple (optioal)
            The maximim value
        """
        # If value is a list, distribute the elements
        if isinstance(value, list):
            value, *other = value
            if len(other) > 1:
                ptype, *other = other
            if len(other) > 0:
                mn, mx = other

        # Set the attributes
        self.name = name
        self.value = value
        self.mn = mn
        self.mx = mx
        self.ptype = ptype

    @property
    def ptype(self):
        """Getter for the ptype"""
        return self._ptype

    @ptype.setter
    def ptype(self, param_type):
        """Setter for ptype

        Parameters
        ----------
        param_type: str
            Parameter type, ['free','fixed','independent','shared']
        """
        if param_type not in ['free', 'fixed', 'independent', 'shared', True, False]:
            raise ValueError("ptype must be 'free','fixed', 'independent', or 'shared'")

        if param_type is True:
            param_type = 'free'

        if param_type is False:
            param_type = 'fixed'

        self._ptype = param_type

    @property
    def values(self):
        """Return all values for this parameter"""
        vals = self.name, self.value, self.ptype, self.mn, self.mx

        return list(filter(lambda x: x is not None, vals))


class Parameters:
    """A class to hold the Parameter instances
    """
    def __init__(self, param_file=None, **kwargs):
        """Initialize the parameter object

        Parameters
        ----------
        param_file: str
            A text file of the parameters to parse

        Example
        -------
        params = lightcurve.Parameters(a=20, ecc=0.1, inc=89,
        limb_dark='quadratic')
        """
        #self.__dict__['list'] = []
        self.__dict__['dict'] = {}

        # Make an empty params dict
        params = {}

        # If a param_file is given, make sure it exists
        if param_file is not None and os.path.exists(param_file):

            # Parse the ASCII file
            if param_file.endswith('.txt'):

                # Add the params to a dict
                data = np.genfromtxt(param_file)
                params = {i: j for i, j in data}
                print(params)

            # Parse the JSON file
            elif param_file.endswith('.json'):

                with open(param_file) as json_data:
                    params = json.load(json_data)

            # Parse Eureka control file
            elif param_file.endswith('.ecf'):

                ecf = rd.read_ecf(param_file)
                paramlist=vars(ecf)
                keylist=list(paramlist.keys())
                params={}
                kwargs={}
                for i in keylist:
                    if len(paramlist[i].getarr())==1:
                        params[i]=paramlist[i].get()
                    else:
                        params[i]=list(paramlist[i].getarr())


        # Add any kwargs to the parameter dict
        params.update(kwargs)

        # Try to store each as an attribute
        for param, value in params.items():
            setattr(self, param, value)

    def __add__(self, other):
        """Add parameters to make a combined model

        Parameters
        ----------
        other: ExoCTK.lightcurve_fitting.parameters.Parameters
            The parameters to  to multiply

        Returns
        -------
        ExoCTK.lightcurve_fitting.parameters.Parameters
            The combined model
        """
        # Make sure it is the right type
        if not type(self) == type(other):
            raise TypeError('Only another Parameters instance may be multiplied.')

        # Combine the model parameters too
        kwargs = self.dict
        kwargs.update(other.dict)
        newParams = Parameters(**kwargs)

        return newParams

    def __setattr__(self, item, value):
        """Maps attributes to values

        Parameters
        ----------
        item: str
            The name for the attribute
        value: any
            The attribute value
        """
        # Convert single items to list
        if isinstance(value, (str, float, int, bool)):
            value = [value,]

        # Convert tuple to list
        if isinstance(value, tuple):
            value = list(value)

        if not isinstance(value, list):
            raise TypeError("Cannot set {}={}.".format(item, value))
            
        # Set the attribute
        self.__dict__[item] = Parameter(item, *value)

        # Add it to the list of parameters
        self.__dict__['dict'][item] = self.__dict__[item].values[1:]
        #Don't want to append new items to list when updating values
        #self.__dict__['list'].append(self.__dict__[item].values)
