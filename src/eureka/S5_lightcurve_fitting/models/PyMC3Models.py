import numpy as np
import matplotlib.pyplot as plt
import os

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import starry
import pymc3 as pm
import theano.tensor as tt
import astropy.constants as const

from ...lib.readEPF import Parameters
from ..utils import COLORS

starry.config.quiet = True


class fit_class:
    def __init__(self):
        pass


BoundedNormal_0 = pm.Bound(pm.Normal, lower=0.0)
BoundedNormal_90 = pm.Bound(pm.Normal, upper=90.)


class StarryModel(pm.Model):
    def __init__(self, **kwargs):
        # Inherit from Model class
        super().__init__()

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')
        # Set parameters for multi-channel fits
        self.longparamlist = kwargs.get('longparamlist')
        self.nchan = kwargs.get('nchan')
        self.paramtitles = kwargs.get('paramtitles')
        self.uniqueparams = np.unique(self.longparamlist)

        self.components = [self]

        required = np.array(['Ms', 'Mp', 'Rs'])
        missing = np.array([name not in self.paramtitles for name in required])
        if np.any(missing):
            message = (f'Missing required params {required[missing]} in your '
                       f'EPF.')
            raise AssertionError(message)

        with self:
            for parname in self.paramtitles:
                param = getattr(self.parameters, parname)
                if param.ptype == 'independent':
                    continue
                elif param.ptype == 'fixed':
                    setattr(self, parname, tt.constant(param.value))
                elif param.ptype not in ['free', 'shared']:
                    message = (f'ptype {param.ptype} for parameter '
                               f'{param.name} is not recognized.')
                    raise ValueError(message)
                else:
                    for c in range(self.nchan):
                        if c != 0:
                            parname_temp = parname+'_'+str(c)
                        else:
                            parname_temp = parname

                        if param.ptype == 'free' or c == 0:
                            if param.prior == 'U':
                                setattr(self, parname_temp,
                                        pm.Uniform(parname_temp,
                                                   lower=param.priorpar1,
                                                   upper=param.priorpar2))
                            elif param.prior == 'N':
                                if parname in ['rp', 'per', 'ecc',
                                               'scatter_mult', 'scatter_ppm',
                                               'c0']:
                                    setattr(self, parname_temp,
                                            BoundedNormal_0(
                                                parname_temp,
                                                mu=param.priorpar1,
                                                sigma=param.priorpar2))
                                elif parname in ['inc']:
                                    setattr(self, parname_temp,
                                            BoundedNormal_90(
                                                parname_temp,
                                                mu=param.priorpar1,
                                                sigma=param.priorpar2))
                                else:
                                    setattr(self, parname_temp,
                                            pm.Normal(parname_temp,
                                                      mu=param.priorpar1,
                                                      sigma=param.priorpar2))
                            elif param.prior == 'LU':
                                setattr(self, parname_temp,
                                        tt.exp(pm.Uniform(
                                            parname_temp, 
                                            lower=param.priorpar1,
                                            upper=param.priorpar2)))
                        else:
                            # If a parameter is shared, make it equal to the
                            # 0th parameter value
                            setattr(self, parname_temp,
                                    pm.Deterministic(parname_temp,
                                                     getattr(self, parname)))

            if hasattr(self, 'u2'):
                self.udeg = 2
            elif hasattr(self, 'u1'):
                self.udeg = 1
            else:
                self.udeg = 0
            if hasattr(self, 'AmpCos2') or hasattr(self, 'AmpSin2'):
                self.ydeg = 2
            elif hasattr(self, 'AmpCos1') or hasattr(self, 'AmpSin1'):
                self.ydeg = 1
            else:
                self.ydeg = 0
            
            self.systems = []
            for c in range(self.nchan):
                # Initialize star object
                star = starry.Primary(starry.Map(ydeg=0, udeg=self.udeg,
                                                 amp=1.0),
                                      m=self.Ms, r=self.Rs, prot=1.0)

                # To save ourselves from tonnes of getattr lines, let's make a
                # new object without the _c parts of the parnames
                # This way we can do `temp.u1` rather than
                # `getattr(self, 'u1_'+c)`
                temp = fit_class()
                for key in self.paramtitles:
                    if getattr(self.parameters, key).ptype in ['free',
                                                               'shared',
                                                               'fixed']:
                        if (getattr(self.parameters, key).ptype != 'fixed'
                                and c > 0):
                            # Remove the _c part of the parname but leave any
                            # other underscores intact
                            setattr(temp, key, getattr(self, key+'_'+str(c)))
                        else:
                            setattr(temp, key, getattr(self, key))

                # FINDME: non-uniform limb darkening does not currently work
                if self.parameters.limb_dark.value == 'kipping2013':
                    # Transform stellar variables to uniform used by starry
                    star.map[1] = pm.Deterministic('u1_quadratic_'+str(c),
                                                   2*tt.sqrt(temp.u1)*temp.u2)
                    star.map[2] = pm.Deterministic('u2_quadratic_'+str(c),
                                                   tt.sqrt(temp.u1) *
                                                   (1-2*temp.u2))
                elif self.parameters.limb_dark.value == 'quadratic':
                    star.map[1] = temp.u1
                    star.map[2] = temp.u2
                elif self.parameters.limb_dark.value == 'linear':
                    star.map[1] = temp.u1
                elif self.parameters.limb_dark.value != 'uniform':
                    message = (f'ERROR: starryModel is not yet able to handle '
                               f'{self.parameters.limb_dark.value} limb'
                               f'darkening.\n'
                               f'       limb_dark must be one of uniform, '
                               f'linear, quadratic, or kipping2013.')
                    raise ValueError(message)
                
                if hasattr(self, 'fp'):
                    amp = temp.fp
                else:
                    amp = 0
                # Initialize planet object
                planet = starry.Secondary(
                    starry.Map(ydeg=self.ydeg, udeg=0, amp=amp, inc=90.0,
                               obl=0.0),
                    # Convert mass to M_sun units
                    m=self.Mp*const.M_jup.value/const.M_sun.value,
                    # Convert radius to R_star units
                    r=temp.rp*self.Rs,
                    # Setting porb here overwrites a
                    a=temp.a,
                    # porb = temp.per,
                    # prot = temp.per,
                    # Another option to set inclination using impact parameter
                    # inc=tt.arccos(b/a)*180/np.pi
                    inc=temp.inc,
                    ecc=temp.ecc,
                    w=temp.w
                )
                # Setting porb here may not override a
                planet.porb = temp.per
                # Setting prot here may not override a
                planet.prot = temp.per
                if hasattr(self, 'AmpCos1'):
                    planet.map[1, 0] = temp.AmpCos1
                if hasattr(self, 'AmpSin1'):
                    planet.map[1, 1] = temp.AmpSin1
                if self.ydeg == 2:
                    if hasattr(self, 'AmpCos2'):
                        planet.map[2, 0] = temp.AmpCos2
                    if hasattr(self, 'AmpSin2'):
                        planet.map[2, 1] = temp.AmpSin2
                # Offset is controlled by AmpSin1
                planet.theta0 = 180.0
                planet.tref = 0

                # Instantiate the system
                sys = starry.System(star, planet)
                self.systems.append(sys)

    @property
    def flux(self):
        """A getter for the flux"""
        return self._flux

    @flux.setter
    def flux(self, flux_array):
        """A setter for the flux

        Parameters
        ----------
        flux_array: sequence
            The flux array
        """
        # Check the type
        if not isinstance(flux_array, (np.ndarray, tuple, list)):
            raise TypeError("flux axis must be a tuple, list, or numpy array.")

        # Set the array
        # self._flux = np.array(flux_array)
        self._flux = np.ma.masked_array(flux_array)

    @property
    def parameters(self):
        """A getter for the parameters"""
        return self._parameters

    @parameters.setter
    def parameters(self, params):
        """A setter for the parameters"""
        # Process if it is a parameters file
        if isinstance(params, str) and os.path.isfile(params):
            params = Parameters(params)

        # Or a Parameters instance
        if (params is not None) and (type(params).__name__ !=
                                     Parameters.__name__):
            raise TypeError("'params' argument must be a JSON file, "
                            "ascii file, or Parameters instance.")

        # Set the parameters attribute
        self._parameters = params

    def setup(self, time, flux, lc_unc):
        self.time = time
        self.flux = flux
        self.lc_unc = lc_unc

        with self:
            if hasattr(self, 'scatter_ppm'):
                scatter_ppm_array = self.scatter_ppm*tt.ones(len(self.time))
                for c in range(1, self.nchan):
                    parname_temp = 'scatter_ppm_'+str(c)
                    scatter_ppm_array = tt.concatenate(
                        [scatter_ppm_array,
                         getattr(self, parname_temp)*tt.ones(len(self.time))])
                self.scatter_ppm_array = pm.Deterministic(
                    "scatter_ppm_array", scatter_ppm_array*1e6)
            if hasattr(self, 'scatter_mult'):
                # Fitting the noise level as a multiplier
                scatter_ppm_array = (self.scatter_mult *
                                     self.lc_unc[:len(self.time)])
                for c in range(1, self.nchan):
                    parname_temp = 'scatter_mult_'+str(c)
                    scatter_ppm_array = tt.concatenate(
                        [scatter_ppm_array,
                         (getattr(self, parname_temp) *
                          self.lc_unc[c*len(self.time):(c+1)*len(self.time)])])
                self.scatter_ppm_array = pm.Deterministic(
                    "scatter_ppm_array", scatter_ppm_array*1e6)
            if not hasattr(self, 'scatter_ppm_array'):
                # Not fitting the noise level
                self.scatter_ppm_array = self.lc_unc*1e6

            # This is how we tell `pymc3` about our observations;
            # we are assuming they are ampally distributed about
            # the true model. This line effectively defines our
            # likelihood function.
            pm.Normal("obs", mu=self.eval(eval=False),
                      sd=self.scatter_ppm_array/1e6, observed=self.flux)

        return

    def eval(self, eval=True, **kwargs):
        sys_eval = self.syseval(eval=eval)
        phys_eval = self.physeval(eval=eval)[0]
        if eval:
            return phys_eval*sys_eval
        else:
            return phys_eval*sys_eval

    def syseval(self, eval=True):
        if eval:
            # This is only called for things like plotting, so looping
            # doesn't matter
            poly_coeffs = np.zeros((self.nchan, 10))
            ramp_coeffs = np.zeros((self.nchan, 6))
            # Add fitted parameters
            for k, v in self.fit_dict.items():
                if k.lower().startswith('c'):
                    k = k[1:]
                    remvisnum = k.split('_')
                    if k.isdigit():
                        poly_coeffs[0, int(k)] = v
                    elif len(remvisnum) > 1 and self.nchan > 1:
                        if remvisnum[0].isdigit() and remvisnum[1].isdigit():
                            ind0 = int(remvisnum[1])
                            ind1 = int(remvisnum[0])
                            poly_coeffs[ind0][ind1] = v
                elif k.lower().startswith('r'):
                    k = k[1:]
                    remvisnum = k.split('_')
                    if k.isdigit():
                        ramp_coeffs[0, int(k)] = v
                    elif len(remvisnum) > 1 and self.nchan > 1:
                        if remvisnum[0].isdigit() and remvisnum[1].isdigit():
                            ind0 = int(remvisnum[1])
                            ind1 = int(remvisnum[0])
                            ramp_coeffs[ind0][ind1] = v

            poly_coeffs = poly_coeffs[:, ~np.all(poly_coeffs == 0, axis=0)]
            poly_coeffs = np.flip(poly_coeffs, axis=1)
            poly_flux = np.zeros(0)
            time_poly = self.time - self.time.mean()
            for c in range(self.nchan):
                poly = np.poly1d(poly_coeffs[c])
                poly_flux = np.concatenate(
                    [poly_flux, np.polyval(poly, time_poly)])

            ramp_flux = np.zeros(0)
            time_ramp = self.time - self.time[0]
            for c in range(self.nchan):
                r0, r1, r2, r3, r4, r5 = ramp_coeffs[c]
                lcpiece = (r0*np.exp(-r1*time_ramp + r2) +
                           r3*np.exp(-r4*time_ramp + r5) +
                           1)
                ramp_flux = np.concatenate([ramp_flux, lcpiece])

            return poly_flux*ramp_flux
        else:
            # This gets compiled before fitting, so looping doesn't matter
            poly_coeffs = np.zeros((self.nchan, 10)).tolist()
            ramp_coeffs = np.zeros((self.nchan, 6)).tolist()
            # Add fitted parameters
            for k in self.uniqueparams:
                if k.lower().startswith('c'):
                    k = k[1:]
                    remvisnum = k.split('_')
                    if k.isdigit():
                        poly_coeffs[0][int(k)] = getattr(self, 'c'+k)
                    elif len(remvisnum) > 1 and self.nchan > 1:
                        if remvisnum[0].isdigit() and remvisnum[1].isdigit():
                            ind0 = int(remvisnum[1])
                            ind1 = int(remvisnum[0])
                            poly_coeffs[ind0][ind1] = getattr(self, 'c'+k)
                elif k.lower().startswith('r'):
                    k = k[1:]
                    remvisnum = k.split('_')
                    if k.isdigit():
                        ramp_coeffs[0][int(k)] = getattr(self, 'r'+k)
                    elif len(remvisnum) > 1 and self.nchan > 1:
                        if remvisnum[0].isdigit() and remvisnum[1].isdigit():
                            ind0 = int(remvisnum[1])
                            ind1 = int(remvisnum[0])
                            ramp_coeffs[ind0][ind1] = getattr(self, 'r'+k)

            poly_flux = tt.zeros(0)
            time_poly = self.time - self.time.mean()
            for c in range(self.nchan):
                lcpiece = tt.zeros(len(self.time))
                for power in range(len(poly_coeffs[c])):
                    lcpiece += poly_coeffs[c][power] * time_poly**power
                poly_flux = tt.concatenate([poly_flux, lcpiece])

            ramp_flux = tt.zeros(0)
            time_ramp = self.time - self.time[0]
            for c in range(self.nchan):
                r0, r1, r2, r3, r4, r5 = ramp_coeffs[c]
                lcpiece = (r0*tt.exp(-r1*time_ramp + r2) +
                           r3*tt.exp(-r4*time_ramp + r5) +
                           1)
                ramp_flux = tt.concatenate([ramp_flux, lcpiece])

            return poly_flux*ramp_flux

    def physeval(self, interp=False, eval=True):
        if interp:
            dt = self.time[1]-self.time[0]
            steps = int(np.round((self.time[-1]-self.time[0])/dt+1))
            new_time = np.linspace(self.time[0], self.time[-1], steps,
                                   endpoint=True)
        else:
            new_time = self.time

        if eval:
            phys_flux = np.zeros(0)
            for c in range(self.nchan):
                lcpiece = self.fit.systems[c].flux(new_time-self.fit.t0).eval()
                phys_flux = np.concatenate([phys_flux, lcpiece])

            return phys_flux, new_time
        else:
            phys_flux = tt.zeros(0)
            for c in range(self.nchan):
                lcpiece = self.systems[c].flux(new_time-self.t0)
                phys_flux = tt.concatenate([phys_flux, lcpiece])

            return phys_flux, new_time

    @property
    def fit_dict(self):
        return self._fit_dict

    @fit_dict.setter
    def fit_dict(self, input_fit_dict):
        self._fit_dict = input_fit_dict

        fit = fit_class()
        for key in self.fit_dict.keys():
            setattr(fit, key, self.fit_dict[key])

        for parname in self.uniqueparams:
            param = getattr(self.parameters, parname)
            if param.ptype == 'independent':
                continue
            elif param.ptype == 'fixed':
                setattr(fit, parname, param.value)
            elif param.ptype == 'shared':
                for c in range(1, self.nchan):
                    parname_temp = parname+'_'+str(c)
                    setattr(fit, parname_temp, getattr(fit, parname))
                    self._fit_dict[parname_temp] = getattr(fit, parname)

        if hasattr(self, 'u2'):
            fit.udeg = 2
        elif hasattr(self, 'u1'):
            fit.udeg = 1
        else:
            fit.udeg = 0
        if hasattr(self, 'AmpCos2') or hasattr(self, 'AmpSin2'):
            fit.ydeg = 2
        elif hasattr(self, 'AmpCos1') or hasattr(self, 'AmpSin1'):
            fit.ydeg = 1
        else:
            fit.ydeg = 0

        fit.systems = []
        for c in range(self.nchan):
            # Initialize star object
            star = starry.Primary(starry.Map(ydeg=0, udeg=fit.udeg, amp=1.0),
                                  m=self.Ms, r=self.Rs, prot=1.0)

            # To save ourselves from tonnes of getattr lines, let's make a new
            # object without the _c parts of the parnames
            # This way we can do `temp.u1` rather than
            # `getattr(self, 'u1_'+c)`
            temp = fit_class()
            for key in self.paramtitles:
                if getattr(self.parameters, key).ptype in ['free', 'shared',
                                                           'fixed']:
                    if (c > 0 and
                            getattr(self.parameters, key).ptype != 'fixed'):
                        # Remove the _c part of the parname but leave any
                        # other underscores intact
                        setattr(temp, key, getattr(fit, key+'_'+str(c)))
                    else:
                        setattr(temp, key, getattr(fit, key))

            # FINDME: non-uniform limb darkening does not currently work
            if self.parameters.limb_dark.value == 'kipping2013':
                # Transform stellar variables to uniform used by starry
                star.map[1] = 2*np.sqrt(temp.u1)*temp.u2
                star.map[2] = np.sqrt(temp.u1)*(1-2*temp.u2)
            elif self.parameters.limb_dark.value == 'quadratic':
                star.map[1] = temp.u1
                star.map[2] = temp.u2
            elif self.parameters.limb_dark.value == 'linear':
                star.map[1] = temp.u1
            elif self.parameters.limb_dark.value != 'uniform':
                message = (f'ERROR: starryModel is not yet able to handle '
                           f'{self.parameters.limb_dark.value} limb_dark.\n'
                           f'       limb_dark must be one of uniform, linear, '
                           f'quadratic, or kipping2013.')
                raise ValueError(message)
            
            if hasattr(self, 'fp'):
                amp = temp.fp
            else:
                amp = 0
            # Initialize planet object
            planet = starry.Secondary(
                starry.Map(ydeg=fit.ydeg, udeg=0, amp=amp, inc=90.0, obl=0.0),
                # Convert mass to M_sun units
                m=self.Mp*const.M_jup.value/const.M_sun.value,
                # Convert radius to R_star units
                r=temp.rp*self.Rs,
                # Setting porb here overwrites a
                a=temp.a,
                # porb = temp.per,
                # prot = temp.per,
                # Another option to set inclination using impact parameter
                # inc=tt.arccos(b/a)*180/np.pi
                inc=temp.inc,
                ecc=temp.ecc,
                w=temp.w
            )
            # Setting porb here may not override a
            planet.porb = temp.per
            # Setting prot here may not override a
            planet.prot = temp.per
            if hasattr(self, 'AmpCos1'):
                planet.map[1, 0] = temp.AmpCos1
            if hasattr(self, 'AmpSin1'):
                planet.map[1, 1] = temp.AmpSin1
            if self.ydeg == 2:
                if hasattr(self, 'AmpCos2'):
                    planet.map[2, 0] = temp.AmpCos2
                if hasattr(self, 'AmpSin2'):
                    planet.map[2, 1] = temp.AmpSin2
            # Offset is controlled by AmpSin1
            planet.theta0 = 180.0
            planet.tref = 0

            # Instantiate the system
            sys = starry.System(star, planet)
            fit.systems.append(sys)

        if hasattr(fit, 'scatter_mult'):
            # Fitting the noise level as a multiplier
            fit.scatter_ppm = fit.scatter_mult*self.lc_unc*1e6
        if not hasattr(fit, 'scatter_ppm'):
            # Not fitting the noise level
            fit.scatter_ppm = self.lc_unc*1e6

        self.fit = fit

        return

    def plot(self, time, components=False, ax=None, draw=False, color='blue',
             zorder=np.inf, share=False, chan=0, **kwargs):
        """Plot the model

        Parameters
        ----------
        time: array-like
            The time axis to use
        components: bool
            Plot all model components
        ax: Matplotlib Axes
            The figure axes to plot on

        Returns
        -------
        bokeh.plotting.figure
            The figure
        """
        # Make the figure
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.gca()

        # Set the time
        self.time = time

        # Plot the model
        label = self.fitter
        if self.name != 'New Model':
            label += ': '+self.name
        
        flux = self.eval(**kwargs)
        if share:
            flux = flux[chan*len(self.time):(chan+1)*len(self.time)]
        
        ax.plot(self.time, flux, '.', ls='', ms=2, label=label,
                color=color, zorder=zorder)

        if components and self.components is not None:
            for comp in self.components:
                comp.plot(self.time, ax=ax, draw=False, color=next(COLORS),
                          zorder=zorder, share=share, chan=chan, **kwargs)

        # Format axes
        ax.set_xlabel(str(self.time_units))
        ax.set_ylabel('Flux')

        if draw:
            fig.show()
        else:
            return

    @property
    def time(self):
        """A getter for the time"""
        return self._time

    @time.setter
    def time(self, time_array, time_units='BJD'):
        """A setter for the time

        Parameters
        ----------
        time_array: sequence, astropy.units.quantity.Quantity
            The time array
        time_units: str
            The units of the input time_array, ['MJD', 'BJD', 'phase']
        """
        # Check the type
        if not isinstance(time_array, (np.ndarray, tuple, list)):
            raise TypeError("Time axis must be a tuple, list, or numpy array.")

        # Set the units
        self.time_units = time_units

        # Set the array
        # self._time = np.array(time_array)
        self._time = np.ma.masked_array(time_array)
