import numpy as np
from bokeh.plotting import figure, show
try:
    import batman
except ImportError:
    print("Could not import batman. Functionality may be limited.")

from . import utils


def simulate_lightcurve(target, snr=1000., npts=1000, nbins=10, radius=None,
                        ldcs=('quadratic', [0.1, 0.1]), plot=False):
    """Simulate lightcurve data for the given target exoplanet

    Parameters
    ----------
    target : str
        The name of the target to simulate.
    snr : float; optional
        The signal to noise to use. Defaults to 1000.
    npts : int; optional
        The number of points in each lightcurve. Defaults to 1000.
    nbins : int; optional
        The number of lightcurves. Defaults to 10.
    radius : array-like, float; optional
        The radius or radii value(s) to use. Defaults to None which
        will use the literature Rp/Rs value if found or 0.1 if not.
    ldcs : sequence; optional
        The limb darkening profile name and coefficients.
        Defaults to ('quadratic', [0.1, 0.1]).
    plot : bool; optional
        If True, plot the figure. Defaults to False.

    Returns
    -------
    tuple
        The time, flux, uncertainty, and transit parameters
    """
    try:

        # Get the system parameters from ExoMAST
        targ, url = utils.get_target_data(target)
        name = targ.get('canonical_name') or target
        t0 = targ.get('transit_time', 0.)
        dt = targ.get('transit_duration', 1.)

        # Generate transit parameters with batman
        params = batman.TransitParams()
        params.t0 = t0
        params.rp = targ.get('Rp/Rs') or 0.1
        params.per = targ.get('orbital_period') or 0.5
        params.inc = targ.get('inclination') or 90.
        params.a = targ.get('a/Rs') or 15.
        params.ecc = targ.get('eccentricity') or 0.
        params.w = targ.get('omega') or 90.
        params.limb_dark = 'nonlinear' if ldcs[0] == '4-parameter' else ldcs[0]
        params.transittype = 'primary'
        params.u = ldcs[1]

        # Generate a time axis
        time = np.linspace(t0-dt, t0+dt, npts)

        # Make the transit model
        transit = batman.TransitModel(params, time, transittype='primary')

        # Generate the lightcurves
        flux = []
        if radius is None:
            radius = params.rp
        radii = [radius]*nbins if isinstance(radius, (int, float)) else radius
        for r in radii:
            params.rp = r
            flux.append(transit.light_curve(params))

        # Add noise
        ideal_flux = np.asarray(flux)
        flux = np.random.normal(loc=ideal_flux, scale=ideal_flux/snr)
        unc = flux - ideal_flux

        # Plot it
        if plot:
            fig = figure(title=name)
            fig.circle(time, flux[0])
            fig.xaxis.axis_label = targ.get('transit_time_unit') or 'MJD'
            fig.yaxis.axis_label = 'Relative Flux'
            show(fig)

        return time, flux, unc, targ

    except IOError:
        raise ValueError(f'{target}: Could not simulate light curve for this '
                         f'target')
