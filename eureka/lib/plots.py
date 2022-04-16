from matplotlib import rc, rcdefaults, rcParams


def set_rc(style='preserve', usetex=False, from_scratch=False, **kwargs):
    """Function to adjust matplotlib rcParams for plotting procedures.

    Parameters
    ----------
    style : str, optional
        Your plotting style from ("custom", "eureka", "preserve", or "default").
        Custom passes all kwargs to the 'font' rcParams group at the moment.
        Eureka sets some nicer rcParams settings recommended by the Eureka team.
        Preserve leaves all rcParams as-is and can be used to toggle the usetex parameter.
        By default uses 'preserve'.
    usetex : bool, optional
        Do you want to use LaTeX fonts (which requires LaTeX to be installed), by default False
    from_scratch : bool, optional
        Should the rcParams first be set to rcdefaults? By default False
    **kwargs : dict, optional
        Any additional parameters to passed to the 'font' rcParams group.

    Raises
    ------
    ValueError
        Ensures that usetex and from_scratch arguments are boolean
    ValueError
        Ensures that input style is one of: "custom", "eureka", "preserve", or "default"
    """
    if not (isinstance(usetex, bool) and isinstance(from_scratch, bool)):
        raise ValueError('"usetex" and "from_scratch" arguments must be boolean.')

    if from_scratch:
        # Reset all rcParams prior to making changes. 
        rcdefaults()

    # Apply the desired style...
    if style == 'custom':
        # Custom user font, kwargs must be from the 'font' rcParams group at the moment. 
        # https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams
        rc('font', **kwargs)
    elif style == 'eureka':
        # Apply default Eureka! font settings.
        family='sans-serif'
        font='Helvetica'
        fontsize=16
        rc('font', **{'family': family, family: [font], 'size': fontsize})
        params = {'legend.fontsize': 11}
        rcParams.update(params)
    elif style == 'default':
        # Use default matplotlib settings
        rcdefaults()
    elif style == 'preserve':
        pass
    else:
        raise ValueError('Input style must be one of: "custom", "eureka", "preserve", or "default".')

    rc('text', usetex=usetex) # TeX fonts may not work on all machines. 
