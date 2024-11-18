import matplotlib
from matplotlib import rcdefaults, rcParams

# Default figure file type
figure_filetype = '.png'


def set_rc(style='preserve', usetex=False, layout='constrained',
           backend=None, filetype='.png', from_scratch=False, **kwargs):
    """Function to adjust matplotlib rcParams for plotting procedures.

    Parameters
    ----------
    style : str; optional
        Your plotting style from ("custom", "eureka", "preserve", or
        "default"). Custom passes all kwargs to the 'font' rcParams
        group at the moment. Eureka sets some nicer rcParams settings
        recommended by the Eureka team. Preserve leaves all rcParams
        as-is and can be used to toggle the usetex parameter. By default
        uses 'preserve'.
    usetex : bool; optional
        Do you want to use LaTeX fonts (which requires LaTeX to be
        installed), by default False. Can also set to None to not change
        the current matplotlib setting.
    layout : str; optional
        Specifies the matplotlib.layout_engine that you want to use.
        Defaults to 'constrained' which is known to make nice plots.
        Can also try 'tight' (which uses the tight layout engine)
        or None, which uses the default layout engine, neither of
        which are tested or recommended.
    backend : bool; optional
        The Matplotlib backend you want to use. Defaults to None which
        will use whatever the result of `matplotlib.get_backend()` is.
    filetype : str
        The file type that all Eureka figures should be saved as
        (e.g. .png, .pdf).
    from_scratch : bool; optional
        Should the rcParams first be set to rcdefaults? By default False.
    **kwargs : dict
        Any additional parameters to be passed to rcParams.update.

    Raises
    ------
    ValueError
        Ensures that usetex and from_scratch arguments are boolean
    ValueError
        Ensures that input style is one of: "custom", "eureka",
        "preserve", or "default"
    """
    if not (isinstance(usetex, (bool, type(None))) and
            isinstance(from_scratch, bool)):
        raise ValueError('"usetex" and "from_scratch" arguments must '
                         'be boolean.')

    if from_scratch:
        # Reset all rcParams prior to making changes
        rcdefaults()

    # Apply the desired style...
    if style == 'custom':
        rcParams.update(**kwargs)
    elif style == 'eureka':
        # Apply default Eureka! font settings
        family = 'serif'
        fontfamily = ['Computer Modern Roman', *rcParams['font.serif']]
        fontsize = 14
        params = {'font.family': [family, ], 'font.'+family: fontfamily,
                  'font.size': fontsize, 'legend.fontsize': 11,
                  'mathtext.fontset': 'dejavuserif',
                  'mathtext.it': 'serif:italic',
                  'mathtext.rm': 'serif', 'mathtext.sf': 'serif',
                  'mathtext.bf': 'serif:bold',
                  'savefig.bbox': 'tight'}
        rcParams.update(params)
    elif style == 'default':
        # Use default matplotlib settings
        rcdefaults()
    elif style == 'preserve':
        pass
    else:
        raise ValueError('Input style must be one of: "custom", "eureka", '
                         '"preserve", or "default".')

    # TeX fonts may not work on all machines
    if usetex is not None:
        rcParams.update({'text.usetex': usetex})

    if layout == 'constrained':
        rcParams.update({'figure.constrained_layout.use': True})
    elif layout == 'tight':
        rcParams.update({'figure.autolayout': True})

    if backend is not None:
        matplotlib.use(backend)

    # Update the figure filetype
    global figure_filetype
    figure_filetype = filetype
