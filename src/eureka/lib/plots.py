import os
import matplotlib
from matplotlib import rcdefaults, rcParams, font_manager
from matplotlib import style as mpl_style
from functools import wraps

# Global configuration dictionary (used by decorator and set_rc)
_current_style = {
    "style": None,  # None = not explicitly set yet
    "usetex": None,
    "layout": "constrained",
    "backend": None,
    "filetype": ".png",
    "from_scratch": False,
    "kwargs": {},
}


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
        raise ValueError('"usetex" and "from_scratch" arguments must be '
                         'boolean or None.')

    _current_style.update({
        "style": style,
        "usetex": usetex,
        "layout": layout,
        "backend": backend,
        "filetype": filetype,
        "from_scratch": from_scratch,
        "kwargs": kwargs
    })

    _set_style()


def _set_style():
    """Apply the currently selected style to matplotlib rcParams."""
    style = _current_style["style"]
    usetex = _current_style["usetex"]
    layout = _current_style["layout"]
    backend = _current_style["backend"]
    from_scratch = _current_style["from_scratch"]
    kwargs = _current_style["kwargs"]

    if backend is not None:
        matplotlib.use(backend)

    if from_scratch:
        rcdefaults()

    if style == 'custom':
        rcParams.update(**kwargs)
    elif style == 'eureka':
        style_path = os.path.join(os.path.dirname(__file__), 'eureka.mplstyle')
        mpl_style.use(style_path)

        # After loading the style, sanitize the font list
        have_stix2 = any("STIX Two Text" in f.name
                         for f in font_manager.fontManager.ttflist)

        if not have_stix2:
            # Remove STIX Two Text from font.family / font.serif if it's first
            families = rcParams["font.family"]
            if isinstance(families, str):
                families = [families]
            families = [f for f in families if "STIX Two Text" not in f]
            if not families:  # ensure we have a fallback
                families = ["STIXGeneral", "DejaVu Serif", "serif"]
            rcParams["font.family"] = families

            serifs = rcParams["font.serif"]
            if isinstance(serifs, str):
                serifs = [serifs]
            serifs = [f for f in serifs if "STIX Two Text" not in f]
            if not serifs:  # ensure we have a fallback
                serifs = ["STIXGeneral", "DejaVu Serif", "serif"]
            rcParams["font.serif"] = serifs
    elif style == 'default':
        rcdefaults()
    elif style == 'preserve' or style is None:
        pass
    else:
        raise ValueError('Input style must be one of: "custom", "eureka", '
                         '"preserve", or "default".')

    if usetex is not None:
        rcParams.update({'text.usetex': usetex})

    if layout == 'constrained':
        rcParams.update({'figure.constrained_layout.use': True})
    elif layout == 'tight':
        rcParams.update({'figure.autolayout': True})


def apply_style(func):
    """Decorator to apply the current or default Eureka matplotlib style."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if _current_style["style"] is None:
            _current_style["style"] = "eureka"
        with matplotlib.rc_context():
            _set_style()
            return func(*args, **kwargs)
    return wrapper


def get_filetype():
    """Return the currently selected figure filetype (e.g., '.png', '.pdf').

    Returns
    -------
    str
        The file extension to use when saving plots, such as '.png' or '.pdf'.
    """
    return _current_style["filetype"]
