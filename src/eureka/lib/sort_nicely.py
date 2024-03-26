import re


def tryint(s):
    """Turn a string into an int if possible.

    Parameters
    ----------
    s : str
        The string to try to convert to an int.

    Returns
    -------
    int OR str
        An int if s was numeric, otherwise just returns s.
    """
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
       "z23a" -> ["z", 23, "a"]

    Parameters
    ----------
    s : str
        The string to break into a list.

    Returns
    -------
    list
        The string broken into a list of strings and ints.
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(list1):
    """Sort the given list in the way that humans expect.

    Parameters
    ----------
    list1 : list
        The list to sort nicely.

    Returns
    -------
    list
        The nicely sorted list.
    """
    list1.sort(key=alphanum_key)
    return list1
