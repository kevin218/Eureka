class Logedit:
    """This object handles writing text outputs into a log file and to
    the screen as well.

    Examples
    --------
    .. highlight:: python
    .. code-block:: python

        >>> from logedit import Logedit
        >>> message1 = 'This message will be logged and displayed.'
        >>> message2 = 'This message too.'
        >>> message3 = 'This one is Not going to delete previous lines.'
        >>> message4 = ('This one copies previous lines and keeps previous '
                        'log, but saves to new log.')
        >>> message5 = 'This one deletes previous lines.'

        >>> logname = 'out.log'
        >>> logname2 = 'out2.log'

        >>> # Create and print lines to a log
        >>> log = Logedit(logname)
        >>> log.writelog(message1)
        This message will be logged and displayed.
        >>> log.writelog(message2)
        This message too.
        >>> log.closelog()

        >>> # Edit log without overiding previous lines
        >>> log = Logedit(logname, read=logname)
        >>> log.writelog(message3)
        This one is Not going to delete previous lines.
        >>> log.closelog()

        >>> # copy a pre-existing log on a new log, and edit it.
        >>> log = Logedit(logname2, read=logname)
        >>> log.writelog(message4)
        This one copies previous lines and keeps previous log, but saves to
        new log.
        >>> log.closelog()

        >>> # overite a pre-existing log
        >>> log = Logedit(logname)
        >>> log.writelog(message5)
        This one deletes previous lines.
        >>> log.closelog()

        >>> # See the output files: 'out.log' and 'out2.log' to see results.

    Notes
    -----
    History:

    - 2010-07-10 Patricio Cubillos
        Initial version
    - 2010-11-24 Patricio Cubillos
        logedit converted to a class.
    """

    def __init__(self, logname, read=None):
        """Creates a new log file with name logname. If a logfile is
        specified in read, copies the content from that log.

        Parameters
        ----------
        logname : str
            The name of the file where to save the log.
        read : str
            Name of an existing logfile. If specified, its content
            will be written to the log.
        """
        # Read from previous log
        content = []
        if read is not None:
            try:
                old = open(read, 'r')
                content = old.readlines()
                old.close()
            except:
                # FINDME: Need to only catch the expected exception
                pass

        # Initiate log
        self.logname = logname
        self.log = open(self.logname, 'w')

        # Append content if there is something
        if content != []:
            self.log.writelines(content)

    def writelog(self, message, mute=False, end='\n'):
        r"""Prints message in the terminal and stores it in the log file.

        Parameters
        ----------
        message : str
            The message to log.
        mute : bool; optional
            If True, only log and do not pring. Defaults to False.
        end : str; optional
            Can be set to '\r' to have the printed line overwritten which
            is useful for progress bars. Defaults to '\n'.
        """
        # print to screen:
        if not mute:
            print(message, end=end, flush=True)
        # print to file:
        try:
            print(message, file=self.log, flush=True)
        except:
            # The file got closed, try reopening it and logging again
            try:
                self.log = open(self.logname, 'a')
                print(message, file=self.log, flush=True)
            except:
                print('ERROR: Unable to write to log file')

    def closelog(self):
        """Closes an existing log file."""
        self.log.close()

    def writeclose(self, message, mute=False, end='\n'):
        r"""Print message in terminal and log, then close log.

        Parameters
        ----------
        message : str
            The message to log.
        mute : bool; optional
            If True, only log and do not pring. Defaults to False.
        end : str; optional
            Can be set to '\\r' to have the printed line overwritten which
            is useful for progress bars. Defaults to '\\n'.
        """
        self.writelog(message, mute, end)
        self.closelog()
