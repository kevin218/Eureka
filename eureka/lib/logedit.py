

class Logedit:
  """
    This object handles writing text outputs into a log file and to
    the screen as well.

    Class methods:
    --------------
    init(logname, read=None):
        Creates a log instance into file logname.
        If read specified copies it's content into log.

    writelog(message):
        Prints message to terminal and to the log file.

    closelog:
        Closes an existing log file.

    writeclose(message):
        Prints message to terminal and to log, then closes the log file.

    Examples:
    ---------
    >>> from logedit import Logedit
    >>> message1 = 'This message will be logged and displayed.'
    >>> message2 = 'This message too.'
    >>> message3 = 'This one is Not going to delete previous lines.'
    >>> message4 = 'This one copies previous lines and keeps previous log, but saves to new log.'
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
    This one copies previous lines and keeps previous log, but saves to new log.
    >>> log.closelog()

    >>> # overite a pre-existing log
    >>> log = Logedit(logname)
    >>> log.writelog(message5)
    This one deletes previous lines.
    >>> log.closelog()

    >>> # See the output files: 'out.log' and 'out2.log' to see the results.

    Revisions
    ---------
    2010-07-10  patricio   Writen by Patricio Cubillos.
                           pcubillos@fulbrightmail.org
    2010-11-24  patricio   logedit converted to a class.
  """

  def __init__(self, logname, read=None):
    """
      Creates a new log file with name logname. If a logfile is
      specified in read, copies the content from that log.

      Parameters:
      -----------
      logname: String
               The name of the file where to save the log.
      read:    String
               Name of an existing logfile. If specified, its content
               will be written to the log.
    """
    # Read from previous log
    content = []
    if read != None:
      try:
        old = open(read, 'r')
        content = old.readlines()
        old.close()
      except:
        pass

    # Initiate log
    self.log = open(logname, 'w')

    # Append content if there is something
    if content != []:
      self.log.writelines(content)


  def writelog(self, message, mute=False, end='\n'):
    """
      Prints message in the terminal and stores it in the log file.
    """
    # print to screen:
    if not mute:
      print(message, end=end, flush=True)
    # print to file:
    print(message, file=self.log, flush=True)


  def closelog(self):
    """
      Closes an existing log file.
    """
    self.log.close()


  def writeclose(self, message, mute=False, end='\n'):
    """
      Print message in terminal and log, then close log.
    """
    self.writelog(message, mute, end)
    self.closelog()
