# $Author: ccampo $
# $Revision: 31 $
# $Date: 2009-06-18 09:55:27 -0400 (Thu, 18 Jun 2009) $
# $HeadURL: file:///home/esp01/svn/code/auto_aor/trunk/julday.py $
# $Id: julday.py 31 2009-06-18 13:55:27Z ccampo $

def julday(month, day, year, hour=12, minute=0, second=0):
    '''
NAME:
      julday

PURPOSE:
      Calculate the Julian Date Number for a given month, day, and year.
      Can also take in hours, minutes, and seconds.

INPUTS:
      month:  Number of the month of the year (1 = jan, ...,  12 = dec)

      day:    Number of the day of the month.

      year:   Number of the desired year.  Year parameters must be valid
              values fron the civil calendar.  Years B.C.E. are represented
              as negative integers.  Years in the common era are represented
              as positive integers.  In particular, note that there is no year
              0 in the civil calendar.  1 B.C.E. (-1) is followed by 1 C.E. (1).

      hour:   Number of the hour of the day.

      minute: Number of the minute of the hour.

      second: Number of the second of the minute.

      round:  Optional; if true, round to one decimal place

OUTPUTS:
     Julian Day Number (which begins at noon) of the specified calendar date is
     returned in double precision format.

SIDE EFFECTS:
     None.

NOTES:
     If a time is not given, the routine defaults to noon.
     
     Adopted from julday.pro; JULDAY is a standard routine in IDL.
     This is roughly equivalent to the IDL routine, with slightly
     more precision in results.

MODIFICATION HISTORY:
     2009-01-06   0.1    Christopher Campo, UCF    Initial version
                         ccampo@gmail.com
    '''
    import numpy as np

    # catches a wrong date input
    if month > 12 or month < 1 or day > 31 or day < 1:
        raise(ValueError, 'Error: Date does not exist. Check the input...')
    
    # Gregorian to Julian conversion formulae; wikipedia
    a = np.floor((14-month)/12.)
    y = year + 4800 - a
    m = month + (12*a) - 3

    jdn = day + np.floor(((153*m) + 2)/5.) + 365*y + np.floor(y/4.)\
        - np.floor(y/100.) + np.floor(y/400.) - 32045

    jd = jdn + ((hour-12)/24.) + (minute/1440.) + (second/86400.)

    return jd

