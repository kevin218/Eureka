import numpy as np
import urllib
import os
import re
import time


def leapdates(rundir, log):
    '''Generates an array of leap second dates.

    The array is automatically updated every six months.
    Uses a local leap second file, but retrieves a leap
    second file from NIST if the current file is missing
    or out of date.

    Parameters
    ----------
    rundir : str
        The folder in which to cache the leapdates array.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    ndarray
        The Julian dates of leap seconds.
    '''
    ntpepoch = 2208988800
    if rundir[-1] != os.sep:
        rundir += os.sep
    if not os.path.isdir(rundir):
        # Make the leapdir folder if needed
        os.mkdir(rundir)
    files = os.listdir(rundir)
    if len(files) != 0:
        recent = np.sort(files)[-1]
        with open(rundir+recent, 'r') as nist:
            doc = nist.read()
        table = doc.split('#@')[1].split('\n#\n')[1].split('\n')
        expiration = float(doc.split('#@')[1].split('\n')[0][1:])
    else:
        expiration = -np.inf
    if time.time() + ntpepoch > expiration:
        log.writelog("  Leap-second file expired. Retrieving new file.",
                     mute=True)
        try:
            with urllib.request.urlopen('ftp://ftp.boulder.nist.gov/'
                                        'pub/time/leap-seconds.list') as nist:
                doc = nist.read().decode()
            use_fallback = False
        except:
            # FINDME: Need to only catch the expected exception.
            # Couldn't connect to the internet, so use the local array
            # defined below
            use_fallback = True

        if not use_fallback:
            newexp = doc.split('#@')[1].split('\n')[0][1:]
            # Remove non-alphanumeric characters with regular expressions
            newexp = re.sub(r'\W+', '', newexp)
            with open(rundir+"leap-seconds."+newexp, 'w') as newfile:
                newfile.write(doc)
            table = doc.split('#@')[1].split('\n#\r\n')[1].split('\n')
            log.writelog("  Leap second file updated.", mute=True)
    else:
        use_fallback = False
        log.writelog("  Local leap second file retrieved.", mute=True)
        t_next = time.asctime(time.localtime(expiration-ntpepoch))
        log.writelog(f"  Next update: {t_next}", mute=True)

    if not use_fallback:
        ls = np.zeros(len(table))
        for i in range(len(table)):
            ls[i] = float(table[i].split('\t')[0])
        jd = ls/86400+2415020.5
        return jd
    else:
        log.writelog('  NIST leap-second file not available. '
                     'Using stored table.')

        return np.array([2441316.5, 2441498.5, 2441682.5, 2442047.5,
                         2442412.5, 2442777.5, 2443143.5, 2443508.5,
                         2443873.5, 2444238.5, 2444785.5, 2445150.5,
                         2445515.5, 2446246.5, 2447160.5, 2447891.5,
                         2448256.5, 2448803.5, 2449168.5, 2449533.5,
                         2450082.5, 2450629.5, 2451178.5, 2453735.5,
                         2454831.5, 2456108.5, 2457203.5, 2457753.5])+1


def leapseconds(jd_utc, dates):
    '''Computes the difference between UTC and TT for a given date.

    Parameters
    ----------
    jd_utc : float
        UTC Julian date.
    dates : array_like
        An array of Julian dates on which leap seconds occur.

    Returns
    -------
    float
        The difference between UTC and TT for a given date.
    '''
    utc_tai = len(np.where(jd_utc > dates)[0])+10-1
    tt_tai = 32.184
    return tt_tai + utc_tai


def utc_tt(jd_utc, leapdir, log):
    '''Converts UTC Julian dates to Terrestrial Time (TT).

    Parameters
    ----------
    jd_utc : array-like
        UTC Julian date.
    leapdir : str
        The folder containing leapdir save files.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    array-like
        Time in TT.
    '''
    dates = leapdates(leapdir, log)
    if len(jd_utc) > 1:
        dt = np.zeros(len(jd_utc))
        for i in range(len(jd_utc)):
            dt[i] = leapseconds(jd_utc[i], dates)
    else:
        dt = leapseconds(jd_utc, dates)
    return jd_utc+dt/86400.


def utc_tdb(jd_utc, leapdir, log):
    '''Converts UTC Julian dates to Barycentric Dynamical Time (TDB).

    Formula taken from USNO Circular 179, based on that found in Fairhead
    and Bretagnon (1990). Accurate to 10 microseconds.

    Parameters
    ----------
    jd_utc : array-like
        UTC Julian date.
    leapdir : str
        The folder containing leapdir save files.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    array-like
        time in JD_TDB.
    '''
    jd_tt = utc_tt(jd_utc, leapdir, log)
    T = (jd_tt-2451545.)/36525
    jd_tdb = jd_tt + (0.001657*np.sin(628.3076*T + 6.2401) +
                      0.000022*np.sin(575.3385*T + 4.2970) +
                      0.000014*np.sin(1256.6152*T + 6.1969) +
                      0.000005*np.sin(606.9777*T + 4.0212) +
                      0.000005*np.sin(52.9691*T + 0.4444) +
                      0.000002*np.sin(21.3299*T + 5.5431) +
                      0.000010*T*np.sin(628.3076*T + 4.2490))/86400.
    return jd_tdb
