import numpy as np
from astropy.io import fits
import astraeus.xarrayIO as xrio
from . import nircam
from ..lib.util import read_time


def read(filename, data, meta, log):
    '''Reads single FITS file from JWST's MIRI instrument.

    Parameters
    ----------
    filename : str
        Single filename to read.
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with the fits data stored inside.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Notes
    -----
    History:

    - Nov 2012 Kevin Stevenson
        Initial Version
    - May 2021  Kevin Stevenson
        Updated for NIRCam
    - Jun 2021  Taylor Bell
        Updated docs for MIRI
    - Jun 2021  Sebastian Zieba
        Updated for MIRI
    - Apr 2022  Sebastian Zieba
        Updated wavelength array
    - Apr 21, 2022 Kevin Stevenson
        Convert to using Xarray Dataset
    '''
    hdulist = fits.open(filename)

    # Load main and science headers
    data.attrs['filename'] = filename
    data.attrs['mhdr'] = hdulist[0].header
    data.attrs['shdr'] = hdulist['SCI', 1].header
    data.attrs['intstart'] = data.attrs['mhdr']['INTSTART']-1
    data.attrs['intend'] = data.attrs['mhdr']['INTEND']

    sci = hdulist['SCI', 1].data
    err = hdulist['ERR', 1].data
    dq = hdulist['DQ', 1].data
    v0 = hdulist['VAR_RNOISE', 1].data
    # If wavelengths are all zero --> use hardcoded wavelengths
    # Otherwise use the wavelength array from the header
    if np.all(hdulist['WAVELENGTH', 1].data == 0):
        if meta.firstFile:
            log.writelog('  WARNING: The wavelength for the simulated MIRI '
                         'data are currently hardcoded because they are not '
                         'in the .fits files themselves')
        wave_2d = np.tile(wave_MIRI_hardcoded(), (sci.shape[2], 1))[:, ::-1]
    else:
        wave_2d = hdulist['WAVELENGTH', 1].data
    int_times = hdulist['INT_TIMES', 1].data

    # Record integration mid-times in BJD_TDB
    if (hasattr(meta, 'time_file') and meta.time_file is not None):
        time = read_time(meta, data, log)
    elif len(int_times['int_mid_BJD_TDB']) == 0:
        if meta.firstFile:
            log.writelog('  WARNING: The timestamps for the simulated MIRI '
                         'data are currently hardcoded because they are not '
                         'in the .fits files themselves')
        if ('WASP_80b' in data.attrs['filename']
                and 'transit' in data.attrs['filename']):
            # Time array for WASP-80b MIRISIM transit observations
            # Assuming transit near August 1, 2022
            phase_i = 0.95434
            phase_f = 1.032726
            t0 = 2456487.425006
            per = 3.06785234
            time_i = phase_i*per+t0
            while np.abs(time_i-2459792.54237) > per:
                time_i += per
            time_f = phase_f*per+t0
            while time_f < time_i:
                time_f += per
            time = np.linspace(time_i, time_f, 4507,
                               endpoint=True)[data.attrs['intstart']:
                                              data.attrs['intend']-1]
        elif ('WASP_80b' in data.attrs['filename']
              and 'eclipse' in data.attrs['filename']):
            # Time array for WASP-80b MIRISIM eclipse observations
            # Assuming eclipse near August 1, 2022
            phase_i = 0.45434
            phase_f = 0.532725929856498
            t0 = 2456487.425006
            per = 3.06785234
            time_i = phase_i*per+t0
            while np.abs(time_i-2459792.54237) > per:
                time_i += per
            time_f = phase_f*per+t0
            while time_f < time_i:
                time_f += per
            time = np.linspace(time_i, time_f, 4506,
                               endpoint=True)[data.attrs['intstart']:
                                              data.attrs['intend']-1]
        elif 'new_drift' in data.attrs['filename']:
            # Time array for the newest MIRISIM observations
            time = np.linspace(0, 47.712*(1849)/3600/24, 1849,
                               endpoint=True)[data.attrs['intstart']:
                                              data.attrs['intend']-1]
        elif data.attrs['mhdr']['EFFINTTM'] == 10.3376:
            # There is no time information in the old simulated MIRI data
            # As a placeholder, I am creating timestamps indentical to the
            # ones in STSci-SimDataJWST/MIRI/Ancillary_files/times.dat.txt
            # converted to days
            time = np.linspace(0, 17356.28742796742/3600/24, 1680,
                               endpoint=True)[data.attrs['intstart']:
                                              data.attrs['intend']]
        elif data.attrs['mhdr']['EFFINTTM'] == 47.712:
            # A new manually created time array for the new MIRI simulations
            # Need to subtract an extra 1 from intend for these data
            time = np.linspace(0, 47.712*(42*44-1)/3600/24, 42*44,
                               endpoint=True)[data.attrs['intstart']:
                                              data.attrs['intend']-1]
        else:
            raise AssertionError('Eureka does not currently know how to '
                                 'generate the time array for these'
                                 'simulations.')
    else:
        time = int_times['int_mid_BJD_TDB']

    # Record units
    flux_units = data.attrs['shdr']['BUNIT']
    time_units = 'BJD_TDB'
    wave_units = 'microns'

    # MIRI appears to be rotated by 90° compared to NIRCam, so rotating arrays
    # to allow the re-use of NIRCam code. Having wavelengths increase from
    # left to right on the rotated frame makes life easier
    if data.attrs['shdr']['DISPAXIS'] == 2:
        sci = np.swapaxes(sci, 1, 2)[:, :, ::-1]
        err = np.swapaxes(err, 1, 2)[:, :, ::-1]
        dq = np.swapaxes(dq, 1, 2)[:, :, ::-1]
        v0 = np.swapaxes(v0, 1, 2)[:, :, ::-1]
        if not np.all(hdulist['WAVELENGTH', 1].data == 0):
            wave_2d = np.swapaxes(wave_2d, 0, 1)[:, :, ::-1]
        if (meta.firstFile and meta.spec_hw == meta.spec_hw_range[0] and
                meta.bg_hw == meta.bg_hw_range[0]):
            # If not, we've already done this and don't want to switch it back
            temp = np.copy(meta.ywindow)
            meta.ywindow = meta.xwindow
            meta.xwindow = sci.shape[2] - temp[::-1]

    data['flux'] = xrio.makeFluxLikeDA(sci, time, flux_units, time_units,
                                       name='flux')
    data['err'] = xrio.makeFluxLikeDA(err, time, flux_units, time_units,
                                      name='err')
    data['dq'] = xrio.makeFluxLikeDA(dq, time, "None", time_units,
                                     name='dq')
    data['v0'] = xrio.makeFluxLikeDA(v0, time, flux_units, time_units,
                                     name='v0')
    data['wave_2d'] = (['y', 'x'], wave_2d)
    data['wave_2d'].attrs['wave_units'] = wave_units

    return data, meta, log


def wave_MIRI_hardcoded():
    '''Compute wavelengths for simulated MIRI observations.

    This code contains the wavelength array for MIRI data. It was generated
    by using the jwst and gwcs packages to get the wavelength information out
    of the WCS.

    Returns
    -------
    lam_x_full : list
        A list of the wavelengths

    Notes
    -----
    History:

    - Apr 2022  Sebastian Zieba
        Initial Version
    '''
    # This array only contains the wavelength information for the BB
    lam_x = [np.nan, np.nan, 14.381619594576934, 14.366161703458102,
             14.350688919921913, 14.335201058149064, 14.319697932320251,
             14.304179356616153, 14.288645145217474, 14.273095112304905,
             14.257529072059135, 14.24194683866086, 14.226348226290767,
             14.210733049129553, 14.19510112135791, 14.179452257156534,
             14.163786270706106, 14.148102976187332, 14.132402187780896,
             14.11668371966749, 14.10094738602781, 14.085193001042548,
             14.069420378892394, 14.053629333758048, 14.03781967982019,
             14.02199123125952, 14.006143802256732, 13.990277206992516,
             13.974391259647563, 13.958485774402563, 13.94256056543822,
             13.926615446935214, 13.910650233074238, 13.894664738035996,
             13.878658776001165, 13.862632161150449, 13.846584707664539,
             13.830516229724124, 13.814426541509894, 13.798315457202545,
             13.782182790982771, 13.766028357031262, 13.74985196952871,
             13.733653442655811, 13.717432590593255, 13.701189227521732,
             13.684923167621939, 13.668634225074564, 13.652322214060304,
             13.635986948759847, 13.619628243353883, 13.603245912023116,
             13.586839768948224, 13.570409628309916, 13.553955304288866,
             13.537476611065783, 13.52097336282135, 13.504445373736257,
             13.487892457991201, 13.471314429766878, 13.454711103243975,
             13.438082292603182, 13.4214278120252, 13.404747475690716,
             13.388041097780418, 13.37130849247501, 13.354549473955174,
             13.337763856401612, 13.32095145399501, 13.304112080916056,
             13.287245551345451, 13.270351679463879, 13.25343027945204,
             13.236481165490625, 13.219504151760326, 13.202499052441837,
             13.185465681715847, 13.168403853763042, 13.15131338276413,
             13.134194082899791, 13.117045768350724, 13.09986825329762,
             13.082661351921168, 13.065424878402064, 13.048158646921001,
             13.030862471658665, 13.01353616679576, 12.996179546512966,
             12.978792424990983, 12.961374616410506, 12.943925934952217,
             12.926446194796814, 12.908935210124994, 12.891392795117444,
             12.873818763954858, 12.85621293081793, 12.838575109887344,
             12.820905115343798, 12.803202761367988, 12.785467862140605,
             12.767700231842339, 12.749899684653887, 12.732066034755931,
             12.714199096329175, 12.696298683554303, 12.678364610612013,
             12.660396691683, 12.642394740947948, 12.624358572587552,
             12.606288000782504, 12.588182839713502, 12.570042903561234,
             12.551868006506393, 12.533657962729674, 12.515412586411763,
             12.497131691733358, 12.47881509287515, 12.460462604017827,
             12.44207403934209, 12.423649213028623, 12.405187939258129,
             12.386690032211286, 12.368155306068797, 12.349583575011351,
             12.330974653219645, 12.312328354874364, 12.2936444941562,
             12.274922885245855, 12.256163342324017, 12.23736567957137,
             12.218529711168618, 12.199655251296448, 12.180742114135553,
             12.161790113866626, 12.14279906467036, 12.123768780727445,
             12.104699076218576, 12.085589765324443, 12.066440662225741,
             12.047251581103161, 12.028022336137397, 12.00875274150914,
             11.989442611399083, 11.970091759987909, 11.950700001456328,
             11.931267149985022, 11.911793019754683, 11.892277424946009,
             11.872720179739684, 11.853121098316409, 11.833479994856871,
             11.813796683541767, 11.794070978551783, 11.774302694067615,
             11.75449164426996, 11.734637643339502, 11.714740505456941,
             11.69480004480296, 11.674816075558264, 11.654788411903533,
             11.634716868019462, 11.614601258086752, 11.594441396286088,
             11.574237096798162, 11.553988173803672, 11.533694441483304,
             11.513355714017752, 11.492971805587715, 11.472542530373877,
             11.452067702556937, 11.431547136317578, 11.410980645836503,
             11.390368045294398, 11.369709148871959, 11.349003770749874,
             11.328251725108839, 11.307452826129548, 11.286606887992685,
             11.265713724878953, 11.24477315096904, 11.223784980443634,
             11.202749027483437, 11.181665106269135, 11.16053303098142,
             11.139352615800984, 11.118123674908524, 11.09684602248473,
             11.075519472710289, 11.0541438397659, 11.032718937832255,
             11.011244581090049, 10.98972058371997, 10.968146759902707,
             10.946522923818957, 10.924848889649411, 10.903124471574762,
             10.881349440027343, 10.859523338039219, 10.83764545390135,
             10.815715023042697, 10.793731280892219, 10.771693462878874,
             10.74960080443163, 10.727452540979442, 10.70524790795127,
             10.682986140776082, 10.660666474882833, 10.638288145700486,
             10.615850388657998, 10.593352439184342, 10.570793532708462,
             10.548172904659328, 10.525489790465903, 10.502743425557139,
             10.479933045362005, 10.457057885309458, 10.434117180828464,
             10.411110167347978, 10.388036080296965, 10.36489415510438,
             10.341683627199187, 10.318403732010351, 10.29505370496683,
             10.27163278149758, 10.24814019703157, 10.22457518699776,
             10.200936986825102, 10.17722483194256, 10.153437957779108,
             10.129575599763687, 10.105636993325268, 10.08162137389282,
             10.057527976895287, 10.03335603776164, 10.009104791920837,
             9.984773474801838, 9.960361321833613, 9.93586756844511,
             9.911291450065294, 9.88663220212313, 9.861889060047574,
             9.83706125926759, 9.812148035212134, 9.787148623310177,
             9.76206225899067, 9.736888177682578, 9.711625614814862,
             9.68627380581648, 9.660831986116396, 9.63529939114357,
             9.609675256326966, 9.583958817095533, 9.55814930887825,
             9.53224596710406, 9.506248027201938, 9.480154724600837,
             9.453965294729722, 9.427678973017546, 9.401294994893282,
             9.374812595785881, 9.348231011124312, 9.321549476337525,
             9.294767226854491, 9.26788349810416, 9.240897525515507,
             9.213808544517486, 9.186615790539054, 9.159318499009181,
             9.131915905356818, 9.104407245010927, 9.076791753400476,
             9.04906866595442, 9.021237218101726, 8.993296645271345,
             8.965246182892248, 8.937085066393387, 8.908812531203731,
             8.880427812752238, 8.851930146467865, 8.823318767779577,
             8.794592912116332, 8.765751814907095, 8.736794711580824,
             8.70772083756648, 8.678529428293027, 8.649219719189416,
             8.61979094568462, 8.590242343207592, 8.560573147187299,
             8.530782593052699, 8.500869916232748, 8.470834352156414,
             8.440675136252652, 8.410391308079337, 8.379980802008749,
             8.3494400623865, 8.318765209406738, 8.287952363263605,
             8.256997644151259, 8.22589717226384, 8.194647067795502,
             8.163243450940392, 8.131682441892654, 8.099960160846441,
             8.068072727995899, 8.03601626353518, 8.003786887658428,
             7.971380720559794, 7.938793882433426, 7.906022493473472,
             7.873062673874081, 7.839910543829399, 7.806562223533577,
             7.773013833180762, 7.739261492965103, 7.705301323080749,
             7.6711294437218465, 7.636741975082545, 7.602135037356991,
             7.567304750739336, 7.5322472354237275, 7.496958611604311,
             7.461434999475239, 7.425672519230658, 7.389667291064718,
             7.353415435171563, 7.316913071745345, 7.28015632098021,
             7.243141303070308, 7.2058641382097885, 7.168320946592797,
             7.130507848413484, 7.092420963865998, 7.054056413144487,
             7.0154103164430985, 6.976478793955981, 6.937257965877284,
             6.897743952401152, 6.8579328737217375, 6.817820850033189,
             6.777404001529653, 6.736677570223653, 6.695631585011559,
             6.65424835842772, 6.612508490409574, 6.570392580894557,
             6.527881229820109, 6.484955037123666, 6.441594602742667,
             6.397780526614547, 6.353493408676744, 6.3087138488667,
             6.263422447121848, 6.217599803379627, 6.171226517577476,
             6.12428318965283, 6.0767504195431306, 6.028608807185811,
             5.979838952518313, 5.930421455478071, 5.8803369160025225,
             5.829565934029109, 5.778089109495263, 5.725887042338427,
             5.672940332496037, 5.6192257699089785, 5.56469686872547,
             5.509271021633106, 5.452857529433617, 5.395365692928728,
             5.336704812920163, 5.27678419020966, 5.21551312559894,
             5.152800919889735, 5.088556873883774, 5.022690288382781,
             4.955110464188486, 4.8857016023133895, 4.814192198722521,
             4.740063284080301, 4.6627402025108715, 4.581648298138374,
             4.49621291508695, 4.4056133461146985, 4.307490381336023,
             4.1970102039552994, 4.068780911055409, 3.9174105997192363,
             3.737507367029659, np.nan, np.nan]
    # Including nans for out of BB area (eg for reference pixels) so that
    # length agrees with detector/subarray size The values here are based
    # on the simulated data.
    lam_x_full = np.array([np.float64(np.nan)]*int(7.0)+lam_x +
                          [np.float64(np.nan)]*int(416-397.0-1))

    return lam_x_full


def flag_bg(data, meta, log):
    '''Outlier rejection of sky background along time axis.

    Uses the code written for NIRCam which works for MIRI as long
    as the MIRI data gets rotated.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with outlier background pixels flagged.
    '''
    return nircam.flag_bg(data, meta, log)


def fit_bg(dataim, datamask, n, meta, isplots=0):
    """Fit for a non-uniform background.

    Uses the code written for NIRCam which works for MIRI as long
    as the MIRI data gets rotated.

    Parameters
    ----------
    dataim : ndarray (2D)
        The 2D image array.
    datamask : ndarray (2D)
        An array of which data should be masked.
    n : int
        The current integration.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    isplots : int; optional
        The plotting verbosity, by default 0.

    Returns
    -------
    bg : ndarray (2D)
        The fitted background level.
    mask : ndarray (2D)
        The updated mask after background subtraction.
    n : int
        The current integration number.
    """
    return nircam.fit_bg(dataim, datamask, n, meta, isplots=isplots)


def cut_aperture(data, meta, log):
    """Select the aperture region out of each trimmed image.

    Uses the code written for NIRCam which works for MIRI as long
    as the MIRI data gets rotated.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    apdata : ndarray
        The flux values over the aperture region.
    aperr : ndarray
        The noise values over the aperture region.
    apmask : ndarray
        The mask values over the aperture region.
    apbg : ndarray
        The background flux values over the aperture region.
    apv0 : ndarray
        The v0 values over the aperture region.

    Notes
    -----
    History:

    - 2022-06-17, Taylor J Bell
        Initial version based on the code in s3_reduce.py
    """
    return nircam.cut_aperture(data, meta, log)
