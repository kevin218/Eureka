import os
import numpy as np
from astropy import units
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table

from .lib.tracing_niriss import mask_method_edges, mask_method_profile, ref_file
from .lib.masking        import (interpolating_row, data_quality_mask,
                              interpolating_image)
from .lib.clipping       import time_removal
from .S3_data_reduction.background     import bkg_sub, fitbg3
from .S3_data_reduction.niriss_extraction   import (dirty_mask, box_extract,
                                              optimal_extraction_routine)
from .S3_data_reduction.niriss  import wave_NIRISS as wavelength
from .lib.simultaneous_order_fitting import fit_orders, fit_orders_fast


__all__ = ['NIRISS_S3']


class NIRISS_S3(object):

    def __init__(self, filename, f277_filename=None,
                 data_dir=None, output_dir=None):
        """
        Initializes the NIRISS S3 data reduction class.

        Parameters
        ----------
        filename : str
           The name of the FITS file with NIRISS observations.
        data_path : str, optional
           The path to where the input FITS files are stored. Default
           is None. If None, will search the current working directory
           for the files.
        output_dir : str, optional
           The path where output files will be saved. Default is None.
           if None, will save all files to the current working directory.

        Attributes
        ----------
        filename : str
           The science file name.
        data_path : str
           The path to where the science file is stored.
        output_dir : str
           The path to where output files will be stored.
        """
        self.filename = filename

        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = os.getcwd()

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = os.getcwd()

        # Opens the science FITS file and sets up all proper
        #   data attributes for the reduction steps.
        self.setup()

        self.dq_mask = data_quality_mask(self.dq)

        if f277_filename is not None:
            self.setup_f277(f277_filename)
        else:
            self.f277 = None
            print('Without F277W filter image, some functions may not be available.')

        self.tab1     = None
        self.tab2     = None
        self.bkg      = None
        self.box_mask = None
        self.box_var1 = None
        self.box_var2 = None
        self.bkg_removed  = None
        self.box_spectra1 = None
        self.box_spectra2 = None
        self.box_mask_separate = None

        return


    def setup(self):
        """
        Sets up all proper attributes from the FITS file.

        Attributes
        ----------
        mhdr : FITS header
           The main header in the FITS file.
        shdr : FITS header
           The header associated with the science frame
           extension.
        intend : float
        time : np.ndarray
           Array of time values based on the exposure start
           and stop times indicated in the FITS header.
        time_units : str
           The units `self.time` is in.
        inttime : float
           The effective integration time (in seconds).
        data : np.ndarray
           3D array of science group images.
        err : np.ndarray
           3D array of errors associated with the science
           frames.
        dq : np.ndarray
           3D array of data quality masks, indicating the
           location of bad pixels.
        var : np.ndarray
           3D array of poisson estimated variances.
        v0 : np.ndarray
           3D array of rnoise estimated variances.
        meta : np.ndarray
           The array of ASDF additional meta data within
           the FITS file.
        median : np.ndarray
           Median frame of all science images.
        """
        hdu = fits.open(os.path.join(self.data_dir, self.filename))

        self.mhdr = hdu[0].header        # Sets in the meta data header
        self.shdr = hdu['SCI',1].header  # Sets the science data header

        self.intend = hdu[0].header['NINTS'] + 0.0
        self.time   = np.linspace(self.mhdr['EXPSTART'],
                                  self.mhdr['EXPEND'],
                                  int(self.intend) )

        self.time_units = 'BJD_TDB'
        self.inttime  = hdu[0].header['EFFINTTM']

        # Loads all the data in
        self.data = hdu['SCI',1].data + 0.0
        self.raw_data = hdu['SCI',1].data + 0.0
        self.err  = hdu['ERR',1].data + 0.0
        self.dq   = hdu['DQ', 1].data + 0.0

        self.var  = hdu['VAR_POISSON',1].data * self.inttime**2.0
        self.v0   = hdu['VAR_RNOISE', 1].data * self.inttime**2.0

        self.meta = hdu[-1].data

        # Removes NaNs from the data & error/variance arrays
        self.data[np.isnan(self.data)==True] = 0.0
        self.err[ np.isnan(self.err )==True] = 0.0
        self.var[ np.isnan(self.var )==True] = 0.0
        self.v0[  np.isnan(self.v0  )==True] = 0.0

        print(hdu['DQ', 1].data.shape)

        self.median = np.nanmedian(self.data, axis=0)
        hdu.close()

        return

    def setup_f277(self, filename):
        """
        Opens and assigns proper attributes for the F277W
        filter observations.

        Parameters
        ----------
        filename : str
           The name of the F277W FITS file.

        Attributes
        ----------
        f277 : np.ndarray
           Science images from this filter.
        """
        hdu = fits.open(os.path.join(self.data_dir,
                                     filename) )
        self.f277 = hdu[1].data + 0.0

        hdu.close()
        return


    def clean_up(self):
        """
        The `clean_up` routine interpolated over bad pixel values,
        which are marked in the data quality images (`self.dq`).
        This routine removes bad quality pixels from the following
        images:
           - `self.data`
           - `self.err`
           - `self.var`
        """
        print('Cleaning data . . .')
        self.data = interpolating_image(self.data, mask=self.dq)
        self.median = np.nanmedian(self.data, axis=0)
        print('Cleaning error . . .')
        self.err  = interpolating_image(self.err,  mask=self.dq)
        print('Cleaning variance . . .')
        self.var  = interpolating_image(self.var,  mask=self.dq)
        self.median = interpolating_image(self.median, mask=np.nanmedian(self.dq,axis=0))


    def map_wavelength(self, orders=[1,2,3]):
        """
        Retrieves the 2D wavelength maps for the first and
        second orders.

        Parameters
        ----------
        orders : np.array, list, optional
           A list of which orders to retrieve the wavelength
           solutions for. Default is [1,2,3]. Options are any
           single order or combinations of these orders.

        Attributes
        ----------
        wavelength_map : np.ndarray
        """
        wmap = wavelength(os.path.join(self.data_dir, self.filename),
                          orders, inclass=True)
        self.wavelength_map = wmap + 0.0


    def map_trace(self, method='profile', ref_filename=None, isplots=0):
        """
        Calculates the trace of the first and second NIRISS
        orders.

        Parameters
        ----------
        method : str, optional
           Decision on which trace extraction routine to run.
           Options are: `edges` (uses a canny-edge detection
           routine), `centers` (uses the spatial profile), and
           `ref` (uses the STScI JWST reference frame).
        ref_filename : str, optional
           The name of the reference frame containing the order
           trace x,y position values. Default is None. This is
           a required parameter if you are running `method=='ref'`.

        Attributes
        ----------
        tab1 : astropy.table.Table
           Astropy table with x,y coordinates for each order.
           `tab1` is initialized when using the method `edges`.
        tab2 : astropy.table.Table
           Astropy table with x,y coordinates for each order.
           `tab2` is initialized when using the method `centers`.
        """
        if method.lower() == 'edges':
            if self.f277 is not None:
                self.tab1 = mask_method_edges(self, isplots=isplots)
            else:
                return('Need F277W filter to run this trace finding method.')

        elif method.lower() == 'profile':
            self.tab2 = mask_method_profile(self, isplots=isplots)

        elif method.lower() == 'ref':
            self.tab3 = ref_file(ref_filename)

        else:
            return('Trace method not implemented. Options are `edges` and `centers`.')


    def create_box_mask(self, boxsize1=60, boxsize2=50, boxsize3=40,
                        booltype=True,
                        return_together=True):
        """
        Creates a box mask to extract the first and second NIRISS orders.
        Can set different box sizes for each order and also return a single
        mask with both orders (`return_together==True`) or return masks for each
        order (`return_together==False`).

        Parameters
        ----------
        boxsize1 : int, optional
           Box size for the first order. Default is 60 pixels.
        boxsize2 : int, optional
           Box size for the second order. Default is 50 pixels.
        booltype : bool, optional
           Sets the dtype of the mask array. Default is True
           (returns array of boolean values).
        return_together : bool, optional
           Determines whether or not to return one combined
           box mask or masks for both orders. Default is True
           (returns 2 separate masks).

        Attributes
        ----------
        boxsize1 : int
           The box size for the first order.
        boxsize2 : int
           The box size for the second order.
        box_mask : np.ndarray
           Attribute for a combined box mask per each order. Created
           when `return_together == True`.
        box_mask_separate : np.ndarray
           Attribute for separate box masks per each order. Created
           when `return_together == False`.
        """
        if self.tab2 is not None:
            t = self.tab2
        elif self.tab1 is not None:
            t = self.tab1
        else:
            return('Need to run the trace identifier to create the box mask.')

        out = dirty_mask(self.median, t,
                         boxsize1=boxsize1,
                         boxsize2=boxsize2,
                         boxsize3=boxsize3,
                         booltype=booltype,
                         return_together=return_together)
        if return_together == True:
            self.box_mask = out
        else:
            self.box_mask_separate = np.array(out)

        self.boxsize1 = boxsize1
        self.boxsize2 = boxsize2
        self.boxsize3 = boxsize3

        return


    def extract_box_spectrum(self):
        """
        Extracts spectra using the box mask.

        Attributes
        ----------
        box_spectra1 : np.ndarray
           Box extracted spectra for the first order.
        box_spectra2 : np.ndarray
           Box extracted spectra for the second order.
        box_var1 : np.ndarray
           Box extracted variance for the first order.
        box_var2 : np.ndarray
           Box extracted variance for the second order.
        """
        if self.box_mask_separate is None:
            self.create_box_mask(return_together=False,
                                 booltype=False)

        if self.bkg_removed is not None:
            d = self.bkg_removed + 0.0
        else:
            d = self.data + 0.0

        s, v = box_extract(d, self.var, self.box_mask_separate)

        self.box_var1     = v[0] + 0.0
        self.box_var2     = v[1] + 0.0
        self.box_var3     = v[2] + 0.0

        self.box_spectra1 = s[0] + 0.0
        self.box_spectra2 = s[1] + 0.0
        self.box_spectra3 = s[2] + 0.0

        return


    def fit_background(self, readnoise=11, sigclip=[4,4,4],
                       box=(5,2), filter_size=(2,2),
                       bkg_estimator=['median'], test=True):
        """
        Subtracts background from non-spectral regions.

        Parameters
        ----------
        data : object
        meta : object
        readnoise : float, optional
           An estimation of the readnoise of the detector.
           Default is 5.
        sigclip : list, array, optional
           A list or array of len(n_iiters) corresponding to the
           sigma-level which should be clipped in the cosmic
           ray removal routine. Default is [4,2,3].

        Returns
        -------
        bkg : np.ndarray
        bkg_var : np.ndarray
        cr_mask : np.ndarray
        bkg_removed : np.ndarray
        """
        if self.box_mask is None:
            self.create_box_mask(return_together=True, booltype=True)

        if test is True:
            ind = 5
        else:
            ind = len(self.data)


        bkg, bkg_var, cr_mask = fitbg3(self.data[:ind],
                                       ~self.box_mask,
                                       readnoise=readnoise,
                                       sigclip=sigclip,
                                       bkg_estimator=bkg_estimator,
                                       box=box,
                                       filter_size=filter_size,
                                       inclass=True)

        self.bkg = bkg + 0.0
        self.bkg_var = bkg_var + 0.0
        self.bkg_removed = cr_mask - bkg + 0.0

        m = np.zeros(cr_mask.shape)
        x,y,z = np.where(np.isnan(cr_mask)==True)
        m[x,y,z] = 1
        self.cr_mask = m + 0

        m = np.zeros(self.bkg_removed.shape)
        x,y,z = np.where(np.isnan(self.bkg_removed)==True)
        m[x,y,z] = 1
        self.bkg_removed = interpolating_image(self.bkg_removed,
                                               mask=m)



    def optimal_extraction(self, proftype='median', sigma=20, Q=1.8,
                           per_quad=True, test=False):
        """
        Runs the optimal extraction routine for the NIRISS orders.
        There is a lot of flexibility in this routine, so please read
        the options carefully. There are 2 options for extracting the
        spectra:
        1. Extracting the orders via quadrants. This will remove the
        first and second orders on the righthand side of the image
        (no contamination) first, then extract the overlapping contaminated
        region together.
        2. Extracting the orders by orders. This will extract the
        *entire* first order and the *entire* second order, including
        the overlapping contaminated region for both orders.

        Parameters
        ----------
        proftype : str, optional
           Which type of profile to use to extract the orders.
           Default is `median` or a median-created profile. Other options
           include `gaussian` (which is a Gaussian profile shape) and
           `moffat` (which is a Moffat profile shape).
        sigma : float, optional
           What sigma to look for when removing outliers during the optimal
           extraction routine. Default = 20.
        Q : float, optional
           An estimate on the gain. Default = 1.8.
        per_quad : bool, optional
           Whether to extract the spectra per quadrants (method 1 above)
           or as a full order (method 2 above). Default is `True` (will
           extract spectra via quadrants).
        test : bool, optional
           Whether to run a few test frames or run the entire dataset. Default
           is False (will run all exposures). If `True`, will run the middle 5
           exposures.

        Attributes
        ----------
        opt_order1_flux : np.array
           Optimally extracted flux for the first order.
        opt_order2_flux : np.array
           Optimally extracted flux for the second order.
        opt_order1_err : np.array
           Optimally extracted flux error for the first order.
        opt_order2_err : np.array
           Optimally extracted flux error for the second order.
        """
        if self.box_spectra1 is None:
            self.extract_box_spectrum()

        if self.tab2 is not None:
            pos1 = self.tab2['order_1']
            pos2 = self.tab2['order_2']
        elif self.tab1 is not None:
            pos1 = self.tab1['order_1']
            pos2 = self.tab1['order_2']
        else:
            self.map_trace()
            pos1 = self.tab2['order_1']
            pos2 = self.tab2['order_2']

        if test == True:
            start, end = 0,5#int(len(self.data)/2-2), int(len(self.data)/2+2)
        else:
            start, end = 0, len(self.data)

        cr_mask = ~np.array(self.cr_mask, dtype=bool)

        all_fluxes, all_errs, all_profs = optimal_extraction_routine(self.data[start:end],
                                                                     self.var[start:end],
                                                                     spectrum=np.array([self.box_spectra1[start:end],
                                                                                        self.box_spectra2[start:end]]),
                                                                     spectrum_var=np.array([self.box_var1[start:end],
                                                                                            self.box_var2[start:end]]),
                                                                     sky_bkg=self.bkg[start:end],
                                                                     medframe=self.median,
                                                                     cr_mask=self.bkg_removed,
                                                                     pos1=pos1,
                                                                     pos2=pos2,
                                                                     sigma=sigma,
                                                                     Q=Q,
                                                                     proftype=proftype,
                                                                     per_quad=per_quad)
        return all_fluxes, all_errs, all_profs
