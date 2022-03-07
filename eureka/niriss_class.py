import os
import numpy as np
import matplotlib.pyplot as plt

from .S3_data_reduction import niriss
from .S3_data_reduction import background
from .S3_data_reduction import niriss_extraction
from .S3_data_reduction.s3_reduce import MetaClass, DataClass

__all__ = ['NIRISS_S3']

class NIRISS_S3(object):

    def __init__(self, filename, wavefile, f277_filename, 
                 path, outdir='.'):
        """
        Initializes the NIRISS class. Loads in the data
        and wavelength images.

        Parameters
        ----------
        filename : str
           The filename for the NIRISS images.
        wavefile : str
           The filename for the NIRISS wavelength images.
        f277_filename : str
           The filename for the F277W filter images.
        path : str
           The path for where all of the S2 output 
           images are stored.
        outdir : str, optional
           The path where all outputs will be saved to.
           Default is your current directory.

        Attributes
        ----------
        path : str
           The path for where all S2 images are stored.
        outdir : str
           The output path for files/images/etc.
        data : np.ndarray
           All NIRISS images from the `filename` file.
        median : np.ndarray
           Median frame for all NIRISS images.
        dq : np.ndarray
           Data quality array for all NIRISS images.
        err : np.ndarray
           Error array for all NIRISS images.
        f277 : np.ndarray
           Images from the F277W file.
        mhdr : FITS header
           The first header extension from the NIRISS images.
        shdr : FITS header
           The science extension header from the NIRISS images.
        time : np.ndarray
           The timestamps for each frame of the NIRISS images.
        time_units : str
           The units for the `time` array.
        wavelength_order1 : np.ndarray
           Wavelength image for the first NIRISS order.
        wavelength_order2 : np.ndarray
           Wavelength image for the second NIRISS order.
        wavelength_order3 : np.ndarray
           Wavelength image for the third NIRISS order.
        var : np.ndarray
           Poisson variance for the NIRISS images.
        v0 : np.ndarray
           Variance for the rnoise for the NIRISS images.
        """
        self.outdir = outdir
        self.path   = path

        data = DataClass()
        meta = MetaClass()

        data, meta = niriss.read(os.path.join(path, filename),
                                 os.path.join(path, f277_filename),
                                 data,
                                 meta)

        self.reassign_attrs(data)
        self.reassign_attrs(meta)

        # get wavelength solutions
        w1, w2, w3 = niriss.wave_NIRISS(os.path.join(path, wavefile), inclass=True)
        self.wavelength_order1 = w1 + 0.0
        self.wavelength_order2 = w2 + 0.0
        self.wavelength_order3 = w3 + 0.0

        self.bkg  = None
        self.tab1 = None
        self.tab2 = None
        self.box_mask = None

    def reassign_attrs(self, obj):
        """
        Reassigns attributes from one class into
        this one.

        Parameters
        ----------
        class : class
        """
        for attr in list(obj.__dict__):
            val = getattr(obj, attr)
            setattr(self, attr, val)
        return
        
        
    def identify_orders(self, method=1, save=False,
                        diagnose=False):
        """
        Identifies where the spectral orders are located.
        
        Parameters
        ----------
        method : int, optional
           Specifies which method to use to identify the
           orders. Default is 1. Method 1 uses image 
           processing and a canny-edge detector to find
           the orders. Method 2 uses the spatial profile.
        save : bool, optional
           An option to write the output tables to a `.csv`.
           Default is False.
        diagnose : bool, optional
           An option to see the order overplotted on the 
           median NIRISS image. Default is False.

        Attributes
        ----------
        tab1 : astropy.table.Table
           Astropy table with the center values for the
           first and second orders. tab1 used Method 1.
        tab2 : astropy.table.Table
           Astropy table with the center values for the 
           first and second orders. tab2 used Method 2.
        """
        def plot(t):
            # Diagnostic plotting for the order locations
            plt.figure(figsize=(14,4))
            plt.imshow(self.median, vmin=0,
                       vmax=np.nanpercentile(self.median, 85))
            plt.plot(t['x'], t['order_1'], 'k', lw=3,
                     label='First order')
            plt.plot(t['x'], t['order_2'], 'r', lw=3,
                     label='Second order')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=2, mode="expand", borderaxespad=0.)
            plt.ylim(self.median.shape[0], 0)
            plt.show()

        # Routine for the first method of order identification
        if method==1:
            self.tab1 = niriss.mask_method_one(self, save=save, 
                                               inclass=True,
                                               outdir=self.outdir)
            if diagnose:
                plot(self.tab1)

        # Routine for the second method of order identification
        elif method==2:
            self.tab2 = niriss.mask_method_two(self, save=save, 
                                               inclass=True,
                                               outdir=self.outdir)
            if diagnose:
                plot(self.tab2)
            
        return

    def box_extraction(self, boxsize1=70, boxsize2=60, perorder=False,
                       set=True):
        """
        Performs a really quick and dirty box mask.
        
        Parameters
        ----------
        boxsize1 : int, optional
           The size of the box for the first order.
           Default is 70 pixels.
        boxsize2 : int, optional
           The size of the box for the second order.
           Default is 60.
        perorder : bool, optional
           Creates an image filled with 1s and 2s for where the
           orders are. Default is False.

        Attributes
        ----------
        box_mask : np.ndarray
           The images for the box masks.
        box_sizes : tuple
           Attribute containing the box sizes used
           to create `box_order1` and `box_order2`.
           Tuple is (boxsize1, boxsize2).
        """
        if perorder==False:
            mask = np.ones(self.median.shape)
            fill1, fill2 = 0, 0
        else:
            mask = np.zeros(self.median.shape)
            fill1, fill2 = 1, 2

        if self.tab2 is not None:
            t = self.tab2
        elif self.tab1 is not None:
            t = self.tab1
        else:
            return('Need to run `identify_orders()`.')

        # Fills the mask with appropriate values for where the orders are
        # and the size of the boxes
        for i in range(self.median.shape[1]):
            s,e = int(t['order_1'][i]-boxsize1/2), int(t['order_1'][i]+boxsize1/2)
            mask[s:e,i] = fill1

            s,e = int(t['order_2'][i]-boxsize2/2), int(t['order_2'][i]+boxsize2/2)
            try:
                mask[s:e,i] += fill2
            except:
                pass

        if set:
            self.box_mask = mask
            self.box_sizes= (boxsize1, boxsize2)
        else:
            return box_mask
        return

    def fit_background(self, readnoise=18, sigclip=[4,4,4]):
        """
        Fits the sky background of the NIRISS images. This
        function calls `background.fitbg3`, which completes 
        an image fit to the sky background.

        Parameters
        ----------
        readnoise : int, optional
           Integer to estimate the read noise of the detector.
           Default is 18.
        sigclip : np.array, list, optional
           The sigma outliers by which to loop over to identify
           cosmic rays. Default is `[4,4,4]`. Can be of any 
           length the user wants.

        Attributes
        ----------
        data_bkg_subbed : np.ndarray
           Sky background subtracted images.
        bkg : np.ndarray
           Sky background fits for each image.
        bkg_var : np.ndarray
           Errors on the sky background fits.
        """
        self.box_extraction()
        
        bkg_outputs = background.fitbg3(self, 
                                        np.array(self.box_mask-1, dtype=bool), 
                                        readnoise=readnoise,
                                        sigclip=sigclip,
                                        inclass=True)

        self.data_bkg_subbed = bkg_outputs[0]
        self.bkg = bkg_outputs[1]
        self.bkg_var = bkg_outputs[2]

        return


    def optimal_extraction(self, proftype='gaussian', quad=1, pos1=None,
                           pos2=None):
        """
        Performs optimal extraction.
        """
        def sum_spectra(order, boxes):
            """ Box extracted spectra and variance. """
            test = np.zeros(boxes.shape)
            x,y = np.where((boxes==order) | (boxes==3))
            test[x,y]=1
            bs =  np.nansum(test*self.data,axis=1)
            bv =  np.sqrt(np.nansum((test*self.var)**2, axis=1))
            return bs, bv

        boxes = self.box_extraction(perorder=True, set=False)
        spectrum1, var1 = sum_spectra(1, boxes) # box extracted spectra order 1
        spectrum2, var2 = sum_spectra(2, boxes) # box extracted spectra order 2

        if self.bkg is None:
            print("Running background modeling with default settings.")
            self.fit_background()

        if quad == 1:
            s, e = niriss_extraction.optimal_extraction(self.data,
                                                        spectrum=spectrum1,
                                                        spectrum_var=var1,
                                                        sky_bkg=self.bkg,
                                                        pos1=pos1,
                                                        pos2=pos2,
                                                        proftype=proftype,
                                                        quad=quad)
        elif quad == 2:
            s, e = niriss_extraction.optimal_extraction(self.data,
                                                        spectrum=spectrum2,
                                                        spectrum_var=var2,
                                                        sky_bkg=self.bkg,
                                                        pos1=pos1,
                                                        pos2=pos2,
                                                        proftype=proftype,
                                                        quad=quad)
