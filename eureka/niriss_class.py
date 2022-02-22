import os
import numpy as np
import matplotlib.pyplot as plt

from .S3_data_reduction import niriss
from .S3_data_reduction.s3_reduce import MetaClass, DataClass

__all__ = ['NIRISS']

class NIRISS(object):

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

        self.tab1 = None
        self.tab2 = None

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
            
