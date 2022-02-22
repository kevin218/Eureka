import numpy as np
import matplotlib.pyplot as plt

from S3_data_reduction.s3_reduce import MetaClass, DataClass
from S3_data_reduction.niriss import *

__all__ = ['NIRISS']

class NIRISS(object):

    def __init__(self, filename, wavefile, f277_filename, path):

        data = DataClass()
        meta = MetaClass()

        data, meta = niriss.read(os.path.join(path, filename),
                                 os.path.join(path, f277_filename),
                                 data,
                                 meta)
        reassign_attrs(data)
        reassign_attrs(meta)

        # get wavelength solutions
        w1, w2, w3 = wave_niriss(os.path.join(path, wavefile))
        self.wavelength_order1 = w1 + 0.0
        self.wavelength_order2 = w2 + 0.0
        self.wavelength_order3 = w3 + 0.0


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
            setattr(self, val)
        return
        
        
