import numpy as np

from eureka.S5_lightcurve_fitting import models
from eureka.lib.readEPF import Parameters


def _params(**values):
    params = Parameters()
    for name, value in values.items():
        setattr(params, name, (value, 'free'))
    return params


def _transit_params(ecc=0.0, w=90.0):
    params = _params(limb_dark='quadratic', u1=0.1, u2=0.2, rp=0.08,
                     per=3.0, t0=0.0, inc=88.5, a=12.0, ecc=ecc, w=w,
                     spotcon=1.0, spotnpts=3000)
    params.limb_dark.ptype = 'fixed'
    return params


def _transit_model(model_class, params, time):
    model = model_class(
        parameters=params, paramtitles=list(params.dict.keys()),
        num_planets=1, ld_from_S4=False, ld_from_file=False
    )
    model.time = time
    return model.eval()


class _ConstantModel(models.Model):
    def __init__(self, values, modeltype='systematic', name='constant',
                 **kwargs):
        super().__init__(name=name, modeltype=modeltype, **kwargs)
        self.values = np.ma.array(values)

    def eval(self, channel=None, **kwargs):
        return self.values


class _FakeGPModel(models.Model):
    def __init__(self, values, **kwargs):
        super().__init__(name='fake GP', modeltype='GP', **kwargs)
        self.values = np.ma.array(values)
        self.seen_fit = None

    def eval(self, fit, channel=None, **kwargs):
        self.seen_fit = np.ma.copy(fit)
        return self.values
