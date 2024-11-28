#! /usr/bin/env python
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.S5_lightcurve_fitting import models, simulations
from eureka.lib.readEPF import Parameters, Parameter
from eureka.S5_lightcurve_fitting.s5_meta import S5MetaClass
from eureka.lib import logedit
from eureka.S5_lightcurve_fitting import s5_fit


class testingMetaClass(S5MetaClass):
    """Apply testing-relevant defaults to S5MetaClass objects.

    Parameters
    ----------
    meta : eureka.S5_lightcurve_fitting.s5_meta.S5MetaClass
        The meta object to prepare.
    """
    def __init__(self, folder=None, file=None, eventlabel=None, **kwargs):
        super().__init__(folder, file, eventlabel, **kwargs)
        self.spec_hw = 1
        self.bg_hw = 2
        self.fit_par = ''
        self.fit_method = 'lsq'
        self.run_myfuncs = []
        self.topdir = './'
        self.inputdir = ''
        self.outputdir = ''
        self.sharedp = False
        self.whitep = False
        self.set_defaults()

# Tests for the parameters.py module


def test_parameter(capsys):
    """Test that a Parameter object can be created"""
    # Create the parameter
    pname = 'p1'
    pval = 12.34
    ptype = 'free'
    priorpar1 = 10
    priorpar2 = 15
    prior = 'U'
    param = Parameter(pname, pval, ptype, priorpar1, priorpar2, prior)

    # Test bogus input
    with pytest.raises(TypeError):
        Parameter(123)
    with pytest.raises(ValueError):
        Parameter('foo', 123, 123)

    # Test the attributes
    assert param.name == pname
    assert param.value == pval
    assert param.ptype == ptype
    assert param.priorpar1 == priorpar1
    assert param.priorpar2 == priorpar2
    assert param.prior == prior
    assert param.values == [pname, pval, ptype, priorpar1, priorpar2, prior]


def test_parameters(capsys):
    """Test that a Parameters object can be created"""
    params = Parameters()
    params.param1 = 123.456, 'free'
    params.param2 = 234.567, 'free', 200, 300

    # Test the auto attribute assignment
    assert params.param1.values == ['param1', 123.456, 'free']
    assert params.param2.values == ['param2', 234.567, 'free', 200, 300]
    # Test FileNotFoundError
    with pytest.raises(FileNotFoundError):
        Parameters(None, 'non-existent.epf')

# Tests for the models.py module


def test_model(capsys):
    """Tests for the generic Model class"""
    # Test model creation
    name = 'Model 1'
    model = models.Model(name=name)
    assert model.name == name

    # Test model units
    assert str(model.time_units) == 'BMJD_TDB'
    model.time_units = 'MJD'
    assert model.time_units == 'MJD'


def test_compositemodel(capsys):
    """Tests for the CompositeModel class"""
    model1 = models.Model()
    model2 = models.Model()
    comp_model = model1*model2
    comp_model.name = 'composite'


def test_polynomialmodel(capsys):
    """Tests for the PolynomialModel class"""
    # create dictionary
    params = {"c1": [0.0005, 'free'], "c0": [0.997, 'free'], "name": 'linear'}

    # Create the model
    lin_model = models.PolynomialModel(parameters=None, coeff_dict=params,
                                       nchan=1)

    # Evaluate and test output
    time = np.linspace(0, 1, 100)
    lin_model.time = time
    vals = lin_model.eval()
    assert vals.size == time.size


def test_transitmodel(capsys):
    """Tests for the BatmanTransitModel class"""
    # Set the intial parameters
    params = Parameters()
    params.rp = 0.22, 'free', 0.0, 0.4, 'U'
    params.per = 10.721490, 'fixed'
    params.t0 = 0.48, 'free', 0, 1, 'U'
    params.inc = 89.7, 'free', 80., 90., 'U'
    params.a = 18.2, 'free', 15., 20., 'U'
    params.ecc = 0., 'fixed'
    params.w = 90., 'fixed'
    params.limb_dark = '4-parameter', 'independent'
    params.u1 = 0.1, 'free', 0., 1., 'U'
    params.u2 = 0.1, 'free', 0., 1., 'U'
    params.u3 = 0.1, 'free', 0., 1., 'U'
    params.u4 = 0.1, 'free', 0., 1., 'U'

    # Make the transit model
    meta = testingMetaClass()
    longparamlist, paramtitles, freenames, params = \
        s5_fit.make_longparamlist(meta, params, 1)
    t_model = models.BatmanTransitModel(parameters=params,
                                        name='transit', fmt='r--',
                                        freenames=freenames,
                                        longparamlist=longparamlist,
                                        nchan=1,
                                        paramtitles=paramtitles,
                                        ld_from_S4=meta.use_generate_ld,
                                        ld_from_file=meta.ld_file,
                                        num_planets=meta.num_planets)

    # Evaluate and test output
    time = np.linspace(0, 1, 100)
    t_model.time = time
    vals = t_model.eval()
    assert vals.size == time.size


def test_eclipsemodel(capsys):
    """Tests for the BatmanEclipseModel class"""
    # Set the intial parameters
    params = Parameters()
    params.rp = 0.22, 'fixed'
    params.fp = 0.08, 'free', 0.0, 0.1, 'U'
    params.per = 10.721490, 'fixed'
    params.t0 = 0.48, 'free', 0, 1, 'U'
    params.inc = 89.7, 'free', 80., 90., 'U'
    params.a = 18.2, 'free', 15., 20., 'U'
    params.ecc = 0., 'fixed'
    params.w = 90., 'fixed'
    params.Rs = 1., 'independent'

    # Make the eclipse model
    meta = testingMetaClass()
    longparamlist, paramtitles, freenames, params = \
        s5_fit.make_longparamlist(meta, params, 1)
    log = logedit.Logedit(f'.{os.sep}data{os.sep}test.log')
    e_model = models.BatmanEclipseModel(parameters=params,
                                        name='transit', fmt='r--',
                                        log=log, freenames=freenames,
                                        longparamlist=longparamlist,
                                        nchan=1,
                                        paramtitles=paramtitles,
                                        num_planets=meta.num_planets)

    # Remove the temporary log file
    os.system(f"rm .{os.sep}data{os.sep}test.log")

    # Evaluate and test output
    time = np.linspace(0, 1, 100)
    e_model.time = time
    vals = e_model.eval()
    assert vals.size == time.size


def test_sinsoidalmodel(capsys):
    """Tests for the SinusoidPhaseCurve class"""
    # create dictionary
    params = Parameters()
    params.rp = 0.22, 'free', 0.0, 0.4, 'U'
    params.fp = 0.08, 'free', 0.0, 0.1, 'U'
    params.per = 10.721490, 'fixed'
    params.t0 = 0.48, 'free', 0, 1, 'U'
    params.inc = 89.7, 'free', 80., 90., 'U'
    params.a = 18.2, 'free', 15., 20., 'U'
    params.ecc = 0., 'fixed'
    params.w = 90., 'fixed'
    params.limb_dark = '4-parameter', 'independent'
    params.u1 = 0.1, 'free', 0., 1., 'U'
    params.u2 = 0.1, 'free', 0., 1., 'U'
    params.u3 = 0.1, 'free', 0., 1., 'U'
    params.u4 = 0.1, 'free', 0., 1., 'U'
    params.AmpSin1 = 0.1, 'free', -0.5, 0.5, 'U'
    params.AmpCos1 = 0.3, 'free', 0.0, 0.5, 'U'
    params.AmpSin2 = 0.01, 'free', -1, 1, 'U'
    params.AmpCos2 = 0.01, 'free', -1, 1, 'U'
    params.Rs = 1., 'independent'

    # Create the model
    meta = testingMetaClass()
    longparamlist, paramtitles, freenames, params = \
        s5_fit.make_longparamlist(meta, params, 1)
    log = logedit.Logedit(f'.{os.sep}data{os.sep}test.log')
    t_model = models.BatmanTransitModel(parameters=params,
                                        name='transit', fmt='r--',
                                        freenames=freenames,
                                        longparamlist=longparamlist,
                                        nchan=1, paramtitles=paramtitles,
                                        ld_from_S4=meta.use_generate_ld,
                                        ld_from_file=meta.ld_file,
                                        num_planets=meta.num_planets)
    e_model = models.BatmanEclipseModel(parameters=params,
                                        name='eclipse', fmt='r--',
                                        log=log, freenames=freenames,
                                        longparamlist=longparamlist,
                                        nchan=1, paramtitles=paramtitles,
                                        num_planets=meta.num_planets)
    phasecurve = \
        models.SinusoidPhaseCurveModel(parameters=params,
                                       name='phasecurve', fmt='r--',
                                       longparamlist=longparamlist,
                                       freenames=freenames,
                                       nchan=1, paramtitles=paramtitles,
                                       transit_model=t_model,
                                       eclipse_model=e_model,
                                       num_planets=meta.num_planets)

    # Remove the temporary log file
    os.system(f"rm .{os.sep}data{os.sep}test.log")

    # Evaluate and test output
    time = np.linspace(0, 1, 100)
    phasecurve.time = time
    vals = phasecurve.eval()
    assert vals.size == time.size


def test_poettr_model(capsys):
    """Tests for the POETModel class"""
    # Set the intial parameters
    params = Parameters()
    params.rp = 0.22, 'free', 0.0, 0.4, 'U'
    params.per = 10.721490, 'fixed'
    params.t0 = 0.48, 'free', 0, 1, 'U'
    params.inc = 89.7, 'free', 80., 90., 'U'
    params.a = 18.2, 'free', 15., 20., 'U'
    params.ecc = 0., 'fixed'
    params.w = 90., 'fixed'
    params.rp1 = 0.12, 'free', 0.0, 0.4, 'U'
    params.per1 = 5.721490, 'fixed'
    params.t01 = 0.28, 'free', 0, 1, 'U'
    params.inc1 = 89.5, 'free', 80., 90., 'U'
    params.a1 = 8.2, 'free', 15., 20., 'U'
    params.ecc1 = 0.1, 'fixed'
    params.w1 = 90., 'fixed'
    params.limb_dark = 'linear', 'independent'
    params.u1 = 0.2, 'free', 0., 1., 'U'

    # Make the transit model
    meta = testingMetaClass()
    longparamlist, paramtitles, freenames, params = \
        s5_fit.make_longparamlist(meta, params, 1)
    t_poet_tr = models.PoetTransitModel(parameters=params,
                                        name='poet_tr', fmt='r--',
                                        freenames=freenames,
                                        longparamlist=longparamlist,
                                        nchan=1, paramtitles=paramtitles,
                                        ld_from_S4=meta.use_generate_ld,
                                        ld_from_file=meta.ld_file,
                                        num_planets=meta.num_planets)

    # Evaluate and test output
    time = np.linspace(0, 1, 100)
    t_poet_tr.time = time
    vals = t_poet_tr.eval()
    assert vals.size == time.size


def test_poetecl_model(capsys):
    """Tests for the POETModel class"""
    # Set the intial parameters
    params = Parameters()
    params.rprs = 0.22, 'fixed'
    params.fpfs = 0.08, 'free', 0.0, 0.1, 'U'
    params.per = 10.721490, 'fixed'
    params.t0 = 0.48, 'free', 0, 1, 'U'
    params.inc = 89.7, 'free', 80., 90., 'U'
    params.ars = 18.2, 'free', 15., 20., 'U'
    params.ecc = 0., 'fixed'
    params.w = 90., 'fixed'
    params.Rs = 1., 'independent'

    # Make the eclipse model
    meta = testingMetaClass()
    longparamlist, paramtitles, freenames, params = \
        s5_fit.make_longparamlist(meta, params, 1)
    log = logedit.Logedit(f'.{os.sep}data{os.sep}test.log')
    t_poet_ecl = models.PoetEclipseModel(parameters=params,
                                         name='eclipse', fmt='r--',
                                         log=log, freenames=freenames,
                                         longparamlist=longparamlist,
                                         nchan=1, paramtitles=paramtitles,
                                         num_planets=meta.num_planets)

    # Remove the temporary log file
    os.system(f"rm .{os.sep}data{os.sep}test.log")

    # Evaluate and test output
    time = np.linspace(0, 1, 100)
    t_poet_ecl.time = time
    vals = t_poet_ecl.eval()
    assert vals.size == time.size


def test_poetpc_model(capsys):
    """Tests for the PoetPC class"""
    # create dictionary
    params = Parameters()
    params.rp = 0.1, 'free', 0.0, 0.4, 'U'
    params.fp = 0.01, 'free', 0.0, 0.1, 'U'
    params.per = 1., 'fixed'
    params.t0 = 0.0, 'free', 0, 1, 'U'
    params.inc = 89.7, 'free', 80., 90., 'U'
    params.a = 8.2, 'free', 15., 20., 'U'
    params.ecc = 0., 'fixed'
    params.w = 90., 'fixed'
    params.limb_dark = 'quadratic', 'independent'
    params.u1 = 0.1, 'free', 0., 1., 'U'
    params.u2 = 0.1, 'free', 0., 1., 'U'
    params.cos1_amp = 0.9, 'free', 0, 2.0, 'U'
    params.cos1_off = 10, 'free', -30, 30, 'U'
    params.cos2_amp = 0.1, 'fixed', -1, 1, 'U'
    params.cos2_off = 45, 'fixed', -1, 1, 'U'
    params.Rs = 1., 'independent'

    # Create the model
    meta = testingMetaClass()
    longparamlist, paramtitles, freenames, params = \
        s5_fit.make_longparamlist(meta, params, 1)
    log = logedit.Logedit(f'.{os.sep}data{os.sep}test.log')
    t_model = models.PoetTransitModel(parameters=params,
                                      name='transit', fmt='r--',
                                      freenames=freenames,
                                      longparamlist=longparamlist,
                                      nchan=1, paramtitles=paramtitles,
                                      ld_from_S4=meta.use_generate_ld,
                                      ld_from_file=meta.ld_file,
                                      num_planets=meta.num_planets)
    e_model = models.PoetEclipseModel(parameters=params,
                                      name='eclipse', fmt='r--',
                                      log=log, freenames=freenames,
                                      longparamlist=longparamlist,
                                      nchan=1, paramtitles=paramtitles,
                                      num_planets=meta.num_planets)
    phasecurve = \
        models.PoetPCModel(parameters=params,
                           name='phasecurve', fmt='r--',
                           longparamlist=longparamlist,
                           freenames=freenames,
                           nchan=1, paramtitles=paramtitles,
                           transit_model=t_model,
                           eclipse_model=e_model,
                           num_planets=meta.num_planets)

    # Remove the temporary log file
    os.system(f"rm .{os.sep}data{os.sep}test.log")

    # Evaluate and test output
    time = np.linspace(0, 1, 100)
    phasecurve.time = time
    vals = phasecurve.eval()
    assert vals.size == time.size


def test_lorentzian_model(capsys):
    """Tests for the LorentzianModel class"""
    # Set the intial parameters
    params = Parameters()
    params.lor_amp_lhs = 0.03, 'free', 0.0, 0.1, 'U'
    params.lor_amp_rhs = 0.03, 'free', 0.0, 0.1, 'U'
    params.lor_hwhm_lhs = 1e-5, 'free', 0, 0.1, 'U'
    params.lor_hwhm_rhs = 1e-5, 'free', 0, 0.1, 'U'
    params.lor_t0 = 0.0, 'fixed'
    params.lor_power = 2., 'fixed'

    # Make the eclipse model
    meta = testingMetaClass()
    longparamlist, paramtitles, freenames, params = \
        s5_fit.make_longparamlist(meta, params, 1)
    log = logedit.Logedit(f'.{os.sep}data{os.sep}test.log')
    t_lorentzian = models.LorentzianModel(parameters=params,
                                          name='transit', fmt='r--',
                                          log=log, freenames=freenames,
                                          longparamlist=longparamlist,
                                          nchan=1, paramtitles=paramtitles)

    # Remove the temporary log file
    os.system(f"rm .{os.sep}data{os.sep}test.log")

    # Evaluate and test output
    time = np.linspace(0, 1, 100)
    t_lorentzian.time = time
    vals = t_lorentzian.eval()
    assert vals.size == time.size


def test_exponentialmodel(capsys):
    """Tests for the ExponentialModel class"""
    # Create the model
    freenames = ['r0', 'r1']
    exp_model = models.ExpRampModel(coeff_dict={'r0': [1., 'free'],
                                                'r1': [0.05, 'free']},
                                    freenames=freenames, nchan=1)

    # Evaluate and test output
    time = np.linspace(0, 1, 100)
    exp_model.time = time
    vals = exp_model.eval()
    assert vals.size == time.size

    # Create the model
    freenames = ['r0', 'r1', 'r2', 'r3']
    exp_model = models.ExpRampModel(coeff_dict={'r0': [1., 'free'],
                                                'r1': [0.05, 'free'],
                                                'r2': [1., 'free'],
                                                'r3': [0.05, 'free']},
                                    freenames=freenames, nchan=1)

    # Evaluate and test output
    time = np.linspace(0, 1, 100)
    exp_model.time = time
    vals = exp_model.eval()
    assert vals.size == time.size

# Test for the simulations.py module


def test_simulation(capsys):
    """Test the simulations can be made properly"""
    # Test to pass
    npts = 1234
    time, _, _, _ = simulations.simulate_lightcurve('WASP-107b', 0.1,
                                                    npts=npts, plot=False)
    assert len(time) == npts

    # Test to fail
    with pytest.raises(KeyError):
        simulations.simulate_lightcurve('foobar', 0.1)
