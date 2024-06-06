#! /usr/bin/env python
import unittest
import numpy as np
import os
import sys

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.S5_lightcurve_fitting import models, simulations
from eureka.lib.readEPF import Parameters, Parameter
from eureka.lib.readECF import MetaClass
from eureka.lib import logedit
from eureka.S5_lightcurve_fitting import s5_fit

meta = MetaClass()
meta.eventlabel = 'NIRCam'


class TestModels(unittest.TestCase):
    """Tests for the models.py module"""
    def setUp(self):
        """Setup for the tests"""
        # Set time to use for evaluations
        self.time = np.linspace(0, 1, 100)

    def test_model(self):
        """Tests for the generic Model class"""
        # Test model creation
        name = 'Model 1'
        self.model = models.Model(name=name)
        self.assertEqual(self.model.name, name)

        # Test model units
        self.assertEqual(str(self.model.time_units), 'BMJD_TDB')
        self.model.time_units = 'MJD'
        self.assertEqual(self.model.time_units, 'MJD')

    def test_compositemodel(self):
        """Tests for the CompositeModel class"""
        model1 = models.Model()
        model2 = models.Model()
        self.comp_model = model1*model2
        self.comp_model.name = 'composite'

    def test_polynomialmodel(self):
        """Tests for the PolynomialModel class"""
        # create dictionary
        params = {"c1": [0.0005, 'free'], "c0": [0.997, 'free'],
                  "name": 'linear'}

        # Create the model
        self.lin_model = models.PolynomialModel(parameters=None,
                                                coeff_dict=params, nchan=1)

        # Evaluate and test output
        self.lin_model.time = self.time
        vals = self.lin_model.eval()
        self.assertEqual(vals.size, self.time.size)

    def test_transitmodel(self):
        """Tests for the BatmanTransitModel class"""
        # Set the intial parameters
        params = Parameters()
        params.rp = 0.22, 'free', 0.0, 0.4, 'U'  # rprs
        params.per = 10.721490, 'fixed'
        params.t0 = 0.48, 'free', 0, 1, 'U'
        params.inc = 89.7, 'free', 80., 90., 'U'
        params.a = 18.2, 'free', 15., 20., 'U'    # ars
        params.ecc = 0., 'fixed'
        params.w = 90., 'fixed'             # omega
        params.limb_dark = '4-parameter', 'independent'
        params.u1 = 0.1, 'free', 0., 1., 'U'
        params.u2 = 0.1, 'free', 0., 1., 'U'
        params.u3 = 0.1, 'free', 0., 1., 'U'
        params.u4 = 0.1, 'free', 0., 1., 'U'

        # Make the transit model
        meta = MetaClass()
        meta.sharedp = False
        meta.multwhite = False
        meta.num_planets = 1
        meta.ld_from_S4 = False
        meta.ld_file = None
        longparamlist, paramtitles = s5_fit.make_longparamlist(meta, params, 1)
        freenames = []
        for key in params.dict:
            if params.dict[key][1] in ['free', 'shared', 'white_free',
                                       'white_fixed']:
                freenames.append(key)
        self.t_model = models.BatmanTransitModel(parameters=params,
                                                 name='transit', fmt='r--',
                                                 freenames=freenames,
                                                 longparamlist=longparamlist,
                                                 nchan=1,
                                                 paramtitles=paramtitles,
                                                 ld_from_S4=meta.ld_from_S4,
                                                 ld_from_file=meta.ld_file,
                                                 num_planets=meta.num_planets)

        # Evaluate and test output
        self.t_model.time = self.time
        vals = self.t_model.eval()
        self.assertEqual(vals.size, self.time.size)

    def test_eclipsemodel(self):
        """Tests for the BatmanEclipseModel class"""
        # Set the intial parameters
        params = Parameters()
        params.rp = 0.22, 'fixed'  # rprs
        params.fp = 0.08, 'free', 0.0, 0.1, 'U'  # fpfs
        params.per = 10.721490, 'fixed'
        params.t0 = 0.48, 'free', 0, 1, 'U'
        params.inc = 89.7, 'free', 80., 90., 'U'
        params.a = 18.2, 'free', 15., 20., 'U'  # ars
        params.ecc = 0., 'fixed'
        params.w = 90., 'fixed'  # omega
        params.Rs = 1., 'independent'

        # Make the eclipse model
        meta = MetaClass()
        meta.sharedp = False
        meta.multwhite = False
        meta.num_planets = 1
        longparamlist, paramtitles = s5_fit.make_longparamlist(meta, params, 1)
        freenames = []
        for key in params.dict:
            if params.dict[key][1] in ['free', 'shared', 'white_free',
                                       'white_fixed']:
                freenames.append(key)
        log = logedit.Logedit(f'.{os.sep}data{os.sep}test.log')
        self.e_model = models.BatmanEclipseModel(parameters=params,
                                                 name='transit', fmt='r--',
                                                 log=log,
                                                 freenames=freenames,
                                                 longparamlist=longparamlist,
                                                 nchan=1,
                                                 paramtitles=paramtitles,
                                                 num_planets=meta.num_planets)

        # Remove the temporary log file
        os.system(f"rm .{os.sep}data{os.sep}test.log")

        # Evaluate and test output
        self.e_model.time = self.time
        vals = self.e_model.eval()
        self.assertEqual(vals.size, self.time.size)

    def test_sinsoidalmodel(self):
        """Tests for the SinusoidPhaseCurve class"""
        # create dictionary
        params = Parameters()
        params.rp = 0.22, 'free', 0.0, 0.4, 'U'  # rprs
        params.fp = 0.08, 'free', 0.0, 0.1, 'U'  # fpfs
        params.per = 10.721490, 'fixed'
        params.t0 = 0.48, 'free', 0, 1, 'U'
        params.inc = 89.7, 'free', 80., 90., 'U'
        params.a = 18.2, 'free', 15., 20., 'U'    # ars
        params.ecc = 0., 'fixed'
        params.w = 90., 'fixed'             # omega
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
        meta = MetaClass()
        meta.sharedp = False
        meta.multwhite = False
        meta.num_planets = 1
        meta.ld_from_S4 = False
        meta.ld_file = None
        longparamlist, paramtitles = s5_fit.make_longparamlist(meta, params, 1)
        freenames = []
        for key in params.dict:
            if params.dict[key][1] in ['free', 'shared', 'white_free',
                                       'white_fixed']:
                freenames.append(key)
        log = logedit.Logedit(f'.{os.sep}data{os.sep}test.log')
        self.t_model = models.BatmanTransitModel(parameters=params,
                                                 name='transit', fmt='r--',
                                                 freenames=freenames,
                                                 longparamlist=longparamlist,
                                                 nchan=1,
                                                 paramtitles=paramtitles,
                                                 ld_from_S4=meta.ld_from_S4,
                                                 ld_from_file=meta.ld_file,
                                                 num_planets=meta.num_planets)
        self.e_model = models.BatmanEclipseModel(parameters=params,
                                                 name='eclipse', fmt='r--',
                                                 log=log,
                                                 freenames=freenames,
                                                 longparamlist=longparamlist,
                                                 nchan=1,
                                                 paramtitles=paramtitles,
                                                 num_planets=meta.num_planets)
        self.phasecurve = \
            models.SinusoidPhaseCurveModel(parameters=params,
                                           name='phasecurve', fmt='r--',
                                           longparamlist=longparamlist,
                                           freenames=freenames,
                                           nchan=1, paramtitles=paramtitles,
                                           transit_model=self.t_model,
                                           eclipse_model=self.e_model,
                                           num_planets=meta.num_planets)

        # Remove the temporary log file
        os.system(f"rm .{os.sep}data{os.sep}test.log")

        # Evaluate and test output
        self.phasecurve.time = self.time
        vals = self.phasecurve.eval()
        self.assertEqual(vals.size, self.time.size)

    def test_poettr_model(self):
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
        meta = MetaClass()
        meta.sharedp = False
        meta.multwhite = False
        meta.num_planets = 1
        meta.ld_from_S4 = False
        meta.ld_file = None
        longparamlist, paramtitles = s5_fit.make_longparamlist(meta, params, 1)
        freenames = []
        for key in params.dict:
            if params.dict[key][1] in ['free', 'shared', 'white_free',
                                       'white_fixed']:
                freenames.append(key)
        self.t_poet_tr = models.PoetTransitModel(parameters=params,
                                                 name='poet_tr', fmt='r--',
                                                 freenames=freenames,
                                                 longparamlist=longparamlist,
                                                 nchan=1,
                                                 paramtitles=paramtitles,
                                                 ld_from_S4=meta.ld_from_S4,
                                                 ld_from_file=meta.ld_file,
                                                 num_planets=meta.num_planets)

        # Evaluate and test output
        self.t_poet_tr.time = self.time
        vals = self.t_poet_tr.eval()
        self.assertEqual(vals.size, self.time.size)

    def test_poetecl_model(self):
        """Tests for the POETModel class"""
        # Set the intial parameters
        params = Parameters()
        params.rprs = 0.22, 'fixed'  # rprs
        params.fpfs = 0.08, 'free', 0.0, 0.1, 'U'  # fprs
        params.per = 10.721490, 'fixed'
        params.t0 = 0.48, 'free', 0, 1, 'U'
        params.inc = 89.7, 'free', 80., 90., 'U'
        params.ars = 18.2, 'free', 15., 20., 'U'  # aprs
        params.ecc = 0., 'fixed'
        params.w = 90., 'fixed'  # omega
        params.Rs = 1., 'independent'

        # Make the eclipse model
        meta = MetaClass()
        meta.sharedp = False
        meta.multwhite = False
        meta.num_planets = 1
        longparamlist, paramtitles = s5_fit.make_longparamlist(meta, params, 1)
        freenames = []
        for key in params.dict:
            if params.dict[key][1] in ['free', 'shared', 'white_free',
                                       'white_fixed']:
                freenames.append(key)
        log = logedit.Logedit(f'.{os.sep}data{os.sep}test.log')
        self.t_poet_ecl = models.PoetEclipseModel(parameters=params,
                                                  name='eclipse', fmt='r--',
                                                  log=log,
                                                  freenames=freenames,
                                                  longparamlist=longparamlist,
                                                  nchan=1,
                                                  paramtitles=paramtitles,
                                                  num_planets=meta.num_planets)

        # Remove the temporary log file
        os.system(f"rm .{os.sep}data{os.sep}test.log")

        # Evaluate and test output
        self.t_poet_ecl.time = self.time
        vals = self.t_poet_ecl.eval()
        self.assertEqual(vals.size, self.time.size)

    def test_poetpc_model(self):
        """Tests for the PoetPC class"""
        # create dictionary
        params = Parameters()
        params.rp = 0.1, 'free', 0.0, 0.4, 'U'  # rprs
        params.fp = 0.01, 'free', 0.0, 0.1, 'U'  # fpfs
        params.per = 1., 'fixed'
        params.t0 = 0.0, 'free', 0, 1, 'U'
        params.inc = 89.7, 'free', 80., 90., 'U'
        params.a = 8.2, 'free', 15., 20., 'U'    # ars
        params.ecc = 0., 'fixed'
        params.w = 90., 'fixed'             # omega
        params.limb_dark = 'quadratic', 'independent'
        params.u1 = 0.1, 'free', 0., 1., 'U'
        params.u2 = 0.1, 'free', 0., 1., 'U'
        params.cos1_amp = 0.9, 'free', 0, 2.0, 'U'
        params.cos1_off = 10, 'free', -30, 30, 'U'
        params.cos2_amp = 0.1, 'fixed', -1, 1, 'U'
        params.cos2_off = 45, 'fixed', -1, 1, 'U'
        params.Rs = 1., 'independent'

        # Create the model
        meta = MetaClass()
        meta.sharedp = False
        meta.multwhite = False
        meta.num_planets = 1
        meta.ld_from_S4 = False
        meta.ld_file = None
        longparamlist, paramtitles = s5_fit.make_longparamlist(meta, params, 1)
        freenames = []
        for key in params.dict:
            if params.dict[key][1] in ['free', 'shared', 'white_free',
                                       'white_fixed']:
                freenames.append(key)
        log = logedit.Logedit(f'.{os.sep}data{os.sep}test.log')
        self.t_model = models.PoetTransitModel(parameters=params,
                                               name='transit', fmt='r--',
                                               freenames=freenames,
                                               longparamlist=longparamlist,
                                               nchan=1,
                                               paramtitles=paramtitles,
                                               ld_from_S4=meta.ld_from_S4,
                                               ld_from_file=meta.ld_file,
                                               num_planets=meta.num_planets)
        self.e_model = models.PoetEclipseModel(parameters=params,
                                               name='eclipse', fmt='r--',
                                               log=log,
                                               freenames=freenames,
                                               longparamlist=longparamlist,
                                               nchan=1,
                                               paramtitles=paramtitles,
                                               num_planets=meta.num_planets)
        self.phasecurve = \
            models.PoetPCModel(parameters=params,
                               name='phasecurve', fmt='r--',
                               longparamlist=longparamlist,
                               freenames=freenames,
                               nchan=1, paramtitles=paramtitles,
                               transit_model=self.t_model,
                               eclipse_model=self.e_model,
                               num_planets=meta.num_planets)

        # Remove the temporary log file
        os.system(f"rm .{os.sep}data{os.sep}test.log")

        # Evaluate and test output
        self.phasecurve.time = self.time
        vals = self.phasecurve.eval()
        self.assertEqual(vals.size, self.time.size)

    def test_lorentzian_model(self):
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
        meta = MetaClass()
        meta.sharedp = False
        meta.multwhite = False
        longparamlist, paramtitles = s5_fit.make_longparamlist(meta, params, 1)
        freenames = []
        for key in params.dict:
            if params.dict[key][1] in ['free', 'shared', 'white_free',
                                       'white_fixed']:
                freenames.append(key)
        log = logedit.Logedit(f'.{os.sep}data{os.sep}test.log')
        self.t_lorentzian = models.LorentzianModel(parameters=params,
                                                   name='transit', fmt='r--',
                                                   log=log,
                                                   freenames=freenames,
                                                   longparamlist=longparamlist,
                                                   nchan=1,
                                                   paramtitles=paramtitles)

        # Remove the temporary log file
        os.system(f"rm .{os.sep}data{os.sep}test.log")

        # Evaluate and test output
        self.t_lorentzian.time = self.time
        vals = self.t_lorentzian.eval()
        self.assertEqual(vals.size, self.time.size)

    def test_exponentialmodel(self):
        """Tests for the ExponentialModel class"""
        # Create the model
        freenames = ['r0', 'r1', 'r2']
        self.exp_model = models.ExpRampModel(coeff_dict={'r0': [1., 'free'],
                                                         'r1': [0.05, 'free'],
                                                         'r2': [0.01, 'free']},
                                             freenames=freenames,
                                             nchan=1)

        # Evaluate and test output
        self.exp_model.time = self.time
        vals = self.exp_model.eval()
        self.assertEqual(vals.size, self.time.size)

        # Create the model
        freenames = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5']
        self.exp_model = models.ExpRampModel(coeff_dict={'r0': [1., 'free'],
                                                         'r1': [0.05, 'free'],
                                                         'r2': [0.01, 'free'],
                                                         'r3': [1., 'free'],
                                                         'r4': [0.05, 'free'],
                                                         'r5': [0.01, 'free']},
                                             freenames=freenames,
                                             nchan=1)

        # Evaluate and test output
        self.exp_model.time = self.time
        vals = self.exp_model.eval()
        self.assertEqual(vals.size, self.time.size)


class TestParameters(unittest.TestCase):
    """Tests for the parameters.py module"""
    def setUp(self):
        """Setup for the tests"""
        pass

    def test_parameter(self):
        """Test that a Parameter object can be created"""
        # Create the parameter
        pname = 'p1'
        pval = 12.34
        ptype = 'free'
        priorpar1 = 10
        priorpar2 = 15
        prior = 'U'
        self.param = Parameter(pname, pval, ptype, priorpar1, priorpar2, prior)

        # Test bogus input
        self.assertRaises(TypeError, Parameter, 123)
        self.assertRaises(ValueError, Parameter, 'foo', 123, 123)

        # Test the attributes
        self.assertEqual(self.param.name, pname)
        self.assertEqual(self.param.value, pval)
        self.assertEqual(self.param.ptype, ptype)
        self.assertEqual(self.param.priorpar1, priorpar1)
        self.assertEqual(self.param.priorpar2, priorpar2)
        self.assertEqual(self.param.prior, prior)
        self.assertEqual(self.param.values, [pname, pval, ptype, priorpar1,
                                             priorpar2, prior])

    def test_parameters(self):
        """Test that a Parameters object can be created"""
        self.params = Parameters()
        self.params.param1 = 123.456, 'free'
        self.params.param2 = 234.567, 'free', 200, 300

        # Test the auto attribute assignment
        self.assertEqual(self.params.param1.values,
                         ['param1', 123.456, 'free'])
        self.assertEqual(self.params.param2.values,
                         ['param2', 234.567, 'free', 200, 300])
        # Test FileNotFoundError
        self.assertRaises(FileNotFoundError, Parameters, None,
                          'non-existent.epf')


class TestSimulations(unittest.TestCase):
    """Test the simulations.py module"""
    def setUp(self):
        """Setup for the tests"""
        pass

    def test_simulation(self):
        """Test the simulations can be made properly"""
        # Test to pass
        npts = 1234
        time, _, _, _ = simulations.simulate_lightcurve('WASP-107b', 0.1,
                                                        npts=npts, plot=False)
        self.assertEqual(len(time), npts)

        # Test to fail
        self.assertRaises(KeyError, simulations.simulate_lightcurve,
                          'foobar', 0.1)
