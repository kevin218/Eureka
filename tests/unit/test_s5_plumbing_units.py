import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table
from astropy.table import Table as AstropyTable

os.environ.setdefault('MPLCONFIGDIR', '/tmp')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..', 'src')))
from eureka.S5_lightcurve_fitting import fitters, likelihood, s5_fit, utils
from eureka.S5_lightcurve_fitting.lightcurve import LightCurve
from eureka.S5_lightcurve_fitting import models
from eureka.lib.readEPF import Parameters


class _Log:
    def __init__(self):
        self.messages = []

    def writelog(self, message, **kwargs):
        self.messages.append(message)


class _FlatModel(models.Model):
    def __init__(self, parameters=None, freenames=None, values=None):
        super().__init__(parameters=parameters, freenames=freenames)
        self.values = np.ma.array([1.0, 1.0] if values is None else values)
        self.modeltype = 'systematic'

    def eval(self, channel=None, **kwargs):
        return self.values


class _LikelihoodGPModel(models.Model):
    def __init__(self, loglike, **kwargs):
        super().__init__(name='GP', modeltype='GP', **kwargs)
        self.loglike = loglike
        self.received_model_lc = None

    def eval(self, fit, channel=None, **kwargs):
        return np.zeros_like(fit)

    def loglikelihood(self, model_lc):
        self.received_model_lc = np.ma.copy(model_lc)
        return self.loglike


def _params(**values):
    params = Parameters()
    for name, spec in values.items():
        setattr(params, name, spec)
    return params


def test_update_uncertainty_applies_scatter_mult_per_channel():
    """Scatter multipliers should inflate each fitted channel locally."""
    theta = np.array([2.0, 3.0])
    unc = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    freenames = ['scatter_mult', 'scatter_mult_ch1']

    result = likelihood.update_uncertainty(theta, [2, 3], unc, freenames, 2)

    np.testing.assert_allclose(result, [0.2, 0.4, 0.9, 1.2, 1.5])
    np.testing.assert_allclose(unc, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_update_uncertainty_applies_scatter_ppm_per_channel():
    """Scatter-ppm parameters replace uncertainty per fitted channel."""
    theta = np.array([100.0, 250.0])
    unc = np.ones(5)
    freenames = ['scatter_ppm', 'scatter_ppm_ch1']

    result = likelihood.update_uncertainty(theta, [2, 3], unc, freenames, 2)

    np.testing.assert_allclose(result, [100e-6, 100e-6,
                                        250e-6, 250e-6, 250e-6])


def test_lnprior_handles_uniform_log_uniform_normal_and_scatter_bounds():
    """Priors should combine types and enforce positive scatter parameters."""
    theta = np.array([0.5, 10.0, 1.5, 2.0])
    prior1 = np.array([0.0, np.log(1.0), 1.0, 0.0])
    prior2 = np.array([1.0, np.log(100.0), 0.5, 10.0])
    priortype = np.array(['U', 'LU', 'N', 'U'])
    freenames = ['a', 'b', 'c', 'scatter_mult']

    result = likelihood.lnprior(theta, prior1, prior2, priortype, freenames)

    expected_normal = -0.5*(((1.5-1.0)/0.5)**2 + np.log(2*np.pi*0.5**2))
    assert result == pytest.approx(expected_normal)
    assert likelihood.lnprior(np.array([1.5, 10, 1.5, 2.0]), prior1, prior2,
                              priortype, freenames) == -np.inf
    assert likelihood.lnprior(np.array([0.5, 0.5, 1.5, 2.0]), prior1, prior2,
                              priortype, freenames) == -np.inf
    assert likelihood.lnprior(np.array([0.5, 10, 1.5, 0.0]), prior1, prior2,
                              priortype, freenames) == -np.inf


def test_lnprior_rejects_unknown_prior_type_and_ptform_transforms():
    """Unknown prior types should fail while valid dynesty transforms agree."""
    with pytest.raises(ValueError, match='PriorType'):
        likelihood.lnprior(np.array([1.0]), np.array([0.0]), np.array([1.0]),
                           np.array(['bad']), ['x'])

    result = likelihood.ptform(np.array([0.25, 0.5, 0.5]),
                               np.array([2.0, 1.0, 10.0]),
                               np.array([6.0, 100.0, 2.0]),
                               np.array(['U', 'LU', 'N']))
    np.testing.assert_allclose(result[:2], [3.0, 10.0])
    assert result[2] == pytest.approx(10.0)


def test_ln_like_and_lnprob_match_manual_gaussian_result():
    """Likelihood should match a hand-computed Gaussian loglike."""
    params = _params(c0=(1.0, 'free', 0.0, 2.0, 'U'))
    component = _FlatModel(parameters=params, freenames=['c0'],
                           values=np.array([1.0, 1.0]))
    model = models.CompositeModel([component], parameters=params)
    model.freenames = ['c0']
    model.time = np.array([0.0, 1.0])
    lc = SimpleNamespace(flux=np.array([1.1, 0.9]), unc=np.array([0.1, 0.2]),
                         nints=[2], nchannel_fitted=1)

    result = likelihood.ln_like(np.array([1.0]), lc, model, ['c0'])

    residuals = lc.flux - np.array([1.0, 1.0])
    expected = -0.5*np.sum((residuals/lc.unc_fit)**2 +
                           np.log(2*np.pi*lc.unc_fit**2))
    assert result == pytest.approx(expected)
    assert likelihood.lnprob(np.array([1.0]), lc, model, np.array([0.0]),
                             np.array([2.0]), np.array(['U']),
                             ['c0']) == pytest.approx(expected)
    assert likelihood.lnprob(np.array([3.0]), lc, model, np.array([0.0]),
                             np.array([2.0]), np.array(['U']),
                             ['c0']) == -np.inf


def test_ln_like_gp_branch_sums_component_loglikelihoods():
    """GP likelihoods should be delegated to and summed over GP components."""
    params = _params(c0=(1.0, 'free', 0.0, 2.0, 'U'))
    systematic = _FlatModel(parameters=params, freenames=['c0'],
                            values=np.array([1.0, 1.0]))
    gp1 = _LikelihoodGPModel(3.0)
    gp2 = _LikelihoodGPModel(-0.5)
    model = models.CompositeModel([systematic, gp1, gp2], parameters=params)
    model.freenames = ['c0']
    model.time = np.array([0.0, 1.0])
    lc = SimpleNamespace(flux=np.array([1.1, 0.9]), unc=np.array([0.1, 0.2]),
                         nints=[2], nchannel_fitted=1)

    result = likelihood.ln_like(np.array([1.0]), lc, model, ['c0'])

    assert result == pytest.approx(2.5)
    np.testing.assert_allclose(gp1.received_model_lc, [1.0, 1.0])
    np.testing.assert_allclose(gp2.received_model_lc, [1.0, 1.0])


def test_compute_reduced_chi_squared_uses_unmasked_points_and_logs():
    """Reduced chi-squared should count only unmasked flux samples."""
    lc = SimpleNamespace(flux=np.ma.array([1.1, 0.9, 2.0],
                                          mask=[False, False, True]),
                         unc_fit=np.array([0.1, 0.2, 0.3]))
    model = SimpleNamespace(eval=lambda incl_GP=False:
                            np.ma.array([1.0, 1.0, 1.0]))
    log = _Log()
    meta = SimpleNamespace(verbose=True)

    result = likelihood.computeRedChiSq(lc, log, model, meta, ['c0'])

    expected = ((0.1/0.1)**2 + (-0.1/0.2)**2)/(2-1)
    assert result == pytest.approx(expected)
    assert 'Reduced Chi-squared' in log.messages[0]


def test_compute_rms_returns_expected_bin_statistics():
    data = np.array([1.0, -1.0, 1.0, -1.0])

    rms, stderr, binsz, rmserr = likelihood.computeRMS(
        data, maxnbins=2, isrmserr=True
    )

    np.testing.assert_array_equal(binsz, [1, 2])
    np.testing.assert_allclose(rms, [1.0, 0.0])
    assert stderr[0] == pytest.approx(np.std(data)/np.sqrt(1)*np.sqrt(4/3))
    np.testing.assert_allclose(rmserr, [1/np.sqrt(8), 0.0])


def test_make_longparamlist_clones_channel_params_and_sorts_freenames():
    """Parameter expansion should clone channel-local fits in order."""
    params = _params(c0=(1.0, 'free'), per=(3.0, 'shared'),
                     inc=(89.0, 'fixed'), limb_dark=('uniform',
                                                     'independent'))
    meta = SimpleNamespace(multwhite=False, sharedp=True, wl_groups=[0, 0])

    longparamlist, paramtitles, freenames, updated = (
        s5_fit.make_longparamlist(meta, params, chanrng=2)
    )

    assert longparamlist == [
        ['time_offset', 'c0', 'per', 'inc', 'limb_dark'],
        ['time_offset', 'c0_ch1', 'per', 'inc_ch1', 'limb_dark'],
    ]
    assert paramtitles == ['time_offset', 'c0', 'per', 'inc', 'limb_dark']
    assert freenames == ['c0', 'c0_ch1', 'per']
    assert updated.c0_ch1.values == ['c0_ch1', 1.0, 'free']
    assert updated.inc_ch1.values == ['inc_ch1', 89.0, 'fixed']


def test_make_longparamlist_prefers_wavelength_groups_and_requires_base():
    """Wavelength-group parameters need a base channel and take precedence."""
    params = _params(c0=(1.0, 'free'), c0_wl2=(2.0, 'free'))
    meta = SimpleNamespace(multwhite=False, sharedp=True, wl_groups=[0, 2])

    longparamlist, _, freenames, _ = s5_fit.make_longparamlist(
        meta, params, chanrng=2
    )

    assert longparamlist == [['time_offset', 'c0'],
                             ['time_offset', 'c0_wl2']]
    assert freenames == ['c0', 'c0_wl2']

    missing_base = _params(depth_wl2=(0.1, 'free'))
    with pytest.raises(AssertionError, match='required for wl=0'):
        s5_fit.make_longparamlist(meta, missing_base, chanrng=2)


def test_group_variables_extracts_priors_and_independent_values():
    """Fitter setup should separate free and independent values."""
    params = _params(c0=(1.0, 'free', 0.0, 2.0, 'U'),
                     per=(3.0, 'fixed'),
                     depth=(0.1, 'shared', -1.0, 1.0, 'N'),
                     limb_dark=('uniform', 'independent'))
    component = _FlatModel(parameters=params, freenames=['c0', 'depth'])
    model = models.CompositeModel([component])
    model.freenames = ['c0', 'depth']

    freepars, prior1, prior2, priortype, indep_vars = (
        fitters.group_variables(model)
    )

    np.testing.assert_allclose(freepars, [1.0, 0.1])
    np.testing.assert_allclose(prior1, [0.0, -1.0])
    np.testing.assert_allclose(prior2, [2.0, 1.0])
    np.testing.assert_array_equal(priortype, ['U', 'N'])
    assert indep_vars['per'] == 3.0
    assert indep_vars['limb_dark'] == 'uniform'


def test_lightcurve_initialization_validates_shapes_and_scatter_mult():
    """LightCurve construction validates shapes and pre-inflates errors."""
    params = _params(scatter_mult=(2.0, 'free'),
                     scatter_mult_ch1=(3.0, 'free'))
    log = _Log()

    lc = LightCurve(time=np.arange(5), flux=np.ones(5), channel=0,
                    nchannel=2, log=log, longparamlist=[[], []],
                    parameters=params,
                    freenames=['scatter_mult', 'scatter_mult_ch1'],
                    unc=np.full(5, 0.1), multwhite=True, nints=[2, 3])

    assert lc.nchannel_fitted == 2
    np.testing.assert_allclose(lc.unc_fit, [0.2, 0.2, 0.3, 0.3, 0.3])

    with pytest.raises(ValueError, match='Time and flux'):
        LightCurve(time=np.arange(2), flux=np.ones(3), channel=0, nchannel=1,
                   log=log, longparamlist=[[]], parameters=Parameters(),
                   freenames=[], unc=np.ones(2))


def test_lightcurve_fit_selects_fitter_and_rejects_unknown(monkeypatch):
    """Fit dispatch should call requested and reject invalid fitters."""
    lc = LightCurve(time=np.arange(2), flux=np.ones(2), channel=0, nchannel=1,
                    log=_Log(), longparamlist=[[]], parameters=Parameters(),
                    freenames=[], unc=np.ones(2))
    model = SimpleNamespace()
    meta = SimpleNamespace()
    fit_result = SimpleNamespace(name='fit-result')

    monkeypatch.setattr(fitters, 'lsqfitter',
                        lambda *args, **kwargs: fit_result)

    lc.fit(model, meta, _Log(), fitter='lsq')
    assert lc.results[0].name == 'fit-result'

    with pytest.raises(ValueError, match='not a valid fitter'):
        lc.fit(model, meta, _Log(), fitter='bogus')


def test_initialize_emcee_walkers_stays_within_uniform_and_log_priors():
    """emcee walkers should start inside uniform and log-uniform bounds."""
    np.random.seed(1234)
    meta = SimpleNamespace(run_nwalkers=12, verbose=False)
    freepars = np.array([0.5, 10.0, 1.0])
    prior1 = np.array([0.0, np.log(1.0), 1.0])
    prior2 = np.array([1.0, np.log(100.0), 0.1])
    priortype = np.array(['U', 'LU', 'N'])

    pos, nwalkers = fitters.initialize_emcee_walkers(
        meta, _Log(), ndim=3, lsq_sol=None, freepars=freepars.copy(),
        prior1=prior1, prior2=prior2, priortype=priortype
    )

    assert nwalkers == 12
    assert pos.shape == (12, 3)
    assert np.all((pos[:, 0] >= 0.0) & (pos[:, 0] <= 1.0))
    assert np.all((np.log(pos[:, 1]) >= np.log(1.0)) &
                  (np.log(pos[:, 1]) <= np.log(100.0)))
    assert np.all(pos[:, 1] > 0)


def test_initialize_emcee_walkers_moves_boundary_start_to_midpoint():
    """Walker initialization should recover when LSQ lands on a bound."""
    np.random.seed(4321)
    meta = SimpleNamespace(run_nwalkers=8, verbose=False)
    freepars = np.array([1.0])
    prior1 = np.array([0.0])
    prior2 = np.array([1.0])
    priortype = np.array(['U'])

    pos, nwalkers = fitters.initialize_emcee_walkers(
        meta, _Log(), ndim=1, lsq_sol=None, freepars=freepars,
        prior1=prior1, prior2=prior2, priortype=priortype
    )

    assert nwalkers == 8
    assert freepars[0] == pytest.approx(0.5)
    assert np.all((pos[:, 0] >= 0.0) & (pos[:, 0] <= 1.0))


def test_save_fit_writes_fitparams_and_stage5_table(monkeypatch, tmp_path):
    """Fit saving should write parameters and S5 table arrays."""
    params = _params(c0=(1.0, 'free', 0.0, 2.0, 'U'))
    component = _FlatModel(parameters=params, freenames=['c0'],
                           values=np.array([1.0, 1.0]))
    model = models.CompositeModel([component], parameters=params)
    model.time = np.array([0.0, 1.0])
    lc = SimpleNamespace(white=False, share=False, channel=1, nchannel=10,
                         fitted_channels=np.array([1]),
                         time=np.array([0.0, 1.0]),
                         flux=np.array([1.1, 0.9]),
                         unc_fit=np.array([0.1, 0.2]))
    meta = SimpleNamespace(outputdir=str(tmp_path)+'/', expand=1, spec_hw=3,
                           bg_hw=7, eventlabel='evt',
                           wave_low=np.array([1.0, 2.0]),
                           wave_hi=np.array([1.5, 2.5]),
                           multwhite=True, nints=[2])
    captured = {}

    def fake_savetable(filename, meta_arg, time, wavelength, bin_width,
                       lcdata, lcerr, individual_models, model_lc,
                       residuals):
        captured['filename'] = filename
        captured['time'] = np.array(time)
        captured['wavelength'] = np.array(wavelength)
        captured['bin_width'] = np.array(bin_width)
        captured['lcdata'] = np.array(lcdata)
        captured['lcerr'] = np.array(lcerr)
        captured['individual_models'] = individual_models
        captured['model_lc'] = np.array(model_lc)
        captured['residuals'] = np.array(residuals)

    monkeypatch.setattr(fitters.astropytable, 'savetable_S5',
                        fake_savetable)
    results = AstropyTable({'Parameter': ['c0'], 'Mean': [1.0]})

    fitters.save_fit(meta, lc, model, 'lsq', results, ['c0'])

    fitparams = tmp_path/'S5_lsq_fitparams_ch01.csv'
    assert fitparams.exists()
    assert meta.tab_filename_s5.endswith('S5_evt_ap3_bg7_Table_Save_ch01.txt')
    assert captured['filename'] == meta.tab_filename_s5
    np.testing.assert_allclose(captured['wavelength'], [2.25])
    np.testing.assert_allclose(captured['bin_width'], [0.25])
    np.testing.assert_allclose(captured['model_lc'], [1.0, 1.0])
    np.testing.assert_allclose(captured['residuals'], [0.1, -0.1])


def test_utils_rebin_spec_preserves_integrated_flux_for_constant_spectrum():
    """Rebinning should preserve flux for a constant spectrum."""
    wave = np.linspace(1.0, 5.0, 5)
    flux = np.ones_like(wave)
    wavnew = np.array([1.5, 2.5, 3.5, 4.5])

    rebinned = utils.rebin_spec((wave, flux), wavnew, oversamp=100)

    np.testing.assert_allclose(rebinned, np.ones(4), rtol=2e-2)


def test_utils_filter_table_numeric_string_and_wildcard_filters():
    """Table filtering should handle numeric ranges and wildcard names."""
    table = Table({'Teff': [1000, 1500, 2000],
                   'name': ['WASP-1', 'HAT-2', 'WASP-3']})

    filtered = utils.filter_table(table, Teff=('>1200', '<=2000'),
                                  name='WASP*')

    assert list(filtered['name']) == ['WASP-3']
    with pytest.raises(KeyError):
        utils.filter_table(table, missing='==1')
    with pytest.raises(ValueError):
        utils.filter_table(table, Teff=['!1000'])


def test_utils_find_closest_returns_indices_or_values_and_outside_none():
    """Grid-neighbor lookup should support indices, values, and misses."""
    axis = np.array([0.0, 1.0, 2.0, 3.0])

    assert [arr.tolist() for arr in utils.find_closest(axis, 1.4, n=1)] == [
        [1, 2]
    ]
    assert [arr.tolist() for arr in utils.find_closest(axis, 1.4, n=1,
                                                       values=True)] == [
        [1.0, 2.0]
    ]
    assert utils.find_closest(axis, 9.0) is None


def test_utils_calc_zoom_color_gen_and_target_url():
    assert utils.calc_zoom(4, np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)
    colors = utils.color_gen(['#111111', '#222222'])
    assert [next(colors), next(colors), next(colors)] == [
        '#111111', '#222222', '#111111'
    ]
    assert utils.build_target_url('WASP 12 b').endswith('WASP%2012%20b/'
                                                        'properties/')
