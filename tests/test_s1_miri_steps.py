import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from eureka.optimizer import S1opt_optimizer, objective_funcs
from eureka.optimizer.S1opt_meta import S1optMetaClass
from eureka.S1_detector_processing import rscd
from eureka.S1_detector_processing.ramp_fitting import Eureka_RampFitStep
from eureka.S1_detector_processing.rscd import Eureka_RscdStep
from eureka.S1_detector_processing.s1_meta import S1MetaClass
from eureka.S1_detector_processing.s1_process import EurekaS1Pipeline


def test_legacy_firstframe_only_maps_to_one_group_rscd():
    """Map legacy firstframe-only processing to one-group RSCD flagging."""
    meta = S1MetaClass(topdir='.', skip_firstframe=False, skip_rscd=True)

    meta.set_MIRI_defaults()

    assert meta.skip_rscd is False
    assert meta.rscd_group_skip1 == 1
    assert meta.rscd_group_skip == 1


@pytest.mark.parametrize('first_count', [None, 0, 1])
def test_explicit_rscd_group_counts_are_preserved(first_count):
    """Preserve user-supplied RSCD group counts in the Stage 1 metadata."""
    meta = S1MetaClass(topdir='.', rscd_group_skip1=first_count,
                       rscd_group_skip=3)

    meta.set_MIRI_defaults()

    assert meta.skip_rscd is True
    assert meta.rscd_group_skip1 == first_count
    assert meta.rscd_group_skip == 3


@pytest.mark.parametrize('parameter', ['rscd_group_skip1', 'rscd_group_skip'])
@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_invalid_rscd_group_count_is_rejected(parameter, value):
    """Reject negative and noninteger RSCD group-count overrides."""
    meta = S1MetaClass(topdir='.', **{parameter: value})

    with pytest.raises(ValueError, match=parameter):
        meta.set_MIRI_defaults()


def test_miri_pipeline_disables_firstframe_and_defers_group_flags():
    """Defer MIRI group flagging when the 390 Hz correction is enabled."""
    pipeline = EurekaS1Pipeline()
    meta = SimpleNamespace(
        remove_390hz=True,
        skip_lastframe=False,
        skip_rscd=False,
        skip_reset=False,
        skip_emicorr=True,
        emicorr_algorithm='joint',
        rscd_group_skip1=1,
        rscd_group_skip=2,
    )

    pipeline._configure_miri_steps(meta)

    assert pipeline.firstframe.skip is True
    assert pipeline.lastframe.skip is True
    assert pipeline.rscd.skip is True
    assert pipeline.rscd.group_skip1 == 1
    assert pipeline.rscd.group_skip == 2


def test_miri_pipeline_runs_rscd_normally_without_390hz_removal():
    """Run RSCD in its normal pipeline position without 390 Hz removal."""
    pipeline = EurekaS1Pipeline()
    meta = SimpleNamespace(
        remove_390hz=False,
        skip_lastframe=False,
        skip_rscd=False,
        skip_reset=False,
        skip_emicorr=True,
        emicorr_algorithm='joint',
        rscd_group_skip1=None,
        rscd_group_skip=None,
    )

    pipeline._configure_miri_steps(meta)

    assert pipeline.firstframe.skip is True
    assert pipeline.lastframe.skip is False
    assert pipeline.rscd.skip is False


def test_pipeline_configuration_preserves_custom_steps(monkeypatch, tmp_path):
    """Keep configured step instances, reference overrides, and parents."""
    pipeline = EurekaS1Pipeline(
        config_file=str(tmp_path / 'pipeline.cfg'),
        steps={
            'rscd': {'override_rscd': str(tmp_path / 'rscd.fits'),
                     'save_results': True},
            'ramp_fit': {'override_gain': str(tmp_path / 'gain.fits'),
                         'save_opt': True},
        })
    rscd_step = pipeline.rscd
    ramp_step = pipeline.ramp_fit
    meta = S1MetaClass(topdir=str(tmp_path), inst='miri')
    meta.set_MIRI_defaults()
    meta.set_defaults()
    monkeypatch.setattr(pipeline, 'run', lambda filename: None)

    pipeline.run_eurekaS1('uncal.fits', meta, SimpleNamespace())

    assert pipeline.rscd is rscd_step
    assert pipeline.ramp_fit is ramp_step
    assert isinstance(rscd_step, Eureka_RscdStep)
    assert isinstance(ramp_step, Eureka_RampFitStep)
    for step, name in [(rscd_step, 'rscd'), (ramp_step, 'ramp_fit')]:
        assert step.parent is pipeline
        assert step.name == name
        assert step.config_file == pipeline.config_file
        assert step.search_attr('output_dir') == meta.outputdir
    assert rscd_step.override_rscd == str(tmp_path / 'rscd.fits')
    assert rscd_step.save_results is True
    assert ramp_step.override_gain == str(tmp_path / 'gain.fits')
    assert ramp_step.save_opt is True


def test_deferred_miri_flags_use_lastframe_then_rscd(monkeypatch):
    """Reuse configured steps in order and restore their deferred flags."""
    calls = []
    pipeline = EurekaS1Pipeline(output_dir='Stage1')
    pipeline.rscd.override_rscd = 'custom_rscd.fits'
    pipeline.rscd.group_skip1 = 1
    pipeline.rscd.group_skip = 2
    pipeline.lastframe.skip = True
    pipeline.rscd.skip = True

    def record(step, model):
        assert step.skip is False
        assert step.parent is pipeline
        assert step.search_attr('output_dir') == 'Stage1'
        calls.append(step.name)
        return model

    monkeypatch.setattr(pipeline.lastframe, 'run',
                        lambda model: record(pipeline.lastframe, model))
    monkeypatch.setattr(pipeline.rscd, 'run',
                        lambda model: record(pipeline.rscd, model))

    step = pipeline.ramp_fit
    step.s1_meta = SimpleNamespace(
        skip_lastframe=False,
        skip_rscd=False,
        rscd_group_skip1=1,
        rscd_group_skip=2,
    )
    model = object()

    result = step._apply_deferred_miri_group_flags(model)

    assert result is model
    assert calls == ['lastframe', 'rscd']
    assert pipeline.lastframe.skip is True
    assert pipeline.rscd.skip is True
    assert pipeline.rscd.override_rscd == 'custom_rscd.fits'
    assert pipeline.rscd.group_skip1 == 1
    assert pipeline.rscd.group_skip == 2


@pytest.mark.parametrize('failing_step', ['lastframe', 'rscd'])
def test_deferred_miri_flags_restore_skip_after_failure(
        monkeypatch, failing_step):
    """Restore deferred flags even when either correction raises."""
    pipeline = EurekaS1Pipeline()
    pipeline.lastframe.skip = pipeline.rscd.skip = True
    pipeline.ramp_fit.s1_meta = SimpleNamespace(
        skip_lastframe=False, skip_rscd=False)
    monkeypatch.setattr(pipeline.lastframe, 'run', lambda model: model)
    monkeypatch.setattr(pipeline.rscd, 'run', lambda model: model)

    def fail(model):
        raise RuntimeError('correction failed')

    monkeypatch.setattr(getattr(pipeline, failing_step), 'run', fail)

    with pytest.raises(RuntimeError, match='correction failed'):
        pipeline.ramp_fit._apply_deferred_miri_group_flags(object())

    assert pipeline.lastframe.skip is True
    assert pipeline.rscd.skip is True


@pytest.mark.parametrize('skip_lastframe,skip_rscd',
                         [(True, True), (True, False), (False, True)])
def test_deferred_miri_flags_honor_user_skip_settings(
        monkeypatch, skip_lastframe, skip_rscd):
    """Keep user-skipped corrections disabled in the deferred path."""
    pipeline = EurekaS1Pipeline()
    pipeline.lastframe.skip = pipeline.rscd.skip = True
    pipeline.ramp_fit.s1_meta = SimpleNamespace(
        skip_lastframe=skip_lastframe, skip_rscd=skip_rscd)
    calls = []
    for name in ['lastframe', 'rscd']:
        monkeypatch.setattr(getattr(pipeline, name), 'run',
                            lambda model, name=name:
                            calls.append(name) or model)

    pipeline.ramp_fit._apply_deferred_miri_group_flags(object())

    assert calls == [name for name, skip in
                     [('lastframe', skip_lastframe), ('rscd', skip_rscd)]
                     if not skip]
    assert pipeline.lastframe.skip is True
    assert pipeline.rscd.skip is True


def test_rscd_step_uses_both_user_group_counts(monkeypatch):
    """Pass both user-supplied group counts to the JWST RSCD correction."""
    model = SimpleNamespace(
        meta=SimpleNamespace(
            instrument=SimpleNamespace(detector='MIRIMAGE'),
            cal_step=SimpleNamespace(),
        )
    )
    correction_args = []

    monkeypatch.setattr(
        Eureka_RscdStep, 'prepare_output',
        lambda self, step_input, open_as_type: model)
    monkeypatch.setattr(
        rscd.rscd_sub, 'correction_skip_groups',
        lambda result, group_skip1, group_skip:
        correction_args.append((group_skip1, group_skip)) or result)

    step = Eureka_RscdStep()
    step.group_skip1 = 1
    step.group_skip = 3

    result = step.process(model)

    assert result is model
    assert correction_args == [(1, 3)]


@pytest.mark.parametrize('first_count,expected',
                         [(None, 4), (0, 0), (1, 1)])
def test_rscd_step_can_mix_user_and_crds_group_counts(
        monkeypatch, first_count, expected):
    """Use CRDS's later count for inheritance and retain explicit overrides."""
    model = SimpleNamespace(
        meta=SimpleNamespace(
            instrument=SimpleNamespace(detector='MIRIMAGE'),
            cal_step=SimpleNamespace(),
        )
    )
    correction_args = []

    class FakeRscdModel:
        """Provide a minimal context manager for a mocked RSCD model."""

        def __enter__(self):
            """Enter the mocked reference-model context.

            Returns
            -------
            self : FakeRscdModel
                The mocked RSCD reference model.
            """
            return self

        def __exit__(self, *args):
            """Leave the mocked reference-model context.

            Parameters
            ----------
            *args : tuple
                Exception details supplied by the context manager protocol.

            Returns
            -------
            suppress_exception : bool
                False so that any exception is propagated.
            """
            return False

    monkeypatch.setattr(
        Eureka_RscdStep, 'prepare_output',
        lambda self, step_input, open_as_type: model)
    monkeypatch.setattr(
        Eureka_RscdStep, 'get_reference_file',
        lambda self, result, reference_type: 'rscd.fits')
    monkeypatch.setattr(
        rscd.datamodels, 'RSCDModel',
        lambda filename: FakeRscdModel())
    monkeypatch.setattr(
        rscd.rscd_sub, 'get_rscd_parameters',
        lambda result, reference: {'skip_int1': 2, 'skip_int2p': 4})
    monkeypatch.setattr(
        rscd.rscd_sub, 'correction_skip_groups',
        lambda result, group_skip1, group_skip:
        correction_args.append((group_skip1, group_skip)) or result)

    step = Eureka_RscdStep()
    step.group_skip1 = first_count
    step.group_skip = None

    result = step.process(model)

    assert result is model
    assert correction_args == [(expected, 4)]


@pytest.mark.parametrize('first_count', [None, 0, 1])
@pytest.mark.parametrize('later_count', [0, 3])
@pytest.mark.parametrize('integration_start', [1, 5])
def test_rscd_inheritance_flags_real_ramps(
        monkeypatch, first_count, later_count, integration_start):
    """Apply shared or explicit counts to real, possibly segmented ramps."""
    shape = (3, 10, 2, 2)
    with rscd.datamodels.RampModel(
            data=np.zeros(shape, dtype=np.float32),
            groupdq=np.zeros(shape, dtype=np.uint8),
            pixeldq=np.zeros(shape[2:], dtype=np.uint32)) as model:
        model.meta.instrument.detector = 'MIRIMAGE'
        model.meta.exposure.ngroups = shape[1]
        model.meta.exposure.nints = integration_start + shape[0] - 1
        model.meta.exposure.integration_start = integration_start
        step = Eureka_RscdStep(group_skip1=first_count,
                               group_skip=later_count)
        monkeypatch.setattr(
            step, 'get_reference_file',
            lambda *args: pytest.fail('Explicit later count needs no CRDS'))

        with step.process(model) as result:
            assert result.meta.cal_step.rscd == 'COMPLETE'
            effective_first = (later_count if first_count is None
                               else first_count)
            for i in range(shape[0]):
                count = (effective_first if i == 0 and integration_start == 1
                         else later_count)
                expected = np.broadcast_to(
                    (np.arange(shape[1]) < count)[:, None, None], shape[1:])
                flagged = (result.groupdq[i] &
                           rscd.datamodels.dqflags.group['DO_NOT_USE']) != 0
                np.testing.assert_array_equal(flagged, expected)


@pytest.mark.parametrize('photometry', [True, False])
@pytest.mark.parametrize('requested', [None, ['rscd_group_skip1']])
def test_miri_optimizer_defaults_use_shared_count(
        monkeypatch, photometry, requested):
    """Sweep only the shared count by default, allowing explicit opt-ins."""
    monkeypatch.setattr(S1opt_optimizer.util, 'readfiles', lambda meta: meta)
    kwargs = {} if requested is None else {'params_to_optimize_s1': requested}
    meta = S1optMetaClass(
        topdir='.', inst='miri', photometry=photometry, **kwargs)

    assert meta.params_to_optimize_s1 == (
        ['jump_rejection_threshold', 'skip_lastframe', 'rscd_group_skip',
         'skip_rscd'] if requested is None else requested)
    assert list(meta.sweep_rscd_group_skip1) == list(range(6))


@pytest.fixture
def rscd_optimizer(monkeypatch, tmp_path):
    """Run real optimizer sweeps with calibration and fitness substituted."""
    (tmp_path / 'S1_test.ecf').write_text(
        'inputdir Stage0\noutputdir Stage1\nskip_rscd True\n'
        '# Group-count parameters are absent from this legacy ECF.')
    output = tmp_path / 'optimizer'
    output.mkdir()
    meta = SimpleNamespace(
        params_to_optimize_s1=['rscd_group_skip'],
        sweep_rscd_group_skip1=[0, 2], sweep_rscd_group_skip=[0, 2],
        sweep_skip_rscd=[True, False], sweep_skip_lastframe=[True, False],
        topdir=str(tmp_path) + os.sep, inputdir='Stage0',
        outputdir=str(output) + os.sep, eventlabel='test', verbose=False,
        delete_intermediate=False, delete_final=False, isplots_S1opt=0,
        scaling_MAED_spec=0.01, scaling_MAED_white=1.0,
        copy_ecf=lambda: None)
    log = SimpleNamespace(writelog=lambda *args, **kwargs: None,
                          closelog=lambda: None)
    calls = []

    def initialize(*args, **kwargs):
        s1_meta = S1MetaClass(folder=str(tmp_path), eventlabel='test',
                              topdir=str(tmp_path), rscd_group_skip=3)
        return s1_meta, None, None, None

    def calibrate(eventlabel, input_meta):
        input_meta.set_MIRI_defaults()
        calls.append(dict(input_meta.params))
        return input_meta

    def fitness(*args):
        last = calls[-1]
        count = last['rscd_group_skip1']
        if count is None:
            count = last['rscd_group_skip']
        return -1 if last['skip_rscd'] else count

    monkeypatch.setattr(S1opt_optimizer, 'initialize_meta', initialize)
    monkeypatch.setattr(S1opt_optimizer.s1, 'rampfitJWST', calibrate)
    monkeypatch.setattr(objective_funcs, '_calculate_fitness', fitness)
    return meta, log, calls


@pytest.mark.parametrize('parameter', ['rscd_group_skip1', 'rscd_group_skip'])
def test_rscd_count_sweep_keeps_rscd_enabled(rscd_optimizer, parameter):
    """Enable RSCD for each count candidate and retain it in the result."""
    meta, log, calls = rscd_optimizer

    _, _, history, best = S1opt_optimizer.optimize(
        meta, log, {}, {}, parameter, 'test', None, 1)

    assert [call[parameter] for call in calls] == [0, 2]
    assert all(call['skip_rscd'] is False for call in calls)
    assert best == {parameter: 0, 'skip_rscd': False}
    assert history[parameter] == 0


@pytest.mark.parametrize('parameter', [
    'rscd_group_skip1__skip_rscd', 'skip_rscd__rscd_group_skip',
    'rscd_group_skip__skip_rscd'])
def test_joint_rscd_sweep_optimizes_count_before_skip(
        rscd_optimizer, parameter):
    """Choose group counts with RSCD enabled before comparing skip values."""
    meta, log, calls = rscd_optimizer

    _, _, history, best = S1opt_optimizer.optimize(
        meta, log, {}, {}, parameter, 'test', None, 1)

    count_parameter = next(name for name in parameter.split('__')
                           if name != 'skip_rscd')
    assert len(calls) == 4
    assert [call['skip_rscd'] for call in calls] == [False, False, True, False]
    assert [call[count_parameter] for call in calls] == [0, 2, 0, 0]
    assert list(history) == [count_parameter, 'skip_rscd']
    assert best == {count_parameter: 0, 'skip_rscd': True}


def test_group_count_pair_enables_rscd(rscd_optimizer):
    """Enable RSCD when jointly sweeping both group-count parameters."""
    meta, log, calls = rscd_optimizer
    parameter = 'rscd_group_skip1__rscd_group_skip'

    _, _, _, best = S1opt_optimizer.optimize(
        meta, log, {}, {}, parameter, 'test', None, 1)

    assert len(calls) == 4
    assert all(call['skip_rscd'] is False for call in calls)
    assert best['skip_rscd'] is False


def test_skip_rscd_sweep_can_replace_enabled_dependency(rscd_optimizer):
    """Let a subsequent skip sweep disable the optimized correction."""
    meta, log, calls = rscd_optimizer
    _, _, history, best = S1opt_optimizer.optimize(
        meta, log, {}, {}, 'rscd_group_skip', 'test', None, 1)

    _, _, _, best = S1opt_optimizer.optimize(
        meta, log, history, best, 'skip_rscd', 'test', None, 1)

    assert best == {'rscd_group_skip': 0, 'skip_rscd': True}
    assert [call['skip_rscd'] for call in calls[-2:]] == [True, False]


def test_failed_count_sweep_does_not_change_selected_skip(
        rscd_optimizer, monkeypatch):
    """Keep the previous result if no group-count candidate succeeds."""
    meta, log, _ = rscd_optimizer

    def fail(*args, **kwargs):
        raise ValueError('calibration failed')

    monkeypatch.setattr(S1opt_optimizer.s1, 'rampfitJWST', fail)
    best = {'skip_rscd': True}

    _, _, history, best = S1opt_optimizer.optimize(
        meta, log, {}, best, 'rscd_group_skip', 'test', None, 1)

    assert history == {}
    assert best == {'skip_rscd': True}


@pytest.fixture
def rscd_optimizer_wrapper(
        rscd_optimizer, monkeypatch):
    """Run the optimizer wrapper without filesystem setup or calibration."""
    meta, log, calls = rscd_optimizer
    output = meta.outputdir
    monkeypatch.setattr(S1opt_optimizer, 'S1optMetaClass',
                        lambda **kwargs: meta)
    monkeypatch.setattr(S1opt_optimizer.util, 'makedirectory',
                        lambda *args: 1)
    monkeypatch.setattr(S1opt_optimizer.util, 'pathdirectory',
                        lambda *args: output)
    monkeypatch.setattr(S1opt_optimizer.logedit, 'Logedit', lambda *args: log)
    monkeypatch.setattr(S1opt_optimizer.s2, 'calibrateJWST',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(S1opt_optimizer.s3, 'reduce',
                        lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(
        S1opt_optimizer.s4, 'genlc',
        lambda *args, **kwargs:
        (None, None, SimpleNamespace(maed_s4=1, maed_s4_binned=[1])))
    return meta, log, calls


def test_count_only_optimizer_saves_enabled_rscd_and_final_run(
        rscd_optimizer_wrapper):
    """Save new count parameters and RSCD's enabled state in legacy ECFs."""
    meta, _, calls = rscd_optimizer_wrapper
    output = meta.outputdir

    _, _, best = S1opt_optimizer.wrapper('test', initial_run=False)

    saved = S1MetaClass(folder=os.path.join(output, 'opt_ECFs'),
                        eventlabel='test')
    assert saved.skip_rscd is False
    assert saved.rscd_group_skip == 0
    assert calls[-1]['skip_rscd'] is False
    assert calls[-1]['rscd_group_skip'] == 0
    with open(os.path.join(output, 'best_params.json')) as stream:
        assert json.load(stream) == best


@pytest.mark.parametrize('requested', [
    ['skip_rscd', 'rscd_group_skip'],
    ['skip_rscd', 'rscd_group_skip1', 'rscd_group_skip'],
    ['rscd_group_skip__skip_rscd'],
    ['skip_rscd__rscd_group_skip1'],
    ['skip_rscd__skip_lastframe', 'rscd_group_skip'],
])
def test_wrapper_optimizes_counts_before_skip(
        rscd_optimizer_wrapper, requested):
    """Use optimized counts for skip decisions even with custom ordering."""
    meta, _, calls = rscd_optimizer_wrapper
    meta.params_to_optimize_s1 = requested

    _, history, best = S1opt_optimizer.wrapper(
        'test', initial_run=False, final_run=False)

    skip_parameter = next(p for p in history if 'skip_rscd' in p)
    assert list(history)[-1] == skip_parameter
    assert best['skip_rscd']
    count_sweeps = [p for p in history if p != skip_parameter]
    count_calls = 2 * len(count_sweeps)
    assert all(call['skip_rscd'] is False for call in calls[:count_calls])
    for parameter in count_sweeps:
        assert best[parameter] == 0
        assert all(call[parameter] == 0 for call in calls[count_calls:])


def test_joint_rscd_request_preserves_explicit_ranges(rscd_optimizer):
    """Use the requested count/skip axes when splitting a joint sweep."""
    meta, log, calls = rscd_optimizer
    parameter = 'skip_rscd__rscd_group_skip'
    setattr(meta, 'sweep_' + parameter, [[False, True], [2, 4]])

    _, _, history, best = S1opt_optimizer.optimize(
        meta, log, {}, {}, parameter, 'test', None, 1)

    assert [call['skip_rscd'] for call in calls] == [False, False, False, True]
    assert [call['rscd_group_skip'] for call in calls] == [2, 4, 2, 2]
    assert list(history) == ['rscd_group_skip', 'skip_rscd']
    assert best == {'rscd_group_skip': 2, 'skip_rscd': True}
