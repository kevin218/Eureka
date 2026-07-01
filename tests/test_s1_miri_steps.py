from types import SimpleNamespace

import pytest

from eureka.S1_detector_processing import ramp_fitting
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


def test_explicit_rscd_group_counts_are_preserved():
    """Preserve user-supplied RSCD group counts in the Stage 1 metadata."""
    meta = S1MetaClass(topdir='.', rscd_group_skip1=1,
                       rscd_group_skip=3)

    meta.set_MIRI_defaults()

    assert meta.skip_rscd is True
    assert meta.rscd_group_skip1 == 1
    assert meta.rscd_group_skip == 3


def test_invalid_rscd_group_count_is_rejected():
    """Reject negative RSCD group-count overrides."""
    meta = S1MetaClass(topdir='.', rscd_group_skip1=-1)

    with pytest.raises(ValueError, match='rscd_group_skip1'):
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


def test_deferred_miri_flags_use_lastframe_then_rscd(monkeypatch):
    """Apply deferred lastframe and RSCD corrections in pipeline order."""
    calls = []

    class RecordingStep:
        """Record the order in which mocked pipeline steps are run."""

        def __init__(self, name):
            """Initialize a recording pipeline step.

            Parameters
            ----------
            name : str
                Step name to append to the shared call list when run.
            """
            self.name = name
            self.skip = None

        def run(self, model):
            """Record this step and return its input unchanged.

            Parameters
            ----------
            model : object
                Mock pipeline input.

            Returns
            -------
            model : object
                The unchanged mock pipeline input.
            """
            calls.append(self.name)
            return model

    monkeypatch.setattr(
        ramp_fitting, 'LastFrameStep',
        lambda: RecordingStep('lastframe'))
    monkeypatch.setattr(
        ramp_fitting, 'Eureka_RscdStep',
        lambda: RecordingStep('rscd'))

    step = Eureka_RampFitStep()
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
    assert step.rscd.group_skip1 == 1
    assert step.rscd.group_skip == 2


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


def test_rscd_step_can_mix_user_and_crds_group_counts(monkeypatch):
    """Combine a user override with the other group count from CRDS."""
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
    step.group_skip1 = 1
    step.group_skip = None

    result = step.process(model)

    assert result is model
    assert correction_args == [(1, 4)]
