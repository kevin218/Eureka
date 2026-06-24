import os
import sys

import pytest

sys.path.insert(0, '..'+os.sep+'src'+os.sep)
from eureka.lib import manageevent as me
from eureka.lib.readECF import MetaClass
from eureka.S4_generate_lightcurves.s4_meta import S4MetaClass

EVENTLABEL = 'testevent'


def _topdir(tmp_path):
    return str(tmp_path)+os.sep


def _make_current_meta(tmp_path, inputdir, allapers=True):
    topdir = _topdir(tmp_path)
    return S4MetaClass(**dict(
        allapers=allapers,
        bg_hw=7,
        clip_binned=False,
        clip_unbinned=False,
        data_format='eureka',
        eventlabel=EVENTLABEL,
        expand=1,
        inputdir=os.path.join(topdir, *inputdir.split(os.sep))+os.sep,
        inputdir_raw=inputdir,
        outputdir='Stage4',
        outputdir_raw='Stage4',
        spec_hw=5,
        topdir=topdir,
    ))


def _save_previous_meta(folder, ap, bg):
    folder.mkdir(parents=True, exist_ok=True)

    old_meta = MetaClass()
    old_meta.allapers = True
    old_meta.bg_hw = bg
    old_meta.bg_hw_range = [7, 9, 11]
    old_meta.data_format = 'eureka'
    old_meta.eventlabel = EVENTLABEL
    old_meta.expand = 1
    old_meta.filename_S3_SpecData = str(
        folder / f'S3_{EVENTLABEL}_ap{ap}_bg{bg}_SpecData.h5')
    old_meta.s3_logname = ''
    old_meta.spec_hw = ap
    old_meta.spec_hw_range = [4, 5, 6]

    me.saveevent(old_meta, str(folder / f'S3_{EVENTLABEL}_Meta_Save'))


def _load_and_filter_s4_meta(meta):
    s3_meta, meta.inputdir, meta.inputdir_raw = me.findevent(
        meta, 'S3', allowFail=False)
    meta = S4MetaClass(**me.mergeevents(meta, s3_meta).__dict__)
    meta.set_defaults()

    return me.filter_allapers_inputdir(meta)


def test_allapers_normal_inputdir_keeps_existing_pair_grid(tmp_path):
    _save_previous_meta(
        tmp_path / 'Stage3' / 'S3_2026-01-01_testevent_run1' / 'ap5_bg7',
        5, 7)

    meta = _make_current_meta(tmp_path, 'Stage3', allapers=True)
    meta = _load_and_filter_s4_meta(meta)

    assert not hasattr(meta, 'allapers_inputdir_candidates')
    assert me.get_allapers_pairs(meta) == [
        (4, 7), (4, 9), (4, 11),
        (5, 7), (5, 9), (5, 11),
        (6, 7), (6, 9), (6, 11),
    ]


def test_allapers_glob_inputdir_filters_to_matching_folders(tmp_path):
    for ap, bg in [(5, 7), (5, 9), (5, 11), (4, 9), (6, 9)]:
        _save_previous_meta(
            tmp_path / 'Stage3' / 'S3_2026-01-01_testevent_run1' /
            f'ap{ap}_bg{bg}', ap, bg)

    meta = _make_current_meta(
        tmp_path, 'Stage3/S3_*_run*/ap5_bg*', allapers=True)
    meta = _load_and_filter_s4_meta(meta)

    assert meta.allapers_inputdir_candidate_count == 3
    assert me.get_allapers_pairs(meta) == [(5, 7), (5, 9), (5, 11)]
    assert me.get_allapers_specific_inputdir(
        meta, 5, 9).endswith(f'ap5_bg9{os.sep}')


def test_allapers_glob_inputdir_no_matches_is_clear(tmp_path):
    meta = _make_current_meta(
        tmp_path, 'Stage3/S3_*_run*/ap5_bg*', allapers=True)

    with pytest.raises(AssertionError, match='No input folders matched'):
        me.findevent(meta, 'S3', allowFail=False)


def test_allapers_glob_does_not_parse_ap_bg_run_suffix(tmp_path):
    """Do not treat unsupported ap<ap>_bg<bg>_runN folders as ap/bg pairs."""
    _save_previous_meta(
        tmp_path / 'Stage3' / 'S3_2026-01-01_testevent_run1' /
        'ap5_bg7_run1', 5, 7)

    meta = _make_current_meta(
        tmp_path, 'Stage3/S3_*_run*/ap5_bg7_run*', allapers=True)

    meta = _load_and_filter_s4_meta(meta)

    assert me.get_allapers_pairs(meta) == [
        (4, 7), (4, 9), (4, 11),
        (5, 7), (5, 9), (5, 11),
        (6, 7), (6, 9), (6, 11),
    ]


def test_allapers_glob_folders_match_but_no_ap_bg_overlap(tmp_path):
    """Raise clearly when glob-matched folders do not overlap ap/bg ranges."""
    _save_previous_meta(
        tmp_path / 'Stage3' / 'S3_2026-01-01_testevent_run1' / 'ap10_bg10',
        10, 10)

    meta = _make_current_meta(
        tmp_path, 'Stage3/S3_*_run*/ap10_bg*', allapers=True)

    with pytest.raises(AssertionError,
                       match='none of their ap/bg values matched'):
        _load_and_filter_s4_meta(meta)


def test_glob_inputdir_without_allapers_uses_existing_findevent_behavior(
        tmp_path):
    _save_previous_meta(
        tmp_path / 'Stage3' / 'S3_2026-01-01_testevent_run1' / 'ap5_bg7',
        5, 7)

    meta = _make_current_meta(
        tmp_path, 'Stage3/S3_*_run*/ap5_bg*', allapers=False)
    s3_meta, _, _ = me.findevent(meta, 'S3', allowFail=False)

    assert not hasattr(meta, 'allapers_inputdir_candidates')
    assert s3_meta.spec_hw == 5
    assert s3_meta.bg_hw == 7
