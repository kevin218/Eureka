"""Regression coverage for portable integration-test cleanup."""
import pytest

from .cleanup import remove_test_outputs


def test_cleanup_expands_globs_and_preserves_fixed_inputs(tmp_path):
    stage2 = tmp_path / 'data' / 'a target' / 'Stage2'
    fixed_input = stage2 / 'fixed_calints.fits'
    generated = stage2 / 'S2_run1'
    generated.mkdir(parents=True)
    (generated / 'generated_calints.fits').touch()
    fixed_input.touch()

    remove_test_outputs(r'data\a target\Stage2\S2_*', root=tmp_path)

    assert not generated.exists()
    assert fixed_input.is_file()


def test_cleanup_removes_files_and_allows_missing_outputs(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    log = data / 'test.log'
    log.touch()

    remove_test_outputs('data/test.log', root=tmp_path)
    remove_test_outputs('data/test.log', root=tmp_path)

    assert not log.exists()


@pytest.mark.parametrize('pattern', ['outside.txt', 'data'])
def test_cleanup_refuses_paths_outside_generated_data(tmp_path, pattern):
    (tmp_path / 'outside.txt').touch()
    (tmp_path / 'data').mkdir()

    with pytest.raises(ValueError, match='Refusing to remove'):
        remove_test_outputs(pattern, root=tmp_path)

    assert (tmp_path / 'outside.txt').is_file()
    assert (tmp_path / 'data').is_dir()
