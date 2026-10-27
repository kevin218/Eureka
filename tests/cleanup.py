"""Portable cleanup for generated integration-test products."""
import os
import shutil
from pathlib import Path


def remove_test_outputs(pattern, root=None):
    """Remove matching outputs strictly inside the test data directory."""
    root = Path(__file__).resolve().parent if root is None else Path(root)
    root = root.resolve()
    data_root = (root / 'data').resolve()
    if not data_root.is_relative_to(root):
        raise ValueError('The test data directory must be inside '
                         'the test root.')
    pattern = os.fspath(pattern).replace('\\', '/')
    paths = list(root.glob(pattern))
    # Validate all targets, including symlinks, before removing anything.
    for path in paths:
        target = path.resolve()
        if target == data_root or not target.is_relative_to(data_root):
            raise ValueError(f'Refusing to remove non-output path: {path}')
    for path in paths:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
