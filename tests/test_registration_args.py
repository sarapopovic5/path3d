"""register() refuses arguments it never implemented."""

import pytest

import path3d.registration as reg


@pytest.mark.parametrize("argument", ["normalizer", "artifact_masks"])
def test_unimplemented_arguments_raise(tmp_path, argument):
    with pytest.raises(NotImplementedError):
        reg.register([], tmp_path, **{argument: object()})
