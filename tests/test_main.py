import pytest
import torch

from hello_tokens.__main__ import main


def test_check_passes_on_a_gpu_machine():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU on this machine")
    assert main(["check"]) == 0


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["fly"])
