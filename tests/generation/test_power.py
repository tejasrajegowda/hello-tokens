import sys

from hello_tokens.generation.power import opt_out_of_power_throttling


def test_the_opt_out_is_accepted_on_windows_and_skipped_elsewhere():
    assert opt_out_of_power_throttling() is (sys.platform == "win32")
