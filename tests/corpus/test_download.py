import io

import pytest

from hello_tokens.corpus.download import download

CONTENT = b"Once upon a time, there was a little dog." * 1000


def fake_open(url):
    # Stands in for the network: returns the same bytes for any address.
    return io.BytesIO(CONTENT)


def test_download_writes_the_file(tmp_path):
    destination = tmp_path / "stories.txt"
    assert download("https://example.test/stories.txt", destination, len(CONTENT), fake_open)
    assert destination.read_bytes() == CONTENT


def test_a_complete_file_is_not_downloaded_again(tmp_path):
    destination = tmp_path / "stories.txt"
    destination.write_bytes(CONTENT)

    def must_not_be_called(url):
        raise AssertionError("tried to download a file that was already complete")

    assert download("https://example.test/stories.txt", destination, len(CONTENT), must_not_be_called) is False


def test_a_short_download_is_rejected_and_never_looks_finished(tmp_path):
    destination = tmp_path / "stories.txt"
    with pytest.raises(OSError):
        download("https://example.test/stories.txt", destination, len(CONTENT) + 1, fake_open)
    assert not destination.exists()
