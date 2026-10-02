import io

import pytest

from hello_tokens.corpus.download import download

CONTENT = b"Once upon a time, there was a little dog." * 1000
URL = "https://example.test/stories.txt"


def fake_open(url, start):
    # Stands in for the network: serves the content from byte `start`.
    return io.BytesIO(CONTENT[start:])


def test_download_writes_the_file(tmp_path):
    destination = tmp_path / "stories.txt"
    assert download(URL, destination, len(CONTENT), fake_open, wait_seconds=0)
    assert destination.read_bytes() == CONTENT


def test_a_complete_file_is_not_downloaded_again(tmp_path):
    destination = tmp_path / "stories.txt"
    destination.write_bytes(CONTENT)

    def must_not_be_called(url, start):
        raise AssertionError("tried to download a file that was already complete")

    assert download(URL, destination, len(CONTENT), must_not_be_called) is False


def test_an_interrupted_download_resumes_where_it_stopped(tmp_path):
    destination = tmp_path / "stories.txt"
    destination.with_name("stories.txt.part").write_bytes(CONTENT[:5000])
    requested_from = []

    def recording_open(url, start):
        requested_from.append(start)
        return fake_open(url, start)

    assert download(URL, destination, len(CONTENT), recording_open, wait_seconds=0)
    assert requested_from == [5000]  # asked only for the missing part
    assert destination.read_bytes() == CONTENT


def test_a_dropped_connection_is_retried(tmp_path):
    destination = tmp_path / "stories.txt"
    calls = []

    def flaky_open(url, start):
        calls.append(start)
        if len(calls) == 1:
            return io.BytesIO(CONTENT[start:start + 7000])  # the connection drops early
        return fake_open(url, start)

    assert download(URL, destination, len(CONTENT), flaky_open, wait_seconds=0)
    assert calls == [0, 7000]
    assert destination.read_bytes() == CONTENT


def test_giving_up_leaves_no_finished_looking_file(tmp_path):
    destination = tmp_path / "stories.txt"

    def always_short(url, start):
        return io.BytesIO(b"")  # the server never sends anything

    with pytest.raises(OSError):
        download(URL, destination, len(CONTENT), always_short, attempts=3, wait_seconds=0)
    assert not destination.exists()
