from hello_tokens.corpus.sample import read_sample
from hello_tokens.tokenizer.tokenizer import END_OF_TEXT


def test_the_sample_ends_at_the_last_complete_story(tmp_path):
    path = tmp_path / "stories.txt"
    text = f"First story.\n{END_OF_TEXT}\nSecond story.\n{END_OF_TEXT}\nThird, cut off"
    path.write_bytes(text.encode("utf-8"))  # raw bytes, as downloaded (no Windows newline changes)
    sample = read_sample(path, megabytes=1)
    assert sample.endswith("Second story.\n")
    assert "Third" not in sample


def test_a_cut_inside_a_multi_byte_character_is_harmless(tmp_path):
    path = tmp_path / "stories.txt"
    text = f"A café.\n{END_OF_TEXT}\n“Hi”"  # “ is 3 bytes in UTF-8
    path.write_bytes(text.encode("utf-8"))  # raw bytes, as downloaded (no Windows newline changes)
    cut = (len(text.encode("utf-8")) - 2) / 1_000_000  # stop in the middle of the closing quote
    assert read_sample(path, cut) == "A café.\n"
