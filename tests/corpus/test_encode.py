import numpy as np
import pytest

from hello_tokens.corpus.encode import encode_file, load_tokens
from hello_tokens.tokenizer.tokenizer import END_OF_TEXT, Tokenizer

TEXT = "".join(
    f"Once upon a time, story number {n} had a dog named Max. “Woof!” said Max.\n{END_OF_TEXT}\n"
    for n in range(200)
) + "The last story has no end marker."


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer.train(TEXT, vocab_size=400)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "stories.txt"
    path.write_bytes(TEXT.encode("utf-8"))  # raw bytes, as downloaded
    return path


def test_the_token_file_matches_encoding_the_text_directly(tokenizer, source, tmp_path):
    destination = tmp_path / "train.bin"
    count = encode_file(source, destination, tokenizer)
    tokens = load_tokens(destination)
    assert count == len(tokens)
    assert tokens.tolist() == tokenizer.encode(TEXT)


def test_small_blocks_give_the_same_tokens(tokenizer, source, tmp_path):
    # Blocks far smaller than a story force many carry-overs; the result must not change.
    whole, blocks = tmp_path / "whole.bin", tmp_path / "blocks.bin"
    encode_file(source, whole, tokenizer)
    encode_file(source, blocks, tokenizer, block=37)
    assert np.array_equal(load_tokens(whole), load_tokens(blocks))


def test_tokens_are_two_bytes_each_and_within_the_vocabulary(tokenizer, source, tmp_path):
    destination = tmp_path / "train.bin"
    count = encode_file(source, destination, tokenizer)
    assert destination.stat().st_size == 2 * count
    assert int(load_tokens(destination).max()) < tokenizer.vocab_size


def test_every_story_end_becomes_the_special_token(tokenizer, source, tmp_path):
    destination = tmp_path / "train.bin"
    encode_file(source, destination, tokenizer)
    assert int((load_tokens(destination) == tokenizer.end_of_text_id).sum()) == 200
