import pytest

from hello_tokens.tokenizer.tokenizer import END_OF_TEXT, Tokenizer, split_chunks

STORIES = (
    "Once upon a time, there was a little dog named Max. Max loved to run.\n"
    f"{END_OF_TEXT}\n"
    "Once upon a time, a girl named Lily saw a big red ball. “Wow!” she said.\n"
    f"{END_OF_TEXT}\n"
    "One day, Max and Lily played in the park. They were happy.\n"
) * 20


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer.train(STORIES, vocab_size=400)


@pytest.mark.parametrize("text", [
    "Once upon a time, there was a dog.",
    "  two  spaces,\ttabs\nand newlines\n\n",
    "snake_case and __dunders__ and 3.14 and “quotes”",
    "",
])
def test_chunks_join_back_into_the_exact_text(text):
    assert "".join(split_chunks(text)) == text


def test_words_keep_their_leading_space():
    assert split_chunks("a big dog.") == ["a", " big", " dog", "."]


def test_vocabulary_has_the_requested_size():
    small = Tokenizer.train(STORIES, vocab_size=300)
    assert small.vocab_size == 300
    assert small.end_of_text_id == 299  # the special token takes the last id


def test_training_stops_early_when_nothing_is_left_to_merge(tokenizer):
    # This small text runs out of pairs before 400: every chunk is already one token.
    assert tokenizer.vocab_size <= 400
    assert tokenizer.end_of_text_id == tokenizer.vocab_size - 1


def test_round_trip_on_stories(tokenizer):
    assert tokenizer.decode(tokenizer.encode(STORIES)) == STORIES


def test_round_trip_on_text_never_seen(tokenizer):
    text = "A zebra 🦓 met a café owner named Zoë."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_the_end_marker_is_one_special_token(tokenizer):
    ids = tokenizer.encode(f"The end.{END_OF_TEXT}Once")
    assert ids.count(tokenizer.end_of_text_id) == 1


def test_merges_never_cross_a_word_boundary(tokenizer):
    # Each learned token may start with a space but never contains one in the middle.
    for token_id in range(256, tokenizer.end_of_text_id):
        assert b" " not in tokenizer.pieces[token_id][1:]


def test_common_words_become_single_tokens(tokenizer):
    assert len(tokenizer.encode(" upon")) == 1


def test_save_then_load_gives_the_same_tokenizer(tokenizer, tmp_path):
    path = tmp_path / "tokenizer.bpe"
    tokenizer.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    assert loaded.encode(STORIES) == tokenizer.encode(STORIES)


def test_training_is_repeatable():
    assert Tokenizer.train(STORIES, 300).merges == Tokenizer.train(STORIES, 300).merges
