from hello_tokens.tokenizer.bpe import decode, encode, merge, pair_counts, train

A, B, C, D = ord("a"), ord("b"), ord("c"), ord("d")  # 97, 98, 99, 100


def test_pair_counts():
    assert pair_counts([A, A, A, B]) == {(A, A): 2, (A, B): 1}


def test_merge_scans_left_to_right():
    # "aaa" holds two overlapping "aa" pairs, but only one can be merged: the first.
    assert merge([A, A, A, B], (A, A), 256) == [256, A, B]


def test_training_on_a_toy_sentence_by_hand():
    # The classic example: "aaabdaaabac".
    # 1. "aa" occurs 4 times, more than any other pair       -> 256 = "aa"
    # 2. now "aa"+"a" and "a"+"b" tie at 2; the first one seen wins -> 257 = "aaa"
    # 3. "aaa"+"b" occurs twice                                -> 258 = "aaab"
    merges = train("aaabdaaabac", vocab_size=259)
    assert merges == {(A, A): 256, (256, A): 257, (257, B): 258}
    assert encode("aaabdaaabac", merges) == [258, D, 258, A, C]


def test_round_trip_gives_back_the_exact_text():
    text = "Once upon a time, a café served 🍰 to everyone."
    merges = train(text, vocab_size=300)
    assert decode(encode(text, merges), merges) == text


def test_text_never_seen_in_training_still_round_trips():
    # Byte-level: any text can be encoded, even characters training never saw.
    merges = train("aaabdaaabac", vocab_size=259)
    text = "zebra 🦓 and ünïcode"
    assert decode(encode(text, merges), merges) == text
