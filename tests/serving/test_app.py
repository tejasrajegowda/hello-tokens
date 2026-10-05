import json

import pytest
import torch
from fastapi.testclient import TestClient

from hello_tokens.generation.sampling import generate
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT
from hello_tokens.serving.app import Busy, Entry, TextStream, create_app
from hello_tokens.tokenizer.tokenizer import Tokenizer

TEXT = "the cat sat on the mat. the dog sat on the log. a café by the sea. " * 10


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer.train(TEXT, 280)


@pytest.fixture
def app(tokenizer):
    torch.manual_seed(0)
    config = dict(vocab_size=tokenizer.vocab_size, context=32, width=32, layers=2, heads=4)
    models = {"v1": Entry(GPT(ModelConfig(**config)), "classic"),
              "v2": Entry(GPT(ModelConfig(**config, norm="rmsnorm", position="rope", feed_forward="swiglu",
                                          kv_heads=2)), "modern")}
    return create_app(models, tokenizer, "cpu")


def stream(client, **body):
    """POST /write and return its events as (kind, data) pairs."""
    with client.stream("POST", "/write", json=body) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())
    events = []
    for block in text.strip().split("\n\n"):
        kind, data = block.split("\n")
        events.append((kind.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_each_model_writes_exactly_what_generate_writes_with_the_same_seed(app, tokenizer):
    client = TestClient(app)
    events = stream(client, prompt="the cat", temperature=0.9, top_p=0.9, max_tokens=12, seed=5)
    assert events[0] == ("start", {"seed": 5, "prompt_tokens": len(tokenizer.encode("the cat"))})
    for name in ("v1", "v2"):
        text = "".join(d["text"] for kind, d in events if kind in ("token", "done") and d["model"] == name)
        expected_ids = generate(app.state.models[name].model, tokenizer.encode("the cat"), 12, 0.9, None, 0.9,
                                stop_id=tokenizer.end_of_text_id, seed=5)
        assert text == tokenizer.decode(expected_ids)


def test_every_token_reports_the_model_s_own_odds(app):
    events = stream(TestClient(app), prompt="the dog", max_tokens=6, seed=1)
    tokens = [d for kind, d in events if kind == "token"]
    assert tokens
    for d in tokens:
        assert 0 < d["probability"] <= 1
        odds = [p for _, p in d["top"]]
        assert odds == sorted(odds, reverse=True) and len(odds) <= 5 and all(p > 0 for p in odds)
        assert d["probability"] <= odds[0] + 1e-6  # nothing beats the likeliest word
        assert d["ms"] > 0


def test_both_models_finish_with_a_reason(app):
    events = stream(TestClient(app), prompt="", max_tokens=5, seed=2)  # empty prompt: a new story
    done = {d["model"]: d["reason"] for kind, d in events if kind == "done"}
    assert set(done) == {"v1", "v2"} and set(done.values()) <= {"end", "length"}


def test_bad_settings_and_long_prompts_are_refused(app):
    client = TestClient(app)
    assert client.post("/write", json={"temperature": 3}).status_code == 422
    assert client.post("/write", json={"max_tokens": 1000}).status_code == 422
    assert client.post("/write", json={"prompt": "the cat sat " * 50}).status_code == 422  # > 32 tokens


def test_a_second_story_while_one_is_running_gets_busy(app):
    client = TestClient(app)
    assert app.state.busy.try_start()  # as if a story were being written
    assert client.post("/write", json={"prompt": "the"}).status_code == 409
    app.state.busy.finish()
    assert client.post("/write", json={"prompt": "the", "max_tokens": 2}).status_code == 200


def test_an_abandoned_story_does_not_block_forever():
    busy = Busy(stale_after=0.0)
    assert busy.try_start()
    assert busy.try_start()  # the first one produced nothing in time: treated as abandoned


def test_a_character_split_across_two_tokens_comes_out_whole(tokenizer):
    stream_text = TextStream(tokenizer)
    first, second = "é".encode()  # two bytes; ids 0-255 are single bytes
    assert stream_text.feed(first) == ""  # held back: half a character
    assert stream_text.feed(second) == "é"
    assert stream_text.feed(first) == "" and stream_text.finish() == "�"  # cut off at the end


def test_the_page_and_the_model_list(app):
    client = TestClient(app)
    assert "hello-tokens playground" in client.get("/").text
    assert [m["name"] for m in client.get("/models").json()] == ["v1", "v2"]


def test_the_cache_fused_and_graphs_switches_write_the_same_story(app):
    client = TestClient(app)
    texts = []
    for switches in ({}, {"cache": True}, {"fused": True}, {"cache": True, "fused": True},
                     {"graphs": True}, {"graphs": True, "fused": True}):
        events = stream(client, prompt="the cat", max_tokens=10, seed=4, **switches)
        texts.append("".join(d["text"] for kind, d in events if kind in ("token", "done") and d["model"] == "v2"))
    assert len(set(texts)) == 1  # fp32 on the CPU: identical text whatever the switches


def test_the_judge_answers_or_abstains_for_every_model(app):
    client = TestClient(app)
    body = {"context": "the cat sat on the mat.", "options": ["the dog sat on the log.", "a café by the sea."]}
    results = client.post("/judge", json=body).json()
    assert [r["model"] for r in results] == ["v1", "v2"]
    for r in results:
        assert sum(r["probabilities"]) == pytest.approx(1.0)
        assert r["confidence"] == pytest.approx(max(r["probabilities"]))
        assert r["decision"] in ("answer", "abstain") and not r["calibrated"]  # no judge run for test models
        assert (r["fallback"] is None) == (r["decision"] == "answer")


def test_the_judge_uses_a_saved_calibration(app):
    entry = app.state.models["v1"]
    entry.judge = {"method": "mean", "temperature": 1e6, "threshold": 0.99}  # huge T: equal odds, so abstain
    results = TestClient(app).post("/judge", json={"context": "the", "options": ["cat.", "dog.", "mat."],
                                                   "fallback_tokens": 0}).json()
    v1 = results[0]
    assert v1["calibrated"] and v1["decision"] == "abstain" and v1["fallback"] is None
    assert v1["probabilities"] == pytest.approx([1 / 3] * 3, abs=1e-4)


def test_bad_judge_requests_are_refused(app):
    client = TestClient(app)
    assert client.post("/judge", json={"context": "x", "options": ["only one"]}).status_code == 422
    assert client.post("/judge", json={"context": "x", "options": ["a", " "]}).status_code == 422