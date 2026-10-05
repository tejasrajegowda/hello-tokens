import pytest
import torch
import torch.nn.functional as F

from hello_tokens.judge.questions import Question, build_questions, sentences
from hello_tokens.judge.scoring import (
    Judgement, choose_threshold, fit_temperature, option_calibration, outcome, score_options,
)
from hello_tokens.model.config import ModelConfig
from hello_tokens.model.gpt import GPT
from hello_tokens.tokenizer.tokenizer import END_OF_TEXT, Tokenizer

STORY = ("Tom had a red ball. He liked to throw it high. One day the ball went over the fence. "
         "Tom was very sad. His friend Sam came to help. They climbed the old tree. "
         "Sam found the ball in the grass. Tom said thank you to Sam.")
TEXT = (STORY + END_OF_TEXT + STORY.replace("Tom", "Mia").replace("He ", "She ") + END_OF_TEXT) * 3


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer.train(TEXT, 300)


def test_sentences_are_split_at_their_ends_and_tiny_fragments_dropped():
    assert sentences('He said, "Hi!" Then he left. Ok. The end came soon.') == [
        'He said, "Hi!"', "Then he left.", "The end came soon."]


def test_each_question_hides_the_true_next_sentence_among_later_ones(tokenizer):
    questions = build_questions(TEXT, tokenizer, count=4, seed=1)
    assert len(questions) == 4
    for q in questions:
        story = next(s for s in TEXT.split(END_OF_TEXT) if s.startswith(q.context))
        parts = sentences(story)
        i = len(sentences(q.context))
        assert q.options[q.answer] == parts[i]  # the true continuation
        later = parts[i + 1 :]
        assert all(o in later for k, o in enumerate(q.options) if k != q.answer)  # same story, later on
        assert len(q.options) == 4 and len(set(q.options)) == 4
    assert build_questions(TEXT, tokenizer, count=4, seed=1) == questions  # repeatable


@pytest.fixture
def model(tokenizer):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=tokenizer.vocab_size, context=48, width=32, layers=2, heads=4)).eval()


def test_option_scores_equal_their_log_probabilities_computed_one_at_a_time(model, tokenizer):
    q = Question("Tom had a red ball.", ["He liked to throw it high.", "Tom was very sad.", "They ran.", "Sam came."], 0)
    judgement = score_options(model, tokenizer, q)
    context = tokenizer.encode(q.context)
    with torch.no_grad():
        for i, option in enumerate(q.options):
            ids = tokenizer.encode(" " + option)
            sequence = torch.tensor([context + ids])
            log_probs = F.log_softmax(model(sequence[:, :-1])[0], -1)  # no padding: scored alone
            expected = sum(log_probs[len(context) - 1 + j, t].item() for j, t in enumerate(ids))
            assert judgement.summed[i] == pytest.approx(expected, abs=1e-4)
            assert judgement.mean[i] == pytest.approx(expected / len(ids), abs=1e-4)


def test_a_context_longer_than_the_window_keeps_its_end(model, tokenizer):
    q = Question(STORY * 4, ["Tom said thank you to Sam.", "Tom was very sad.", "They climbed.", "Sam found it."], 0)
    judgement = score_options(model, tokenizer, q)  # 4 stories >> 48 tokens: must not fail
    assert len(judgement.summed) == 4


HAND = [Judgement([0.0, -5.0, -5.0, -5.0], [0, 0, 0, 0]),  # very sure, right
        Judgement([0.0, -0.1, -5.0, -5.0], [0, 0, 0, 0]),  # unsure, right
        Judgement([-0.2, 0.0, -5.0, -5.0], [0, 0, 0, 0])]  # unsure, wrong (chooses 1, answer 0)
ANSWERS = [0, 0, 0]


def test_the_switch_by_hand():
    everything = outcome(HAND, ANSWERS, "summed", 0.0)
    assert (everything.coverage, everything.accuracy) == (1.0, pytest.approx(2 / 3))
    sure_only = outcome(HAND, ANSWERS, "summed", 0.9)
    assert (sure_only.coverage, sure_only.accuracy) == (pytest.approx(1 / 3), 1.0)
    # For 100% accuracy the judge must answer only the very sure question.
    threshold = choose_threshold(HAND, ANSWERS, "summed", target=1.0)
    assert threshold == pytest.approx(float(HAND[0].probabilities("summed").max()))
    assert choose_threshold(HAND, ANSWERS, "summed", target=0.6) < 0.6  # 2/3 >= 0.6: answer everything


def test_the_judge_s_own_calibration_counts_every_question():
    result = option_calibration(HAND, ANSWERS, "summed")
    assert result.predictions == 3 and result.accuracy == pytest.approx(2 / 3)


def test_temperature_scaling_recovers_an_overconfidence_factor():
    # True odds: softmax of `honest`. The judge reports scores 5x too sharp. The answers are drawn
    # from the true odds, so the right temperature is 5: it should be found, it should fix the
    # calibration, and it must not change which option is chosen.
    g = torch.Generator().manual_seed(0)
    honest = torch.randn(3000, 4, generator=g)
    answers = torch.multinomial(torch.softmax(honest, -1), 1, generator=g).squeeze(1).tolist()
    judgements = [Judgement((5 * row).tolist(), row.tolist()) for row in honest]
    t = fit_temperature(judgements, answers, "summed")
    assert t == pytest.approx(5, rel=0.15)
    assert option_calibration(judgements, answers, "summed", t).ece < option_calibration(judgements, answers, "summed").ece / 3
    assert outcome(judgements, answers, "summed", 0.0, t).accuracy == outcome(judgements, answers, "summed", 0.0).accuracy