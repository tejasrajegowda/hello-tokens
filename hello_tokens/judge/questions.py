"""A multiple-choice test built automatically from held-out stories.

Each question shows the start of a story and four candidate next sentences: the true one, and three
sentences from later in the same story. Taking the wrong answers from the same story means the names,
the topic and the style can't give the right one away; picking the three closest in length to the
right answer means its length can't either. No hand labelling is needed, and the answer is known.
"""

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from hello_tokens.tokenizer.tokenizer import END_OF_TEXT, Tokenizer

OPTIONS = 4
MIN_WORDS = 3
# A sentence ends at . ! or ?, or one of those followed by a closing quote, and then whitespace. The
# split happens in the whitespace, so the punctuation and the quote stay with their sentence.
SENTENCE_END = re.compile(r'(?<=[.!?]["”])\s+|(?<=[.!?])\s+')


@dataclass(frozen=True)
class Question:
    context: str  # the story so far
    options: list[str]  # four candidate next sentences, in shuffled order
    answer: int  # index of the true next sentence


def sentences(story: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_END.split(story.strip())]
    return [p for p in parts if len(p.split()) >= MIN_WORDS]


def build_questions(text: str, tokenizer: Tokenizer, count: int, seed: int = 0, min_context: int = 2) -> list[Question]:
    """Up to `count` questions, at most one per story, chosen with a fixed seed."""
    rng = random.Random(seed)
    stories = [s for s in text.split(END_OF_TEXT) if s.strip()]
    rng.shuffle(stories)
    questions = []
    for story in stories:
        parts = sentences(story)
        # The true sentence needs at least `min_context` sentences before it and 3 after it.
        positions = range(min_context, len(parts) - (OPTIONS - 1))
        if not positions:
            continue
        i = rng.choice(positions)
        true = parts[i]
        later = parts[i + 1 :]
        length = len(tokenizer.encode(" " + true))
        # The three later sentences closest in length (ties keep story order), then shuffle all four.
        wrong = sorted(later, key=lambda s: abs(len(tokenizer.encode(" " + s)) - length))[: OPTIONS - 1]
        if true in wrong:  # a repeated sentence would make two right answers
            continue
        options = wrong + [true]
        rng.shuffle(options)
        questions.append(Question(" ".join(parts[:i]), options, options.index(true)))
        if len(questions) == count:
            break
    return questions


def save_questions(questions: list[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(q) for q in questions], indent=1) + "\n", encoding="utf-8")


def load_questions(path: Path) -> list[Question]:
    return [Question(**q) for q in json.loads(path.read_text(encoding="utf-8"))]
