"""The book in docs/book/ must stay true to the code.

Every Python block in a chapter is marked as either an excerpt of a real file
(`<!-- from: <path> -->` on the line before the fence) or an illustration (`<!-- illustration -->`).
An excerpt must match its file: each run of lines between `...` lines appears in the file as
consecutive lines, in order, compared with surrounding whitespace stripped and blank lines ignored.
A run needs at least two lines, so a common one-liner can't match by accident, and an excerpt that
starts at a decorated `def` or `class` must include the decorator. Relative links must resolve to
files that are tracked (not ignored), every `python -m hello_tokens <command>` must name a real
command, and no chapter may refer to the project's internal lesson numbers.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "docs" / "book"

FENCE = re.compile(r"^```(\w*)\s*$")
FROM = re.compile(r"^<!--\s*from:\s*(\S+)\s*-->$")
ILLUSTRATION = re.compile(r"^<!--\s*illustration\s*-->$")
LINK = re.compile(r"\]\(([^)\s]+)\)")
COMMAND = re.compile(r"python -m hello_tokens\s+([a-z_]+)")
ELISION = {"...", "# ..."}


def code_blocks(text: str) -> list[tuple[str, str, list[str]]]:
    """(language, the line before the fence, the block's lines) for every fenced block."""
    lines = text.splitlines()
    blocks, i = [], 0
    while i < len(lines):
        opening = FENCE.match(lines[i])
        if opening is None:
            i += 1
            continue
        before = lines[i - 1].strip() if i > 0 else ""
        end = i + 1
        while end < len(lines) and not lines[end].startswith("```"):
            end += 1
        blocks.append((opening.group(1), before, lines[i + 1 : end]))
        i = end + 1
    return blocks


def segments(block: list[str]) -> list[list[str]]:
    """Split an excerpt at its `...` lines into runs of stripped, non-blank lines."""
    runs, current = [], []
    for line in block:
        if line.strip() in ELISION:
            runs.append(current)
            current = []
        elif line.strip():
            current.append(line.strip())
    runs.append(current)
    return [run for run in runs if run]


def excerpt_matches(block: list[str], source: list[str]) -> bool:
    """True if every run appears in the source as consecutive non-blank lines, each after the last."""
    wanted = segments(block)
    if not wanted or any(len(run) < 2 for run in wanted):
        return False
    lines = [line.strip() for line in source if line.strip()]
    start = 0
    for number, run in enumerate(wanted):
        found = next(
            (i for i in range(start, len(lines) - len(run) + 1) if lines[i : i + len(run)] == run),
            None,
        )
        if found is None:
            return False
        decorated = found > 0 and lines[found - 1].startswith("@")
        if number == 0 and decorated and re.match(r"(async\s+)?(def|class)\s", run[0]):
            return False
        start = found + len(run)
    return True


def excerpt_problems(text: str, root: Path) -> list[str]:
    problems = []
    for number, (language, before, block) in enumerate(code_blocks(text), start=1):
        marked = FROM.match(before)
        if language != "python":
            continue
        if marked is None:
            if ILLUSTRATION.match(before) is None:
                problems.append(f"python block {number} has no `from:` or `illustration` marker")
            continue
        path = root / marked.group(1)
        if not path.is_file():
            problems.append(f"python block {number} names a missing file: {marked.group(1)}")
        elif not excerpt_matches(block, path.read_text(encoding="utf-8").splitlines()):
            problems.append(f"python block {number} does not match {marked.group(1)}")
    return problems


def without_code(text: str) -> str:
    return re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)


def ignored_by_git(path: Path) -> bool:
    check = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=ROOT, capture_output=True)
    return check.returncode == 0


def link_problems(text: str, folder: Path) -> list[str]:
    problems = []
    for target in LINK.findall(without_code(text)):
        if re.match(r"^(https?:|mailto:|#)", target):
            continue
        path = folder / target.split("#")[0]
        if not path.exists():
            problems.append(f"broken link: {target}")
        elif ignored_by_git(path):
            problems.append(f"link to a file that is not in the repository: {target}")
    return problems


def wording_problems(text: str) -> list[str]:
    return [f"refers to an internal lesson: {m}" for m in re.findall(r"(?i)\blessons?\s*\d+\w*", text)]


def known_commands(root: Path) -> set[str]:
    main = (root / "hello_tokens" / "__main__.py").read_text(encoding="utf-8")
    return set(re.findall(r'add_parser\(\s*"([a-z_]+)"', main))


def command_problems(text: str, commands: set[str]) -> list[str]:
    return [f"unknown command: {c}" for c in COMMAND.findall(text) if c not in commands]


# --- the checker itself ---------------------------------------------------------------------------

SOURCE = """\
def add(a, b):
    total = a + b
    return total


def twice(x):
    return add(x, x)
""".splitlines()


def test_an_exact_excerpt_matches():
    assert excerpt_matches(["def add(a, b):", "    total = a + b", "    return total"], SOURCE)


def test_indentation_may_differ():
    assert excerpt_matches(["total = a + b", "return total"], SOURCE)


def test_elided_runs_must_appear_in_order():
    first, second = ["def add(a, b):", "total = a + b"], ["def twice(x):", "return add(x, x)"]
    assert excerpt_matches([*first, "...", *second], SOURCE)
    assert not excerpt_matches([*second, "...", *first], SOURCE)


def test_a_changed_line_fails():
    assert not excerpt_matches(["def add(a, b):", "    total = b + a"], SOURCE)


def test_lines_must_be_consecutive_within_a_run():
    assert not excerpt_matches(["def add(a, b):", "return total"], SOURCE)


def test_blank_lines_are_ignored_on_both_sides():
    assert excerpt_matches(["return total", "def twice(x):"], SOURCE)
    assert excerpt_matches(["total = a + b", "", "return total"], SOURCE)


def test_a_run_of_one_line_is_too_weak_to_count():
    assert not excerpt_matches(["return total"], SOURCE)
    assert not excerpt_matches(["def add(a, b):", "total = a + b", "...", "return add(x, x)"], SOURCE)


def test_a_decorated_definition_must_show_its_decorator():
    source = ["@torch.no_grad()", "def step(model):", "    return model()"]
    assert not excerpt_matches(["def step(model):", "return model()"], source)
    assert excerpt_matches(["@torch.no_grad()", "def step(model):"], source)


def test_internal_lesson_numbers_are_reported():
    assert wording_problems("See lesson 17b. The lessons learned were few.") == [
        "refers to an internal lesson: lesson 17b"
    ]


def test_an_empty_excerpt_fails():
    assert not excerpt_matches(["", "..."], SOURCE)


def test_unmarked_python_blocks_are_reported(tmp_path):
    text = "Some prose.\n```python\nx = 1\n```\n"
    assert excerpt_problems(text, tmp_path) == ["python block 1 has no `from:` or `illustration` marker"]


def test_marked_blocks_are_checked_against_their_file(tmp_path):
    (tmp_path / "m.py").write_text("\n".join(SOURCE), encoding="utf-8")
    good = "<!-- from: m.py -->\n```python\ndef twice(x):\nreturn add(x, x)\n```\n"
    bad = "<!-- from: m.py -->\n```python\ndef twice(x):\nreturn add(x, 1)\n```\n"
    drawn = "<!-- illustration -->\n```python\nanything()\n```\n"
    assert excerpt_problems(good + drawn, tmp_path) == []
    assert excerpt_problems(bad, tmp_path) == ["python block 1 does not match m.py"]


def test_links_and_commands(tmp_path):
    (tmp_path / "there.md").write_text("", encoding="utf-8")
    text = "[a](there.md#part) [b](missing.md) [c](https://example.com) [d](#top)\n"
    assert link_problems(text, tmp_path) == ["broken link: missing.md"]
    assert command_problems("python -m hello_tokens train and python -m hello_tokens fly", {"train"}) == [
        "unknown command: fly"
    ]


def test_the_real_commands_are_found():
    assert {"train", "write", "eval"} <= known_commands(ROOT)


# --- the book ---------------------------------------------------------------------------------------

CHAPTERS = sorted(BOOK.glob("*.md")) if BOOK.is_dir() else []


@pytest.mark.parametrize("chapter", CHAPTERS, ids=lambda p: p.name)
def test_the_book_is_true_to_the_code(chapter):
    text = chapter.read_text(encoding="utf-8")
    problems = (
        excerpt_problems(text, ROOT)
        + link_problems(text, chapter.parent)
        + command_problems(text, known_commands(ROOT))
        + wording_problems(text)
    )
    assert problems == []
