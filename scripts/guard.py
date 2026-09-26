"""Privacy and identity checks for commits and pushes.

    python scripts/guard.py staged            check the staged files (pre-commit)
    python scripts/guard.py message <file>    check a commit message (commit-msg)
    python scripts/guard.py outgoing          check commits about to be pushed (pre-push)

Every rule is read from .local/guard.json, which is never committed: the words that must not
appear, the paths that must not be committed, and the expected author identity. If that file is
missing, every check fails. The guard fails closed, never open.
"""

import json
import re
import subprocess
import sys
from pathlib import Path


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, check=True)
    return result.stdout.decode("utf-8", errors="replace").strip()


def git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def fail(problems: list[str]) -> None:
    print("\nguard: refused.\n" + "\n".join(f"  - {p}" for p in problems) + "\n", file=sys.stderr)
    sys.exit(1)


ROOT = Path(git("rev-parse", "--show-toplevel"))
CONFIG = ROOT / ".local" / "guard.json"
if not CONFIG.exists():
    fail([".local/guard.json is missing, so nothing can be checked. Refusing (fail closed)."])

cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
WORDS = [str(w).lower() for w in cfg.get("forbiddenWords", []) if w]
PATHS = [str(p) for p in cfg.get("forbiddenPaths", []) if p]

# An absolute path to a user folder reveals the machine's user and folder names.
ABSOLUTE = re.compile(r"\b[a-z]:[\\/]+users[\\/]", re.IGNORECASE)
# A trailer line such as a co-author credit.
TRAILER = re.compile(r"^\s*[a-z-]+-by:", re.IGNORECASE | re.MULTILINE)


def word_matcher(word: str):
    # Short entries (4 characters or fewer) match only as whole words, so a run of letters
    # inside a hash cannot trip them. Longer entries match anywhere, including inside other words.
    if len(word) <= 4:
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])")
        return pattern.search
    return lambda text: word in text


MATCHERS = [word_matcher(w) for w in WORDS]


def scan_text(label: str, text: str) -> list[str]:
    lower = text.lower()
    problems = [
        f"{label} contains a forbidden word (entry {i + 1} in guard.json)"
        for i, matches in enumerate(MATCHERS)
        if matches(lower)
    ]
    if ABSOLUTE.search(text):
        problems.append(f"{label} contains an absolute path to a user folder")
    return problems


def is_forbidden_path(path: str) -> bool:
    return any(
        path == p or path.startswith(p) or path.lower().endswith(p.lower()) for p in PATHS
    )


def check_identity() -> list[str]:
    problems = []
    if cfg.get("expectedEmail") and git("config", "user.email") != cfg["expectedEmail"]:
        problems.append("the commit author email is not the expected personal identity")
    if cfg.get("expectedName") and git("config", "user.name") != cfg["expectedName"]:
        problems.append("the commit author name is not the expected personal identity")
    return problems


def check_staged() -> None:
    problems = check_identity()
    files = [f for f in git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines() if f]
    for f in files:
        if is_forbidden_path(f):
            problems.append(f"{f} is a local-only file")
        problems += scan_text(f"the file name {f}", f)
        blob = git_bytes("show", f":{f}")
        if b"\x00" in blob:
            continue  # binary file: path checks only
        problems += scan_text(f, blob.decode("utf-8", errors="replace"))
    if problems:
        fail(problems)
    print(f"guard: {len(files)} staged file(s) checked, clean.")


def check_message(path: str) -> None:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    text = "\n".join(line for line in lines if not line.startswith("#"))
    problems = scan_text("the commit message", text)
    if TRAILER.search(text):
        problems.append("the commit message has a trailer line (for example a co-author line)")
    if problems:
        fail(problems)


def check_outgoing() -> None:
    problems = []
    for line in sys.stdin.read().strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        local_sha, remote_sha = parts[1], parts[3]
        if re.fullmatch(r"0+", local_sha):
            continue  # deleting a ref: nothing to scan
        new = re.fullmatch(r"0+", remote_sha) is not None
        commit_range = local_sha if new else f"{remote_sha}..{local_sha}"
        for sha in git("rev-list", commit_range).splitlines():
            message = git("log", "-1", "--format=%B", sha)
            problems += scan_text(f"commit {sha[:7]} message", message)
            if TRAILER.search(message):
                problems.append(f"commit {sha[:7]} has a trailer line")
            author = git("log", "-1", "--format=%ae", sha)
            if cfg.get("expectedEmail") and author != cfg["expectedEmail"]:
                problems.append(f"commit {sha[:7]} has an unexpected author email")
        for f in git("ls-tree", "-r", "--name-only", local_sha).splitlines():
            if is_forbidden_path(f):
                problems.append(f"{f} is a local-only file")
    if problems:
        fail(problems)
    print("guard: outgoing commits checked, clean.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "staged":
        check_staged()
    elif mode == "message" and len(sys.argv) > 2:
        check_message(sys.argv[2])
    elif mode == "outgoing":
        check_outgoing()
    else:
        fail([f'unknown mode "{mode}": use staged, message <file>, or outgoing'])
