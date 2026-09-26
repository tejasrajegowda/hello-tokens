"""The one way changes get committed to this repo.

    python scripts/commit.py "Plain description of what changed"

1. runs the tests, so a broken state is never committed
2. stages everything the ignore rules allow
3. runs the privacy guard on the message and on the staged files
4. commits (the git hooks run the same guard again)

It never pushes. Pushing is done by hand, after reviewing what was committed.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*cmd: str) -> None:
    if subprocess.run(cmd).returncode != 0:
        print(f"\ncommit: stopped, {' '.join(cmd)} failed.", file=sys.stderr)
        sys.exit(1)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    message = " ".join(sys.argv[1:]).strip()
    if not message:
        print('usage: python scripts/commit.py "what changed"', file=sys.stderr)
        sys.exit(2)

    root = Path(git("rev-parse", "--show-toplevel"))
    os.chdir(root)
    guard = str(Path("scripts") / "guard.py")

    if any(Path("tests").rglob("test_*.py")):
        run("uv", "run", "pytest", "-q")
    else:
        print("commit: no tests yet, skipping the test run.")

    run("git", "add", "-A")
    if not git("diff", "--cached", "--name-only"):
        print("commit: nothing to commit.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        msg_file = Path(tmp) / "MSG"
        msg_file.write_text(message + "\n", encoding="utf-8")
        run(sys.executable, guard, "message", str(msg_file))
        run(sys.executable, guard, "staged")
        run("git", "commit", "--file", str(msg_file))

    print("\ncommit: done. Nothing was pushed. Review it with:")
    print("  git show --stat HEAD")


if __name__ == "__main__":
    main()
