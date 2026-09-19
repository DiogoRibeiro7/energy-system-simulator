"""Guard documentation math against constructs that GitHub does not render.

GitHub renders ``$...$``, ``$`...`$`` and fenced ``math`` blocks, but it runs
Markdown over the contents of plain ``$...$`` first. These checks are offline
approximations of the failures that were found by rendering the docs through
GitHub's Markdown API.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INLINE_MATH = re.compile(r"(?<![\\$`\w])\$(?![\s$`])([^$\n]+?)(?<![\s`])\$(?![\d$])")
INLINE_CODE = re.compile(r"`[^`\n]*`")
FORBIDDEN_IN_INLINE = {
    "}_": "an underscore after '}' starts Markdown emphasis; put the subscript "
    "first (x_{b}^{a}) or use the $`...`$ form",
    "\\{": "Markdown consumes the backslash; use \\lbrace",
    "\\}": "Markdown consumes the backslash; use \\rbrace",
    "<": "GitHub double-escapes '<' in inline math; use \\lt",
    ">": "GitHub double-escapes '>' in inline math; use \\gt",
}


def _tracked_markdown() -> list[Path]:
    listing = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [ROOT / line for line in listing.stdout.splitlines() if line]


def find_math_problems(text: str) -> list[str]:
    """Return one message per construct that GitHub would not render as math."""
    problems: list[str] = []
    in_fence = False
    for number, raw_line in enumerate(text.splitlines(), start=1):
        if raw_line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = INLINE_CODE.sub("", raw_line)
        if "\\(" in line or "\\)" in line or line.strip() in {"\\[", "\\]"}:
            problems.append(f"{number}: \\( \\) and \\[ \\] delimiters are not rendered by GitHub")
        for match in INLINE_MATH.finditer(line):
            for token, reason in FORBIDDEN_IN_INLINE.items():
                if token in match.group(1):
                    problems.append(f"{number}: ${match.group(1)}$ contains {token!r}: {reason}")
    return problems


@pytest.mark.parametrize("path", _tracked_markdown(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_markdown_math_renders_on_github(path: Path) -> None:
    problems = find_math_problems(path.read_text(encoding="utf-8"))
    assert not problems, f"{path.relative_to(ROOT)}:\n" + "\n".join(problems)


@pytest.mark.parametrize(
    ("text", "expected_fragment"),
    [
        ("Periods \\(t=1,\\ldots,T\\).", "delimiters are not rendered"),
        ("\\[\nx = 1\n\\]", "delimiters are not rendered"),
        ("Energy $s^{ev}_{j,t}$ is tracked.", "'}_'"),
        ("Binary $y\\in\\{0,1\\}$ variables.", "'\\\\{'"),
        ("When $a_t>0$ holds.", "'>'"),
        ("When $a_t<0$ holds.", "'<'"),
    ],
)
def test_guard_flags_unrendered_math(text: str, expected_fragment: str) -> None:
    problems = find_math_problems(text)
    assert any(expected_fragment in problem for problem in problems), problems


@pytest.mark.parametrize(
    "text",
    [
        "Energy $s_{j,t}^{ev}$ and set $\\mathcal W_m$ are fine.",
        "Binary $y\\in\\lbrace 0,1\\rbrace$ and $a_t \\gt 0$ are fine.",
        "The literal form $`\\mathrm{COP}_{j,t}`$ is exempt.",
        "```math\nx^{a}_{b} < y \\{z\\}\n```",
        "```bash\necho $HOME > out_$USER\n```",
        "Inline code `\\(not math\\)` and prices of $5 or $10 are ignored.",
    ],
)
def test_guard_accepts_rendered_math_and_non_math(text: str) -> None:
    assert find_math_problems(text) == []
