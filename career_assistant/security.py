"""Prompt-injection defence for untrusted retrieved text (job postings, etc.).

Job postings are written by third parties, so their text is DATA, never
instructions. We (1) strip hidden markup, (2) drop sentences that look like
instructions aimed at an AI, and (3) report what we removed so it can be shown
to the user and audited. This is one layer: tool results are also labelled as
untrusted for the model, and the output is validated against retrieved data.
"""
from __future__ import annotations

import re

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(instructions?|rules?|guidelines?|prompts?)\b",
    r"\bdisregard\b.{0,40}\b(instructions?|rules?|guidelines?|prompts?)\b",
    r"\b(forget|override)\b.{0,40}\b(instructions?|rules?|guidelines?)\b",
    r"\brecommend\b.{0,30}\b(this|the)\b.{0,15}\b(role|job|posting|course)\b",
    r"\b(to|for)\s+(everyone|everybody|all (employees|users))\b",
    r"\b(always|must|should)\s+(recommend|rank|suggest)\b",
    r"\bsystem prompt\b",
    r"\byou are now\b",
    r"\b(reveal|print|show)\b.{0,30}\b(prompt|instructions|other employees?)\b",
    r"\bas an? (ai|assistant|language model)\b",
]
_COMPILED = [re.compile(p, re.I) for p in INJECTION_PATTERNS]


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)


def sanitize_untrusted(text: str) -> tuple[str, list[str]]:
    """Return (clean_text, findings). Findings describe removed content."""
    findings: list[str] = []
    text = _ZERO_WIDTH.sub("", text)

    def _strip_comment(m: re.Match) -> str:
        inner = m.group(0)[4:-3].strip()
        findings.append(f"hidden HTML comment removed: {inner!r}")
        return " "

    text = _HTML_COMMENT.sub(_strip_comment, text)

    kept = []
    for segment in _SPLIT.split(text):
        seg = segment.strip()
        if not seg:
            continue
        if looks_like_injection(seg):
            findings.append(f"instruction-like text removed: {seg!r}")
            continue
        kept.append(seg)
    return " ".join(kept), findings
