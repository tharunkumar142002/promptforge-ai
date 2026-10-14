"""Transparent, deterministic keyword-coverage checks for evaluation examples."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

MAX_EVAL_ROWS = 5


@dataclass(frozen=True)
class EvalCase:
    task: str
    expected_keywords: tuple[str, ...]


def load_cases(csv_text: str) -> list[EvalCase]:
    """Parse a CSV containing input,expected_keywords; semicolon-separated keywords."""
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    if not reader.fieldnames or not {"input", "expected_keywords"}.issubset(reader.fieldnames):
        raise ValueError("CSV must contain columns named input and expected_keywords.")
    rows = []
    for line, record in enumerate(reader, start=2):
        task = (record.get("input") or "").strip()
        expected = tuple(
            k.strip() for k in (record.get("expected_keywords") or "").split(";") if k.strip()
        )
        if not task or not expected:
            raise ValueError(f"CSV row {line}: provide input and at least one expected keyword.")
        if len(task) > 4_000:
            raise ValueError(f"CSV row {line}: input exceeds 4000 characters.")
        if len(expected) > 12 or any(len(kw) > 100 for kw in expected):
            raise ValueError(f"CSV row {line}: too many or too-long expected keywords.")
        rows.append(EvalCase(task, expected))
        if len(rows) > MAX_EVAL_ROWS:
            raise ValueError(f"Evaluate at most {MAX_EVAL_ROWS} rows per run to protect API quotas.")
    if not rows:
        raise ValueError("The CSV has no evaluation cases.")
    return rows


def check_keywords(text: str, keywords: tuple[str, ...]) -> tuple[bool, list[str]]:
    """Coverage check, NOT a measure of overall factual quality or accuracy."""
    normalized = re.sub(r"\s+", " ", text).casefold()
    missing = [keyword for keyword in keywords
               if re.sub(r"\s+", " ", keyword).casefold() not in normalized]
    return not missing, missing
