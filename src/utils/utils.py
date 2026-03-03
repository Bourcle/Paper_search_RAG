import re
import json
from pathlib import Path
from typing import Optional, Any


def looks_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def parse_filter_from_question(question: str) -> tuple[str, Optional[dict[str, Any]]]:
    parsed_filter: dict[str, Any] = dict()
    striped_question = question.strip()

    # free-form JSON filter has the highest priority.
    matched = re.search(r"@filter\s*=\s*(\{.*\})\s*$", striped_question)
    if matched:
        try:
            parsed_filter = json.loads(matched.group(1))
            striped_question = striped_question[: matched.start()].strip()
            return striped_question, parsed_filter
        except json.JSONDecodeError:
            return striped_question, None

    token_patterns: list[tuple[str, str, Any]] = [
        ("filename", r"@file\s*=\s*([^\s]+)\s*$", lambda v: Path(v).stem),
        ("doc_id", r"@doc_id\s*=\s*([0-9a-fA-F-]+)\s*$", lambda v: v),
        ("page", r"@page\s*=\s*(\d+)\s*$", int),
    ]

    # Supports mixed ordering like: @page=2 @file=paper.pdf or @file=... @doc_id=...
    while True:
        changed = False
        for key, pattern, caster in token_patterns:
            matched = re.search(pattern, striped_question)
            if not matched:
                continue
            parsed_filter[key] = {"$eq": caster(matched.group(1).strip())}
            striped_question = striped_question[: matched.start()].strip()
            changed = True
            break
        if not changed:
            break

    return striped_question, (parsed_filter or None)
