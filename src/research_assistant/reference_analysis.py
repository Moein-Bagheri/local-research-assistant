from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import fitz


YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b", re.IGNORECASE)
REFERENCES_PATTERN = re.compile(r"\breferences\s*$", re.IGNORECASE | re.MULTILINE)


def extract_reference_year_counts(pdf_path: Path) -> Counter[int]:
    """Count publication years in the bibliography using a documented heuristic.

    The parser starts at the first page containing a standalone References heading,
    then counts four-digit years. A suffix such as 2020a is treated as 2020.
    """

    counts: Counter[int] = Counter()
    in_references = False
    with fitz.open(pdf_path) as document:
        earliest_bibliography_page = max(0, int(len(document) * 0.45))
        for page_index, page in enumerate(document):
            text = page.get_text("text", sort=True)
            if (
                not in_references
                and page_index >= earliest_bibliography_page
                and REFERENCES_PATTERN.search(text)
            ):
                in_references = True
                match = REFERENCES_PATTERN.search(text)
                assert match is not None
                text = text[match.end() :]
            if not in_references:
                continue
            for match in YEAR_PATTERN.findall(text):
                # findall returns the complete match for this non-capturing pattern.
                year = int(re.match(r"\d{4}", match).group(0))
                if 1950 <= year <= 2030:
                    counts[year] += 1
    return counts
