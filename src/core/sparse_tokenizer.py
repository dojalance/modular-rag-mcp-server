"""Shared lexical tokenization for building and querying the BM25 index."""

import re
from collections import Counter

import jieba


_PUNCTUATION = re.compile(r"[\s\W]+")
_COMPOUND = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+(?![A-Za-z0-9_])")


def tokenize_sparse_text(text: str) -> list[str]:
    """Preserve compound identifiers alongside the original segment tokens."""
    tokens = [
        token for raw in jieba.lcut(text)
        if (token := raw.strip()) and not _PUNCTUATION.fullmatch(token)
    ]
    existing = Counter(token for token in tokens if "-" in token or "_" in token)
    for match in _COMPOUND.finditer(text):
        term = match.group()
        if existing[term]:
            existing[term] -= 1
        else:
            tokens.append(term)
    return tokens
