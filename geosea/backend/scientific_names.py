"""Conservative name formatting shared by GeoSea API data paths.

This handles whitespace, underscores, case, and taxonomic author/rank suffixes.
It deliberately does not perform authoritative synonym resolution.
"""
from __future__ import annotations
import re


def canonical_binomial_name(value: str | None) -> str | None:
    if not value:
        return None
    tokens = re.sub(r"[_\s]+", " ", str(value).strip()).split()
    if len(tokens) < 2:
        return None
    genus_token = tokens[0].rstrip(".")
    epithet_token = tokens[1]
    if not re.fullmatch(r"[A-Za-z-]+", genus_token):
        return None
    # Lowercase epithets are standard; accept all-uppercase user input but
    # reject title-case author surnames (e.g. "Sargassum Bory").
    if not re.fullmatch(r"(?:[a-z][a-z-]*|[A-Z][A-Z-]*)", epithet_token):
        return None
    genus = genus_token[0].upper() + genus_token[1:].lower()
    epithet = epithet_token.lower()
    if len(genus) < 2 or len(epithet) < 2:
        return None
    return f"{genus} {epithet}"


def normalize_taxon_label(value: str | None) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "").replace("_", " ").strip())
    if not cleaned:
        return cleaned
    binomial = canonical_binomial_name(cleaned)
    if binomial:
        return binomial
    tokens = cleaned.split()
    genus = tokens[0]
    has_genus_shape = bool(re.fullmatch(r"[A-Za-z][A-Za-z-]+", genus))
    second_looks_like_authority = len(tokens) > 1 and (
        bool(re.match(r"^[A-Z](?:\.|[a-z])", tokens[1]))
        or tokens[1].startswith("(")
        or any(ch in tokens[1] for ch in ",0123456789")
    )
    if has_genus_shape and second_looks_like_authority:
        return genus[0].upper() + genus[1:].lower()
    return cleaned
