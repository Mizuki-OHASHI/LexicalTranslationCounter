#!/usr/bin/env python3
"""Bootstrap a smoke project for a language pair from a raw parallel TSV.

Public parallel corpora (ParaCrawl, OPUS, ...) ship as `la1<TAB>la2` text with
no ids. LTC wants a 5-column corpus CSV plus one wordlist per language and part
of speech. This turns the former into the latter so a brand-new pair has
something real to run against before any production data exists.

The wordlists are derived from the sampled corpus itself: every sentence is run
through the pair's morphological and normalizer functions, and normalized forms
that clear a frequency threshold become wordlist entries. That is the same
frequency-based idea as `src/add_wordlist.py`, scoped to a smoke-sized sample.

    python3 scripts/build_smoke_project.py \\
        --pair en_ru --input ~/Downloads/en-ru.txt \\
        --scan 200000 --rows 200
"""

from __future__ import annotations

import argparse
import csv
import importlib
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ltc.constants import POS_TAG_TO_NAME  # noqa: E402

POS_NAMES = tuple(POS_TAG_TO_NAME.values())
URL_RE = re.compile(r"https?://|www\.")
# Rough script checks so an obviously mismatched line can be dropped early.
SCRIPT_RANGES = {
    "en": re.compile(r"[A-Za-z]"),
    "de": re.compile(r"[A-Za-zÄÖÜäöüß]"),
    "fr": re.compile(r"[A-Za-zÀ-ÿ]"),
    "es": re.compile(r"[A-Za-zÁ-ÿ]"),
    "it": re.compile(r"[A-Za-zÀ-ÿ]"),
    "ru": re.compile(r"[А-Яа-яЁё]"),
    "ja": re.compile(r"[぀-ヿ一-鿿]"),
    "zh": re.compile(r"[一-鿿]"),
    "ko": re.compile(r"[가-힯]"),
}


def looks_usable(text_1, text_2, la1, la2, min_chars, max_chars, max_ratio):
    for text in (text_1, text_2):
        if not (min_chars <= len(text) <= max_chars):
            return False
        if URL_RE.search(text):
            return False
        if text.count("@"):
            # `@` is stripped as a corpus marker downstream; avoid the ambiguity.
            return False

    shorter, longer = sorted((len(text_1), len(text_2)))
    if longer > shorter * max_ratio:
        return False

    for text, la in ((text_1, la1), (text_2, la2)):
        pattern = SCRIPT_RANGES.get(la)
        if pattern is not None and not pattern.search(text):
            return False
    return True


def sample_rows(path, la1, la2, scan, rows, min_chars, max_chars, max_ratio):
    """Take an evenly-spread sample so the slice is not all from one crawl block."""
    kept = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f):
            if line_number >= scan:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            text_1, text_2 = (part.strip() for part in parts)
            if looks_usable(text_1, text_2, la1, la2, min_chars, max_chars, max_ratio):
                kept.append((text_1, text_2))

    if not kept:
        raise SystemExit(f"no usable rows found in the first {scan} lines of {path}")
    if len(kept) <= rows:
        return kept
    step = len(kept) / rows
    return [kept[int(index * step)] for index in range(rows)]


def load_language_functions(la):
    morphological = getattr(
        importlib.import_module(f"morphological.{la}_morphological"),
        f"{la}_morphological_batch",
    )
    normalizer = getattr(
        importlib.import_module(f"normalizer.{la}_normalizer"),
        f"{la}_normalizer",
    )
    return morphological, normalizer


def is_wordlist_candidate(normalized):
    """Reject punctuation, clitics and single characters before they become entries."""
    if not normalized or not normalized.strip():
        return False
    for part in normalized.split():
        if len(part) < 2 or not part.isalpha():
            return False
    return True


def build_wordlist(sentences, morphological_batch, normalizer, min_count):
    """Count normalized forms per POS across the sample."""
    counts = {pos_name: Counter() for pos_name in POS_NAMES}
    tokens_l, pos_codes_l = morphological_batch(sentences)
    for tokens, pos_codes in zip(tokens_l, pos_codes_l):
        for token, pos_code in zip(tokens, pos_codes):
            if pos_code not in POS_TAG_TO_NAME:
                continue
            normalized = normalizer(token, pos_code, {}, True)
            if not is_wordlist_candidate(normalized):
                continue
            counts[POS_TAG_TO_NAME[pos_code]][normalized] += 1

    wordlists = {}
    for pos_name, counter in counts.items():
        words = sorted(word for word, count in counter.items() if count >= min_count)
        wordlists[pos_name] = {word: index + 1 for index, word in enumerate(words)}
    return wordlists


def write_corpus(path, pairs):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for index, (text_1, text_2) in enumerate(pairs):
            writer.writerow([index, text_1, text_2, "{}", "False"])


def write_wordlist(path, wordlist):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for word, word_id in wordlist.items():
            writer.writerow([word_id, word, "False"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, help="e.g. en_ru")
    parser.add_argument("--input", required=True, type=Path, help="raw la1<TAB>la2 TSV")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scan", type=int, default=200_000, help="lines to read")
    parser.add_argument("--rows", type=int, default=200, help="corpus rows to keep")
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument("--max-length-ratio", type=float, default=2.0)
    parser.add_argument("--min-word-count", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    la1, la2 = args.pair.split("_")
    output_dir = args.output_dir or ROOT / "projects" / "smoke" / args.pair / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = sample_rows(
        args.input.expanduser(),
        la1,
        la2,
        args.scan,
        args.rows,
        args.min_chars,
        args.max_chars,
        args.max_length_ratio,
    )
    print(f"sampled {len(pairs)} rows from the first {args.scan} lines")

    corpus_path = output_dir / f"corpus_{args.pair}.csv"
    write_corpus(corpus_path, pairs)
    print(f"wrote {corpus_path.relative_to(ROOT)}")

    for la, column in ((la1, 0), (la2, 1)):
        morphological_batch, normalizer = load_language_functions(la)
        sentences = [pair[column] for pair in pairs]
        wordlists = build_wordlist(
            sentences, morphological_batch, normalizer, args.min_word_count
        )
        for pos_name, wordlist in wordlists.items():
            path = output_dir / f"wordlist_{la}_{pos_name}.csv"
            write_wordlist(path, wordlist)
        sizes = ", ".join(
            f"{pos_name}={len(wordlists[pos_name])}" for pos_name in POS_NAMES
        )
        print(f"  {la}: {sizes}")


if __name__ == "__main__":
    raise SystemExit(main())
