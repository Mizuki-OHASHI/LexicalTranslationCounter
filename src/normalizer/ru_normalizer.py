"""Russian normalizer.

Contract: `ru_normalizer(word, pos_tag, wordlist, test=False)` returns
`(id, word_normalized)`, or just `word_normalized` when `test=True`.
"""

import csv
import os

from ltc.backends.linguistic.natasha_ru import BACKEND
from ltc.constants import POS_TAG_TO_NAME

BASE = os.path.dirname(os.path.abspath(__file__))


# The placeholder row the other languages ship in these files. It is data to
# csv.reader, so it has to be filtered out rather than treated as a header.
PLACEHOLDER_ROW = ("original_word***", "view_word***")


def load_exception_table(language, pos_tag):
    """Read `{pos_tag}_normalize.csv` as a flat `original -> normalized` map.

    NOTE: the other languages load these tables via
    `pandas.read_csv(..., index_col=0).to_dict()`, which yields
    `{1: {original: normalized}}` and therefore never matches a lookup — their
    exception tables have never taken effect. This reads them as intended.
    """
    path = os.path.join(BASE, "normalize_data", language, f"{pos_tag}_normalize.csv")
    table = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2 or tuple(row[:2]) == PLACEHOLDER_ROW:
                continue
            table[row[0]] = row[1]
    return table


EXCEPTIONS = {
    pos_tag: load_exception_table("ru", pos_tag) for pos_tag in POS_TAG_TO_NAME
}


def ru_normalizer(word, pos_tag, wordlist, test=False):
    """Normalize one Russian word or phrase.

    The exception table is consulted twice: once on the surface form, and again
    on whatever the lemmatizer produced. The second lookup is what makes the
    table usable for Russian at all -- a surface-keyed table would need every
    case and gender form of a word spelled out, whereas correcting a bad lemma
    takes one row. `больший -> большой` is the motivating example; see
    documents/en-ru/Handoff_ja.md.
    """
    if pos_tag not in EXCEPTIONS:
        raise ValueError(
            f"unknown LTC pos tag {pos_tag!r}; expected one of "
            f"{sorted(POS_TAG_TO_NAME)}"
        )
    exceptions = EXCEPTIONS[pos_tag]

    word_normalized = exceptions.get(word)
    if word_normalized is None:
        word_normalized = BACKEND.lemmatize_word(word, pos_tag)
        word_normalized = exceptions.get(word_normalized, word_normalized)

    if test:
        return word_normalized

    wordlist_key = f"ru_{POS_TAG_TO_NAME[pos_tag]}"
    word_id = wordlist[wordlist_key].get(word_normalized)
    return word_id, word_normalized


def runtime_check():
    BACKEND.runtime_check()


def runtime_metadata():
    return BACKEND.runtime_metadata()
