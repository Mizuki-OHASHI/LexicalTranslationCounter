"""Shared constants used by the modernized LTC core."""

POS_TAG_TO_NAME = {"n": "noun", "v": "verb", "a": "adj", "r": "adverb"}
POS_NAME_TO_TAG = {name: tag for tag, name in POS_TAG_TO_NAME.items()}
CONTENT_POS_NAMES = tuple(POS_NAME_TO_TAG.keys())

# Universal Dependencies POS tag -> LTC code. Shared by every backend that
# emits UD tags (spaCy for de/es/fr/it, Natasha for ru). Anything absent is a
# non-content word and gets an empty code.
UD_POS_TO_LTC_CODE = {
    "ADJ": "a",
    "NOUN": "n",
    "PRON": "n",
    "PROPN": "n",
    "ADV": "r",
    "CCONJ": "r",
    "SCONJ": "r",
    "VERB": "v",
}

# The reverse direction is lossy (n covers NOUN/PRON/PROPN), so pick the tag a
# lemmatizer should be asked for when all it gets is an LTC code.
LTC_CODE_TO_PRIMARY_UD_POS = {"a": "ADJ", "n": "NOUN", "r": "ADV", "v": "VERB"}
