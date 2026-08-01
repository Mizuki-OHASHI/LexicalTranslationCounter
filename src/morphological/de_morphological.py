"""German tokenization and POS tagging.

Lemmatization is left to `normalizer.de_normalizer` (germalemma), so the spaCy
lemmatizer pipe stays disabled.
"""

from ltc.backends.linguistic.spacy_pos import build_spacy_morphological

(
    de_morphological,
    de_morphological_batch,
    runtime_check,
    runtime_metadata,
    TAGGER,
) = build_spacy_morphological("de", default_model="de_dep_news_trf")
