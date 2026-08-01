"""Shared spaCy-backed tokenizer and POS tagger.

Every spaCy language in LTC (de, es, fr, it, ...) needs the same three things:
split a sentence into tokens, map spaCy's universal POS tags onto the four LTC
content-word codes, and do that in batch fast enough for a full corpus run.
This module holds that logic once so a new language only has to name its model.
"""

from __future__ import annotations

import os

from ltc.constants import UD_POS_TO_LTC_CODE
from ltc.env import get_bool_env, get_positive_int_env

DEFAULT_POS_MAP = UD_POS_TO_LTC_CODE

# The LTC normalizers do their own lemmatization, and no pair uses the
# dependency parse, so both pipes are dead weight during a corpus run.
DEFAULT_DISABLED_PIPES = ("parser", "lemmatizer")


class SpacyPosTagger:
    """Lazily-loaded spaCy pipeline that yields (tokens, pos_codes)."""

    def __init__(
        self,
        language,
        default_model,
        pos_map=None,
        disable=DEFAULT_DISABLED_PIPES,
    ):
        self.language = language
        self.model_env_var = f"LTC_{language.upper()}_SPACY_MODEL"
        self.model_name = os.environ.get(self.model_env_var, default_model)
        self.pos_map = dict(DEFAULT_POS_MAP if pos_map is None else pos_map)
        self.disable = list(disable)
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy

            # Defaults on: prefer_gpu() is a no-op when no GPU is present.
            if get_bool_env("LTC_SPACY_PREFER_GPU", True):
                spacy.prefer_gpu()
            self._nlp = spacy.load(self.model_name, disable=self.disable)
        return self._nlp

    @property
    def pipe_kwargs(self):
        batch_size = get_positive_int_env("LTC_SPACY_PIPE_BATCH_SIZE")
        return {} if batch_size is None else {"batch_size": batch_size}

    def _doc_to_tokens_and_tags(self, doc):
        tokens = []
        pos_codes = []
        for token in doc:
            tokens.append(token.text)
            pos_codes.append(self.pos_map.get(token.pos_, ""))
        return tokens, pos_codes

    def analyze(self, sentence):
        return self._doc_to_tokens_and_tags(self.nlp(sentence))

    def analyze_batch(self, sentences):
        if get_bool_env("LTC_BUCKETING", False) and len(sentences) > 1:
            return self._analyze_bucketed(sentences)

        tokens_l = []
        pos_codes_l = []
        for doc in self.nlp.pipe(sentences, **self.pipe_kwargs):
            tokens, pos_codes = self._doc_to_tokens_and_tags(doc)
            tokens_l.append(tokens)
            pos_codes_l.append(pos_codes)
        return tokens_l, pos_codes_l

    def _analyze_bucketed(self, sentences):
        # Feeding spaCy similar-length sentences together cuts padding waste.
        # Character length approximates token count well enough, and the sort
        # itself is n log n against a transformer forward pass.
        order = sorted(
            range(len(sentences)), key=lambda i: len(sentences[i]), reverse=True
        )
        sorted_sentences = [sentences[i] for i in order]
        tokens_l = [None] * len(sentences)
        pos_codes_l = [None] * len(sentences)
        for position, doc in enumerate(
            self.nlp.pipe(sorted_sentences, **self.pipe_kwargs)
        ):
            tokens, pos_codes = self._doc_to_tokens_and_tags(doc)
            original_index = order[position]
            tokens_l[original_index] = tokens
            pos_codes_l[original_index] = pos_codes
        return tokens_l, pos_codes_l

    def runtime_check(self):
        next(self.nlp.pipe(["Smoke test."], **self.pipe_kwargs))

    def runtime_metadata(self):
        return {
            "note": f"spaCy model: {self.model_name}",
            "spacy_model": self.model_name,
            "spacy_model_env_var": self.model_env_var,
            "disabled_pipes": list(self.disable),
        }


def build_spacy_morphological(language, default_model, pos_map=None):
    """Return the `({la}_morphological, {la}_morphological_batch, ...)` bundle.

    Language modules under `src/morphological/` use this so each one stays a
    thin declaration of *which* spaCy model to use.
    """
    tagger = SpacyPosTagger(language, default_model, pos_map=pos_map)
    return (
        tagger.analyze,
        tagger.analyze_batch,
        tagger.runtime_check,
        tagger.runtime_metadata,
        tagger,
    )
