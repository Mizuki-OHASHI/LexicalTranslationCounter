"""Russian tokenization, POS tagging and lemmatization via Natasha.

Natasha covers both LTC language contracts with one dependency: Razdel
tokenizes, Slovnet's `NewsMorphTagger` emits Universal Dependencies POS tags,
and `MorphVocab` lemmatizes. Inference is CPU-only NumPy and the models are
~30 MB, so no GPU setup is involved.

Caveat worth knowing before trusting the output: the Slovnet models are trained
on news text. A web-crawled corpus like ParaCrawl is a different domain, so
tagging quality there is not the number the upstream benchmarks report.
"""

from __future__ import annotations

from ltc.constants import LTC_CODE_TO_PRIMARY_UD_POS, UD_POS_TO_LTC_CODE

# The normalizer is called once per candidate word combination per POS tag, so
# the same handful of forms recur constantly within a corpus.
LEMMA_CACHE_SIZE = 100_000


class NatashaRussianBackend:
    """Lazily-loaded Natasha pipeline shared by the ru morphological/normalizer."""

    language = "ru"

    def __init__(self, pos_map=None):
        self.pos_map = dict(UD_POS_TO_LTC_CODE if pos_map is None else pos_map)
        self._segmenter = None
        self._morph_tagger = None
        self._morph_vocab = None
        self._lemma_cache = {}

    def _ensure_loaded(self):
        if self._segmenter is not None:
            return
        from natasha import MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter

        self._segmenter = Segmenter()
        self._morph_tagger = NewsMorphTagger(NewsEmbedding())
        self._morph_vocab = MorphVocab()

    @property
    def morph_vocab(self):
        self._ensure_loaded()
        return self._morph_vocab

    def _tagged_doc(self, sentence):
        from natasha import Doc

        self._ensure_loaded()
        doc = Doc(sentence)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)
        return doc

    # -- morphological contract -------------------------------------------

    def analyze(self, sentence):
        doc = self._tagged_doc(sentence)
        tokens = [token.text for token in doc.tokens]
        pos_codes = [self.pos_map.get(token.pos, "") for token in doc.tokens]
        return tokens, pos_codes

    def analyze_batch(self, sentences):
        tokens_l = []
        pos_codes_l = []
        for sentence in sentences:
            tokens, pos_codes = self.analyze(sentence)
            tokens_l.append(tokens)
            pos_codes_l.append(pos_codes)
        return tokens_l, pos_codes_l

    # -- normalizer contract ----------------------------------------------

    def tokenize(self, text):
        from razdel import tokenize

        return [token.text for token in tokenize(text)]

    def _lemmatize_token(self, token, ud_pos):
        """Lemmatize a single token, disambiguating with its morphological features.

        `MorphVocab.lemmatize` picks between competing pymorphy parses using the
        feature dict it is handed. Passing `{}` leaves it guessing, which is how
        `снимок` came back as `снимка`. Tagging the token first — with its
        original casing, which the tagger uses as a cue — supplies the features
        that Natasha's own `token.lemmatize(morph_vocab)` would have had.
        """
        cache_key = (token, ud_pos)
        cached = self._lemma_cache.get(cache_key)
        if cached is not None:
            return cached

        tagged = self._tagged_doc(token).tokens
        feats = tagged[0].feats if tagged else {}
        lemma = self.morph_vocab.lemmatize(token, ud_pos, feats or {}).lower()

        if len(self._lemma_cache) < LEMMA_CACHE_SIZE:
            self._lemma_cache[cache_key] = lemma
        return lemma

    def lemmatize_word(self, word, ltc_pos_tag):
        """Lemmatize one whitespace-joined word or phrase to its LTC form.

        Multi-token phrases are lemmatized token by token under a single POS
        tag, which does not respect Russian agreement — see
        `documents/en-ru/Handoff_ja.md`.
        """
        ud_pos = LTC_CODE_TO_PRIMARY_UD_POS.get(ltc_pos_tag)
        if ud_pos is None:
            raise ValueError(
                f"unknown LTC pos tag {ltc_pos_tag!r}; expected one of "
                f"{sorted(LTC_CODE_TO_PRIMARY_UD_POS)}"
            )
        return " ".join(
            self._lemmatize_token(token, ud_pos) for token in self.tokenize(word)
        )

    # -- diagnostics -------------------------------------------------------

    def runtime_check(self):
        tokens, pos_codes = self.analyze("Кошки быстро бегают.")
        if not tokens or len(tokens) != len(pos_codes):
            raise RuntimeError(
                "natasha returned an inconsistent tokenization for the smoke "
                f"sentence: {tokens!r} / {pos_codes!r}"
            )
        if self.lemmatize_word("кошки", "n") != "кошка":
            raise RuntimeError(
                "natasha lemmatization smoke check failed: "
                f"кошки -> {self.lemmatize_word('кошки', 'n')!r}, expected 'кошка'"
            )

    def runtime_metadata(self):
        from importlib.metadata import PackageNotFoundError, version

        try:
            natasha_version = version("natasha")
        except PackageNotFoundError:  # pragma: no cover - editable/source installs
            natasha_version = "unknown"

        return {
            "note": f"natasha {natasha_version} (razdel + slovnet news models)",
            "natasha_version": natasha_version,
            "domain_caveat": "slovnet models are trained on news text",
        }


BACKEND = NatashaRussianBackend()
