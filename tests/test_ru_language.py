"""Tests for the Russian morphological and normalizer functions.

These pin the behaviour a contributor is most likely to change while tuning
Russian, and they document the known-bad cases as expected failures rather than
leaving them to be rediscovered. See documents/en-ru/Handoff_ja.md.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import natasha  # noqa: F401

    HAS_NATASHA = True
except ImportError:  # pragma: no cover - exercised on installs without the ru extra
    HAS_NATASHA = False


@unittest.skipUnless(HAS_NATASHA, "natasha is required for the Russian backend")
class RussianMorphologicalTest(unittest.TestCase):
    def setUp(self):
        from morphological.ru_morphological import (
            ru_morphological,
            ru_morphological_batch,
        )

        self.analyze = ru_morphological
        self.analyze_batch = ru_morphological_batch

    def test_tokens_and_pos_codes_line_up(self):
        tokens, pos_codes = self.analyze("Кошки быстро бегают по большому городу.")
        self.assertEqual(len(tokens), len(pos_codes))
        self.assertEqual(tokens[0], "Кошки")

    def test_content_words_get_ltc_codes(self):
        tokens, pos_codes = self.analyze("Кошки быстро бегают.")
        by_token = dict(zip(tokens, pos_codes))
        self.assertEqual(by_token["Кошки"], "n")
        self.assertEqual(by_token["быстро"], "r")
        self.assertEqual(by_token["бегают"], "v")

    def test_punctuation_and_particles_get_no_code(self):
        tokens, pos_codes = self.analyze("Не читал книгу.")
        by_token = dict(zip(tokens, pos_codes))
        # `не` is a PART, which is deliberately unmapped: the negation handling
        # lives in the alignment ignore rules, not in the POS codes.
        self.assertEqual(by_token["Не"], "")
        self.assertEqual(by_token["."], "")

    def test_batch_matches_single(self):
        sentences = ["Кошки бегают.", "Я читал книгу вчера."]
        tokens_l, pos_codes_l = self.analyze_batch(sentences)
        self.assertEqual(len(tokens_l), 2)
        for sentence, tokens, pos_codes in zip(sentences, tokens_l, pos_codes_l):
            self.assertEqual((tokens, pos_codes), self.analyze(sentence))


@unittest.skipUnless(HAS_NATASHA, "natasha is required for the Russian backend")
class RussianNormalizerTest(unittest.TestCase):
    def setUp(self):
        from normalizer.ru_normalizer import ru_normalizer

        self.normalize = ru_normalizer

    def lemma(self, word, pos_tag):
        return self.normalize(word, pos_tag, {}, True)

    def test_noun_case_forms_collapse_to_the_nominative(self):
        for word, expected in (
            ("книгу", "книга"),
            ("кошки", "кошка"),
            ("городах", "город"),
            ("человека", "человек"),
            ("людьми", "человек"),
        ):
            with self.subTest(word=word):
                self.assertEqual(self.lemma(word, "n"), expected)

    def test_verb_forms_collapse_to_the_infinitive(self):
        for word, expected in (
            ("читал", "читать"),
            ("бегают", "бегать"),
            ("сделал", "сделать"),
        ):
            with self.subTest(word=word):
                self.assertEqual(self.lemma(word, "v"), expected)

    def test_adjective_forms_collapse_to_the_masculine_nominative(self):
        for word, expected in (
            ("красивую", "красивый"),
            ("новые", "новый"),
            ("крупная", "крупный"),
            # Short form; the long form is the representative.
            ("опасен", "опасный"),
        ):
            with self.subTest(word=word):
                self.assertEqual(self.lemma(word, "a"), expected)

    def test_lemmatization_is_idempotent(self):
        # Regression: without the morphological features, MorphVocab returned
        # `снимка` for `снимок`, so normalizing twice flipped between the two.
        for word, pos_tag in (
            ("снимок", "n"),
            ("снимка", "n"),
            ("книгу", "n"),
            ("читал", "v"),
            ("красивую", "a"),
        ):
            with self.subTest(word=word):
                once = self.lemma(word, pos_tag)
                self.assertEqual(once, self.lemma(once, pos_tag))

    def test_capitalized_forms_normalize_like_lowercase_ones(self):
        self.assertEqual(self.lemma("Кошки", "n"), self.lemma("кошки", "n"))

    def test_returns_none_id_for_words_outside_the_wordlist(self):
        wordlist = {"ru_noun": {"книга": 42}}
        self.assertEqual(self.normalize("книгу", "n", wordlist), (42, "книга"))
        self.assertEqual(self.normalize("телефона", "n", wordlist), (None, "телефон"))

    def test_rejects_an_unknown_pos_tag(self):
        with self.assertRaises(ValueError):
            self.lemma("книга", "x")

    def test_exception_table_can_correct_a_bad_lemma(self):
        # The table is keyed on the surface form AND on the produced lemma, so a
        # single row fixes every inflected form. This is what makes it usable
        # for an inflected language.
        from normalizer.ru_normalizer import EXCEPTIONS

        EXCEPTIONS["a"]["больший"] = "большой"
        try:
            for word in ("большая", "большие", "Большая"):
                with self.subTest(word=word):
                    self.assertEqual(self.lemma(word, "a"), "большой")
        finally:
            del EXCEPTIONS["a"]["больший"]

    def test_shipped_exception_tables_are_empty_and_parse(self):
        # The placeholder row the repo ships must not become a real entry.
        from normalizer.ru_normalizer import EXCEPTIONS

        for pos_tag, table in EXCEPTIONS.items():
            with self.subTest(pos_tag=pos_tag):
                self.assertEqual(table, {})


@unittest.skipUnless(HAS_NATASHA, "natasha is required for the Russian backend")
class RussianKnownLimitationsTest(unittest.TestCase):
    """Documented upstream failures, asserted so a fix is noticed when it lands."""

    def lemma(self, word, pos_tag):
        from normalizer.ru_normalizer import ru_normalizer

        return ru_normalizer(word, pos_tag, {}, True)

    def test_positive_adjective_still_lemmatizes_to_the_comparative(self):
        # pymorphy2 scores `большой` and `больший` identically for these forms
        # and MorphVocab takes the first, even though Slovnet tags them
        # Degree=Pos. Whether to merge the two is a lexical decision.
        self.assertEqual(self.lemma("большая", "a"), "больший")

    def test_yo_is_folded_into_ye(self):
        # MorphVocab returns `е` where the input had `ё`. Consistent within the
        # pipeline, but an externally sourced wordlist spelled with `ё` will
        # never match unless it is normalized the same way first.
        for word, pos_tag in (("берёза", "n"), ("ёлка", "n"), ("полёт", "n")):
            with self.subTest(word=word):
                self.assertNotIn("ё", self.lemma(word, pos_tag))

    def test_unparseable_tokens_pass_through_unchanged(self):
        # No signal that normalization failed, so corpus junk reaches the
        # wordlist looking like a legitimate lemma.
        self.assertEqual(self.lemma("xhamster", "n"), "xhamster")


if __name__ == "__main__":
    unittest.main()
