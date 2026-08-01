"""Structural checks on the per-pair alignment configuration.

These run without any language runtime installed, so a contributor adding a pair
gets fast feedback before touching models or corpora.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ltc.config.pairs import PAIR_CONFIGS, get_pair_config  # noqa: E402
from ltc.constants import LTC_CODE_TO_PRIMARY_UD_POS, UD_POS_TO_LTC_CODE  # noqa: E402
from ltc.diagnostics import MODULE_GROUPS  # noqa: E402

ALIGNMENT_DIR = ROOT / "src" / "alignment"


class PairConfigRegistryTest(unittest.TestCase):
    def test_pair_names_are_alphabetical(self):
        for pair in PAIR_CONFIGS:
            la1, la2 = pair.split("_")
            self.assertLess(la1, la2, f"{pair} is not in alphabetical order")

    def test_config_key_matches_declared_pair(self):
        for pair, config in PAIR_CONFIGS.items():
            self.assertEqual(pair, config.pair)

    def test_every_config_has_an_alignment_module_and_exceptions_file(self):
        for pair in PAIR_CONFIGS:
            with self.subTest(pair=pair):
                self.assertTrue((ALIGNMENT_DIR / pair / "__init__.py").is_file())
                self.assertTrue((ALIGNMENT_DIR / pair / "exceptions.csv").is_file())

    def test_every_configured_pair_is_registered_for_diagnostics(self):
        registered = set(MODULE_GROUPS["alignment"])
        for pair in PAIR_CONFIGS:
            self.assertIn(f"alignment.{pair}", registered)

    def test_language_modules_are_registered_for_diagnostics(self):
        normalizers = set(MODULE_GROUPS["normalizer"])
        morphologicals = set(MODULE_GROUPS["morphological"])
        for pair in PAIR_CONFIGS:
            for la in pair.split("_"):
                with self.subTest(language=la):
                    self.assertIn(f"normalizer.{la}_normalizer", normalizers)
                    self.assertIn(f"morphological.{la}_morphological", morphologicals)

    def test_ignore_rules_are_well_formed(self):
        for pair, config in PAIR_CONFIGS.items():
            for side, rules in (
                ("src", config.src_ignore_rules),
                ("trg", config.trg_ignore_rules),
            ):
                for rule in rules:
                    with self.subTest(pair=pair, side=side, rule=rule.words):
                        self.assertIsInstance(rule.words, tuple)
                        self.assertTrue(rule.words, "rule has no marker words")
                        self.assertIsInstance(rule.offset, int)
                        if rule.require_next_pos is not None:
                            self.assertIn(
                                rule.require_next_pos, UD_POS_TO_LTC_CODE.values()
                            )

    def test_word_separator_is_empty_only_for_unspaced_scripts(self):
        unspaced = {"ja", "zh"}
        for pair, config in PAIR_CONFIGS.items():
            with self.subTest(pair=pair):
                if config.trg_word_sep == "":
                    self.assertIn(config.la2, unspaced)
                else:
                    self.assertEqual(config.trg_word_sep, " ")

    def test_get_pair_config_reports_known_pairs_on_miss(self):
        with self.assertRaises(KeyError) as ctx:
            get_pair_config("xx_yy")
        self.assertIn("de_en", str(ctx.exception))


class PosMappingTest(unittest.TestCase):
    def test_primary_ud_pos_round_trips_to_its_ltc_code(self):
        for ltc_code, ud_pos in LTC_CODE_TO_PRIMARY_UD_POS.items():
            self.assertEqual(UD_POS_TO_LTC_CODE[ud_pos], ltc_code)

    def test_every_ltc_content_code_has_a_primary_ud_pos(self):
        self.assertEqual(
            sorted(LTC_CODE_TO_PRIMARY_UD_POS), sorted(set(UD_POS_TO_LTC_CODE.values()))
        )


if __name__ == "__main__":
    unittest.main()
