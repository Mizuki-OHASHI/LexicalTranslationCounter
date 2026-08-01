"""The shared alignment core must match the per-pair implementations it replaced.

`tests/data/alignment_golden.json` was produced by running the pre-refactor
`src/alignment/{pair}/__init__.py` modules on deterministic synthetic inputs
(see the header inside that file). Any diff here is a behaviour change.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover - exercised only on torch-less installs
    torch = None

from ltc.backends.alignment.awesome_pair import (  # noqa: E402
    AwesomePairAligner,
    PairLanguageComponents,
)
from ltc.config.pairs import PAIR_CONFIGS  # noqa: E402

GOLDEN_PATH = ROOT / "tests" / "data" / "alignment_golden.json"


def stub_normalizer(word, pos_tag, wordlist, test=False):
    """Mirrors the stub the golden generator used."""
    normalized = f"{word.lower()}|{pos_tag}"
    if test:
        return normalized
    if any(ch.isdigit() for ch in word):
        return None, normalized
    return (len(word) * 7 + ord(pos_tag)) % 1000, normalized


class AlignerUnderTest(AwesomePairAligner):
    """Skips model resolution; only the pure stages are exercised here."""

    def __init__(self, config):
        self.config = config
        self.components = PairLanguageComponents(
            src_morphological_batch=None,
            trg_morphological_batch=None,
            src_normalizer=stub_normalizer,
            trg_normalizer=stub_normalizer,
            exceptions=[["banned_src", "banned_trg"]],
        )
        self.backend_name = f"alignment.{config.pair}"


@unittest.skipIf(torch is None, "torch is required for the alignment core")
class AwesomePairEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(GOLDEN_PATH.read_text())

    def test_golden_covers_every_configured_pair(self):
        self.assertEqual(
            sorted(self.golden["pairs"]),
            sorted(pair for pair in PAIR_CONFIGS if pair in self.golden["pairs"]),
        )
        for pair in self.golden["pairs"]:
            self.assertIn(pair, PAIR_CONFIGS)

    def test_shared_core_reproduces_legacy_output(self):
        cases = self.golden["cases"]
        for pair, expected_results in self.golden["pairs"].items():
            aligner = AlignerUnderTest(PAIR_CONFIGS[pair])
            for index, (case, expected) in enumerate(zip(cases, expected_results)):
                with self.subTest(pair=pair, case=index):
                    softmax_inter = torch.tensor(case["matrix"], dtype=torch.bool)
                    alignmented = aligner.postprocess_alignment(
                        softmax_inter,
                        case["sub2word_map_src"],
                        case["sub2word_map_trg"],
                        case["pos_src"],
                        case["pos_trg"],
                        case["sent_src"],
                        case["sent_trg"],
                    )
                    serialized = [
                        [list(a), list(b), list(c), list(d)]
                        for a, b, c, d in alignmented
                    ]
                    self.assertEqual(serialized, expected["alignmented"])

                    # Relation order follows set iteration over POS tags, which
                    # varies with the process hash seed in the legacy code too,
                    # so only the multiset of relations is a real contract.
                    relations = aligner.extract_relations(alignmented, {}, True)
                    self.assertEqual(sorted(relations), sorted(expected["relations"]))

    def test_golden_data_is_not_vacuous(self):
        for pair, results in self.golden["pairs"].items():
            groups = sum(len(result["alignmented"]) for result in results)
            relations = sum(len(result["relations"]) for result in results)
            self.assertGreater(groups, 0, f"{pair} produced no alignment groups")
            self.assertGreater(relations, 0, f"{pair} produced no relations")


if __name__ == "__main__":
    unittest.main()
