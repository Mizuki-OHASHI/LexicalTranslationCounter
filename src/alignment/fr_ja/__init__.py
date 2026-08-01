"""French-Japanese alignment.

Everything below is declaration only; the pipeline itself lives in
`ltc.backends.alignment.awesome_pair` and the constants in `ltc.config.pairs`.
"""

import os

from ltc.backends.alignment.awesome_pair import build_pair_aligner
from ltc.config.pairs import get_pair_config
from ltc.japanese.morphology import ja_morphological_batch
from ltc.japanese.normalization import ja_normalizer
from morphological.fr_morphological import fr_morphological_batch
from normalizer.fr_normalizer import fr_normalizer

CONFIG = get_pair_config("fr_ja")

ALIGNER = build_pair_aligner(
    CONFIG,
    module_file=__file__,
    src_morphological_batch=fr_morphological_batch,
    trg_morphological_batch=ja_morphological_batch,
    src_normalizer=fr_normalizer,
    trg_normalizer=ja_normalizer,
    exceptions_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "exceptions.csv"
    ),
)

alignment = ALIGNER.alignment
alignment_batch = ALIGNER.alignment_batch
runtime_check = ALIGNER.runtime_check
runtime_metadata = ALIGNER.runtime_metadata
