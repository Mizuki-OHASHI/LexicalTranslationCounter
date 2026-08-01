"""German-English alignment.

Everything below is declaration only; the pipeline itself lives in
`ltc.backends.alignment.awesome_pair`.
"""

import os

from ltc.backends.alignment.awesome_pair import build_pair_aligner
from ltc.config.pairs import get_pair_config
from morphological.de_morphological import de_morphological_batch
from morphological.en_morphological import en_morphological_batch
from normalizer.de_normalizer import de_normalizer
from normalizer.en_normalizer import en_normalizer

CONFIG = get_pair_config("de_en")

ALIGNER = build_pair_aligner(
    CONFIG,
    module_file=__file__,
    src_morphological_batch=de_morphological_batch,
    trg_morphological_batch=en_morphological_batch,
    src_normalizer=de_normalizer,
    trg_normalizer=en_normalizer,
    exceptions_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "exceptions.csv"
    ),
)

alignment = ALIGNER.alignment
alignment_batch = ALIGNER.alignment_batch
runtime_check = ALIGNER.runtime_check
runtime_metadata = ALIGNER.runtime_metadata
