"""English-Spanish alignment.

Everything below is declaration only; the pipeline itself lives in
`ltc.backends.alignment.awesome_pair` and the constants in `ltc.config.pairs`.
"""

import os

from ltc.backends.alignment.awesome_pair import build_pair_aligner
from ltc.config.pairs import get_pair_config
from morphological.en_morphological import en_morphological_batch
from morphological.es_morphological import es_morphological_batch
from normalizer.en_normalizer import en_normalizer
from normalizer.es_normalizer import es_normalizer

CONFIG = get_pair_config("en_es")

ALIGNER = build_pair_aligner(
    CONFIG,
    module_file=__file__,
    src_morphological_batch=en_morphological_batch,
    trg_morphological_batch=es_morphological_batch,
    src_normalizer=en_normalizer,
    trg_normalizer=es_normalizer,
    exceptions_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "exceptions.csv"
    ),
)

alignment = ALIGNER.alignment
alignment_batch = ALIGNER.alignment_batch
runtime_check = ALIGNER.runtime_check
runtime_metadata = ALIGNER.runtime_metadata
