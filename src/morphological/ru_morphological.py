"""Russian tokenization and POS tagging.

Lemmatization lives in `normalizer.ru_normalizer`, which shares this backend
instance so the Natasha models are loaded once per process.
"""

from ltc.backends.linguistic.natasha_ru import BACKEND

ru_morphological = BACKEND.analyze
ru_morphological_batch = BACKEND.analyze_batch
runtime_check = BACKEND.runtime_check
runtime_metadata = BACKEND.runtime_metadata
