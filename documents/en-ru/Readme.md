# en-ru notes

## Status

`en_ru` runs end to end on the shared alignment core. It has **no fine-tuned
Awesome Align model**, so it resolves to `bert-base-multilingual-cased`. Treat
current output as indicative, not production quality.

## Russian runtime

[Natasha](https://github.com/natasha/natasha) (MIT) covers both LTC language
contracts:

- `morphological.ru_morphological` — Razdel tokenizer + Slovnet `NewsMorphTagger`
  (Universal Dependencies POS tags)
- `normalizer.ru_normalizer` — `MorphVocab.lemmatize`

Both share one lazily-loaded backend instance in
`ltc.backends.linguistic.natasha_ru`, so the models load once per process.

Install with `pip install natasha`, or `python3 -m pip install -e '.[en-ru]'`.
Inference is CPU-only NumPy; models are roughly 30 MB and load in well under a
second. No spaCy model download step is involved.

### Known caveats

- **Domain mismatch.** The Slovnet models are trained on news text. ParaCrawl is
  web-crawled, so upstream accuracy figures do not transfer directly. This is
  the first thing to check if alignment quality looks off.
- **pymorphy2.** Natasha lemmatizes through pymorphy2, which is unmaintained and
  emits a `pkg_resources` deprecation warning on import. It works today, but it
  is the most likely future breakage in this runtime.
- Russian normalization exceptions live in
  `src/normalizer/normalize_data/ru/`. Unlike the other languages, `ru` reads
  these tables correctly — see the note in
  [Add language](../Add_language.md).

## Corpus

ParaCrawl en-ru, distributed as a two-column `en<TAB>ru` TSV with no ids. The
alphabetical pair order (`en` < `ru`) matches the file's column order, so no
column swap is needed.

Convert a slice into a runnable project with
`scripts/build_smoke_project.py`; see
[projects/smoke/en_ru/README.md](../../projects/smoke/en_ru/README.md).

## Alignment config

In `src/ltc/config/pairs.py`. Russian negation is a preposed `не`, so the target
rule is a single `offset=1` entry, structurally the same as English `not`.

The threshold is `4e-7`, matching the other multilingual-BERT pairs (`en_es`,
`en_ko`). Revisit it once a fine-tuned model is registered.
