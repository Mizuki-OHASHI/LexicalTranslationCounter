# en_ru Smoke Project

The smallest Docker-free check that the `en_ru` alignment path runs end to end.

This is not a quality benchmark. It is a real-backend smoke check.

## Runtime notes

Read [documents/en-ru/Readme.md](../../../documents/en-ru/Readme.md) first.

Russian tokenization, POS tagging and lemmatization all come from
[Natasha](https://github.com/natasha/natasha) (MIT). Inference is CPU-only, so no
GPU setup is involved.

`en_ru` has **no fine-tuned Awesome Align model yet**, so model resolution falls
back to `bert-base-multilingual-cased`:

1. `LTC_AWESOME_ALIGN_MODEL_EN_RU`
2. `LTC_AWESOME_ALIGN_MODEL`
3. canonical repo registry: `models/en-ru/awesome-align/production/`
4. fallback: `bert-base-multilingual-cased`

## Local setup

```bash
python3 scripts/setup_local_runtime.py --language-pair en-ru
```

Or the package extra:

```bash
python3 -m pip install -e '.[en-ru]'
```

## Quick doctor check

```bash
PYTHONPATH=src python3 -m ltc.cli.doctor --group alignment
```

`alignment.en_ru` should report `OK`, noting the smoke/dev model.

## Small real-backend run

```bash
ROOT=$(pwd) .venv/bin/python src/count_function.py 0 en ru \
  --input-dir projects/smoke/en_ru/input \
  --output-dir .tmp/en_ru_smoke \
  --max-rows 20
```

Check `relations_en_ru_*.csv`, `corpus_en_ru.csv` and `timing_en_ru.json` under
`.tmp/en_ru_smoke/`. To start fresh, remove that directory.

## About the tracked input

`input/` holds 200 sentence pairs sampled from **ParaCrawl en-ru**, plus
wordlists derived from that sample by frequency. It was generated with:

```bash
python3 scripts/build_smoke_project.py \
    --pair en_ru --input <paracrawl-en-ru.txt> --scan 200000 --rows 200
```

ParaCrawl is web-crawled, and the sample shows it: some rows are site
boilerplate, some pairs are only loosely parallel, and adult-site vocabulary
survives into the wordlist. That is acceptable for a smoke fixture and **not**
acceptable for a production run — clean the corpus first (ParaCrawl publishes
Bicleaner quality scores for this purpose).
