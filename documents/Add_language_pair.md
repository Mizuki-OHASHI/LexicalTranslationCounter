# Add a language pair

A pair connects two languages that already have a normalizer and a morphological
function. If one of yours does not, start with
[Add language](Add_language.md).

The alignment pipeline itself is shared. Adding a pair is **a config entry plus a
30-line module**, not a new implementation. `en_ru` is the worked example; the
commands below are the ones it was actually built with.

## 0. Name the pair

Language codes are always in **alphabetical order**: `en_ru`, not `ru_en`. Within
the pipeline, `src` means the first code and `trg` the second. Paths and Python
identifiers use `_`; Docker build args and `shell_scripts/` directories use `-`.

## 1. Add the alignment config

`src/ltc/config/pairs.py` holds every pair's constants in one place. Add an entry:

```python
"en_ru": PairAlignmentConfig(
    pair="en_ru",
    model_dir_name=None,
    src_ignore_rules=(EN_NEGATION,),
    trg_ignore_rules=(TokenIgnoreRule(words=("не",), offset=1),),
    threshold=4e-7,
),
```

**`*_ignore_rules`** describe negation and other constructions where the aligned
token should be dropped. `offset` is where the token to drop sits relative to the
marker word: English `not` negates what follows it (`+1`), Japanese `ない` trails
what it negates (`-1`), and French needs two rules for `ne ... pas`. Use
`require_next_pos` when the rule is conditional — `en_zh` uses it to drop an
English copula only when a verb follows.

**`trg_word_sep`** is `""` for languages that do not put spaces between words
(`ja`, `zh`) and `" "` otherwise.

**`threshold`** is the alignment cutoff. Existing pairs use `4e-7` or `1e-3`;
match the pair whose model you are closest to and revisit once you have a
fine-tuned model.

**`model_dir_name`** names a legacy directory under `src/model/`. Use `None` when
no fine-tuned model exists — resolution then falls back to
`bert-base-multilingual-cased`, which is enough to smoke-test but is not a
production setup.

## 2. Add the pair module

`src/alignment/{la1}_{la2}/__init__.py` only wires the two languages in:

```python
CONFIG = get_pair_config("en_ru")

ALIGNER = build_pair_aligner(
    CONFIG,
    module_file=__file__,
    src_morphological_batch=en_morphological_batch,
    trg_morphological_batch=ru_morphological_batch,
    src_normalizer=en_normalizer,
    trg_normalizer=ru_normalizer,
    exceptions_path=...,
)

alignment = ALIGNER.alignment
alignment_batch = ALIGNER.alignment_batch
runtime_check = ALIGNER.runtime_check
runtime_metadata = ALIGNER.runtime_metadata
```

Also create an empty `src/alignment/{la1}_{la2}/exceptions.csv`. It holds
`normalized_src,normalized_trg` pairs that should never be counted as a
translation, and is the right place for corrections found during review.

## 3. Register the pair

1. `src/ltc/diagnostics.py` — add `alignment.{la1}_{la2}` to `MODULE_GROUPS`.
2. `pyproject.toml` — add a `{la1}-{la2}` extra.
3. `shell_scripts/{la1}-{la2}/` — `requirements.txt` and `install.sh`.
4. `scripts/setup_local_runtime.py` — add a `RUNTIMES` entry so contributors can
   build the environment with one command.

`tests/test_pair_config_registry.py` checks most of this and needs no language
runtime installed, so run it first:

```bash
python3 -m unittest tests.test_pair_config_registry
```

## 4. Get data

You need a bilingual corpus and a wordlist per language and part of speech.
Public parallel corpora (ParaCrawl, OPUS) ship as `la1<TAB>la2` text with no ids;
`scripts/build_smoke_project.py` turns that into a runnable smoke project,
deriving wordlists from the sample by frequency:

```bash
python3 scripts/build_smoke_project.py \
    --pair en_ru --input ~/Downloads/en-ru.txt --scan 200000 --rows 200
```

This writes `projects/smoke/{pair}/input/`. Add a README there describing where
the data came from and under what licence.

**Look at what comes out.** Web-crawled corpora carry boilerplate, adult-site
spam and misaligned rows, and all of that lands in the wordlist. The smoke
sample is for proving the pipeline runs; a production run needs a real cleaning
pass (ParaCrawl publishes Bicleaner scores for exactly this).

Schemas for every file are in [File reference](File_reference.md).

## 5. Verify

```bash
PYTHONPATH=src python3 -m ltc.cli.doctor --group alignment
```

Then a real run over a handful of rows:

```bash
ROOT=$(pwd) .venv/bin/python src/count_function.py 0 en ru \
  --input-dir projects/smoke/en_ru/input \
  --output-dir .tmp/en_ru_smoke \
  --max-rows 20
```

Decode a few relations from `relations_{pair}_{pos}.csv` back through the
wordlists and read them as a bilingual speaker would. On the first `en_ru` run,
11 of 12 relations were correct translations with no fine-tuned model — that is
the kind of check that belongs in the pull request.

## 6. Optional: register a fine-tuned model

The multilingual-BERT fallback is a starting point, not a production aligner.
When you have an Awesome Align model for the pair:

```bash
PYTHONPATH=src python3 -m ltc.cli.register_awesome_model \
    --pair en-ru --source /path/to/model --copy-files
PYTHONPATH=src python3 -m ltc.cli.awesome_model_status --pair en-ru
```

Set `LTC_REQUIRE_PRODUCTION_MODEL=1` for production runs so a missing fine-tuned
model fails loudly instead of silently falling back.
