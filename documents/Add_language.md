# Add a language

A language contributes two functions to LTC: a **morphological** function that
splits a sentence into tokens and tags each one, and a **normalizer** that maps
any surface form to the single representative form the word network counts.

Once a language exists, pair it with another one via
[Add language pair](Add_language_pair.md).

`ru` was the most recent language added, so its files are the closest thing to a
worked example of everything below.

## 1. Choose a backend, and check it before you commit to it

Look for one library that can do **both** tokenization+POS and lemmatization.
That keeps the two functions consistent and halves the dependency surface.

What the existing languages use:

| Language | Tokenize + POS | Lemmatize |
| --- | --- | --- |
| `de`, `es`, `fr`, `it` | spaCy | spaCy tags + germalemma (`de`) |
| `en` | NLTK | NLTK WordNet |
| `ja` | Sudachi (default) / Juman++ | see `src/ltc/japanese/` |
| `ko` | konlpy | konlpy |
| `ru` | Natasha (Razdel + Slovnet) | Natasha (`MorphVocab`) |
| `zh` | jieba | jieba |

Before writing code, actually install the candidate and run a few real sentences
through it. Check the lemmas by hand. A backend that tags well but lemmatizes
badly will quietly degrade every pair the language appears in.

Note what the backend was **trained on**. Natasha's models, for example, are
news-domain; a web-crawled corpus is a different distribution and the published
accuracy numbers will not transfer.

## 2. Implement the morphological function

`src/morphological/{la}_morphological.py`

```
{la}_morphological(sentence)        -> (tokens, pos_codes)
{la}_morphological_batch(sentences) -> (tokens_l, pos_codes_l)
```

`pos_codes` is one entry per token, using the LTC codes `n`, `v`, `a`, `r`, or
`""` for anything that is not a content word.

If your backend emits **Universal Dependencies** POS tags, do not write your own
mapping — `ltc.constants.UD_POS_TO_LTC_CODE` already defines it, and there are
shared backends built on top:

- spaCy: `ltc.backends.linguistic.spacy_pos.build_spacy_morphological`, which
  also gives you the `LTC_{LA}_SPACY_MODEL`, `LTC_SPACY_PIPE_BATCH_SIZE`,
  `LTC_SPACY_PREFER_GPU` and `LTC_BUCKETING` knobs for free. `de` is one line of
  configuration on top of it.
- Anything else: model it on `ltc.backends.linguistic.natasha_ru`, which keeps
  the loaded models behind a lazy property so that merely importing the module
  never requires the model files.

Load models **lazily**. `ltc.cli.doctor` reports a clear missing-dependency error
only if the import itself succeeds.

## 3. Implement the normalizer

`src/normalizer/{la}_normalizer.py`

```
{la}_normalizer(word, pos_tag, wordlist, test=False)
    -> (id, word_normalized)     # normal
    -> word_normalized           # test=True
```

`pos_tag` is an LTC code, and `id` is `None` when the normalized form is not in
the wordlist. `word` can be a multi-token phrase, so tokenize inside the
function rather than assuming a single word.

`ltc.constants.LTC_CODE_TO_PRIMARY_UD_POS` maps the LTC code back to the UD tag
most lemmatizers expect.

### Exception tables

`src/normalizer/normalize_data/{la}/{n,v,a,r}_normalize.csv` holds
`original,normalized` overrides for forms the lemmatizer gets wrong. Create the
four files; starting with just the placeholder row is fine.

> **Known issue:** every language except `ru` loads these tables with
> `pandas.read_csv(..., index_col=0).to_dict()`, which produces
> `{1: {original: normalized}}`. The lookup then tests the word against the
> outer dict, so **those exception tables have never taken effect** — including
> the ~6000 curated English entries. `ru_normalizer` reads them correctly. Fixing
> the others changes counting output, so it needs its own change, not a
> drive-by edit.

## 4. Register the language

Four places, none of them optional:

1. `src/ltc/diagnostics.py` — add to both the `normalizer` and `morphological`
   entries of `MODULE_GROUPS`.
2. `pyproject.toml` — add a `{la}` extra listing the runtime dependencies.
3. `shell_scripts/{la}/` — `requirements.txt` and `install.sh`, run at Docker
   build time.
4. `documents/{la}/Readme.md` — only if the language needs setup notes that do
   not fit in a requirements file.

## 5. Verify

```bash
PYTHONPATH=src python3 -m ltc.cli.doctor --group normalizer --group morphological
```

Your language should be `OK`. Implementing `runtime_check()` and
`runtime_metadata()` in both modules makes that line report the model actually in
use instead of just "imported without error".

```bash
python3 -m unittest discover -s tests
```

Then add the language to a pair: [Add language pair](Add_language_pair.md).
