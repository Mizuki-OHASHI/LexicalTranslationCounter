"""Alignment configuration for every language pair.

This is the whole per-pair surface of the Awesome Align pipeline. Adding a pair
means adding one entry here plus a thin `src/alignment/{pair}/__init__.py` that
wires in the two languages' morphological and normalizer functions.

Language codes in a pair name are always in alphabetical order, and `src` refers
to the first code, `trg` to the second.
"""

from __future__ import annotations

from ltc.backends.alignment.awesome_pair import PairAlignmentConfig, TokenIgnoreRule

# English negation attaches to the token that follows it.
EN_NEGATION = TokenIgnoreRule(words=("not", "n't"), offset=1)

PAIR_CONFIGS = {
    "de_en": PairAlignmentConfig(
        pair="de_en",
        model_dir_name="awesome_model_with_co",
        src_ignore_rules=(TokenIgnoreRule(words=("nicht",), offset=1),),
        trg_ignore_rules=(EN_NEGATION,),
        threshold=4e-7,
    ),
    "en_es": PairAlignmentConfig(
        pair="en_es",
        # Never had a fine-tuned model; runs on plain multilingual BERT.
        model_dir_name=None,
        src_ignore_rules=(EN_NEGATION,),
        trg_ignore_rules=(TokenIgnoreRule(words=("no",), offset=1),),
        threshold=4e-7,
    ),
    "en_fr": PairAlignmentConfig(
        pair="en_fr",
        model_dir_name="awesome_model_without_co",
        src_ignore_rules=(EN_NEGATION,),
        # French wraps the verb: `ne ... pas`.
        trg_ignore_rules=(
            TokenIgnoreRule(words=("pas",), offset=1),
            TokenIgnoreRule(words=("ne",), offset=-1),
        ),
        threshold=1e-3,
    ),
    "en_it": PairAlignmentConfig(
        pair="en_it",
        model_dir_name="awesome_model_with_co",
        # BUG (preserved): these were copied from de_en and never adapted, so
        # the src rule is German and the trg rule is English on an en->it pair.
        # Italian negation (`non`) is not handled at all. Fixing this changes
        # counting output, so it belongs in a behaviour change, not a refactor.
        src_ignore_rules=(TokenIgnoreRule(words=("nicht",), offset=1),),
        trg_ignore_rules=(EN_NEGATION,),
        threshold=4e-7,
    ),
    "en_ko": PairAlignmentConfig(
        pair="en_ko",
        model_dir_name=None,
        src_ignore_rules=(EN_NEGATION,),
        trg_ignore_rules=(
            TokenIgnoreRule(
                words=(
                    "안",
                    "아니다",
                    "아닙니다",
                    "아니에요",
                    "아닙니다",
                    "아니에요",
                ),
                offset=1,
            ),
            # BUG (preserved): French `ne`, copied from en_fr. Inert on Korean
            # text, but it is not a Korean rule.
            TokenIgnoreRule(words=("ne",), offset=-1),
        ),
        threshold=4e-7,
    ),
    "en_ru": PairAlignmentConfig(
        pair="en_ru",
        # No fine-tuned en-ru model yet, so this runs on plain multilingual
        # BERT. Register one with `ltc.cli.register_awesome_model --pair en-ru`
        # and it takes precedence without touching this file.
        model_dir_name=None,
        src_ignore_rules=(EN_NEGATION,),
        # Russian negates with a preposed particle `не`, like English `not`.
        trg_ignore_rules=(TokenIgnoreRule(words=("не",), offset=1),),
        # Matches the other multilingual-BERT pairs (en_es, en_ko); revisit
        # once a fine-tuned model is registered.
        threshold=4e-7,
    ),
    "en_zh": PairAlignmentConfig(
        pair="en_zh",
        model_dir_name="awesome_model_without_co",
        src_ignore_rules=(
            EN_NEGATION,
            # Drop an English copula when it merely supports a following verb,
            # since Chinese does not realise it as a separate word.
            TokenIgnoreRule(
                words=("is", "are", "was", "were", "am", "be", "been", "being"),
                offset=0,
                require_next_pos="v",
            ),
        ),
        trg_ignore_rules=(TokenIgnoreRule(words=("不", "没"), offset=1),),
        trg_word_sep="",
        threshold=1e-3,
    ),
    "fr_ja": PairAlignmentConfig(
        pair="fr_ja",
        model_dir_name="awesome_model_without_co",
        src_ignore_rules=(
            TokenIgnoreRule(words=("pas",), offset=1),
            TokenIgnoreRule(words=("ne",), offset=-1),
        ),
        # Japanese negation trails the verb it negates.
        trg_ignore_rules=(
            TokenIgnoreRule(words=("ない", "なかろう", "なく", "なかっ", "なければ"), offset=-1),
        ),
        trg_word_sep="",
        threshold=1e-3,
    ),
}


def get_pair_config(pair):
    try:
        return PAIR_CONFIGS[pair]
    except KeyError:
        raise KeyError(
            f"no alignment config for {pair!r}; known pairs: "
            f"{', '.join(sorted(PAIR_CONFIGS))}"
        ) from None
