"""Shared Awesome Align pipeline for a bilingual pair.

Every pair under `src/alignment/` ran its own ~550-line copy of the same
BERT-attention alignment flow, differing only in a small block of constants.
This module holds the flow once; a pair module declares a
:class:`PairAlignmentConfig` and wires in its language components.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

from ltc.backends.alignment.awesome_utils import (
    build_awesome_input_ids_and_subword_map,
    check_awesome_model_runtime,
    ensure_production_model,
    load_awesome_model_and_tokenizer,
    resolve_awesome_model_selection,
    warn_on_smoke_model,
)
from ltc.env import get_bool_env, get_positive_int_env
from ltc.timing import timed


@dataclass(frozen=True)
class TokenIgnoreRule:
    """Drop alignment candidates sitting at a fixed offset from a marker word.

    Negation is the motivating case: for English `not`/`n't` the *following*
    token carries the negated meaning, so `offset=1` removes it from the
    alignment. French splits negation across `ne ... pas`, which is two rules
    with offsets -1 and +1.

    `require_next_pos` covers the Chinese copula case, where a `be` verb is only
    dropped when the token after it is itself a verb.

    `case_insensitive` exists because matching is on the raw token, so a
    sentence-initial negator is missed by an all-lowercase word list. It
    defaults to False to preserve the behaviour of the pairs that predate this
    option; new pairs should set it.
    """

    words: Tuple[str, ...]
    offset: int
    require_next_pos: Optional[str] = None
    case_insensitive: bool = False

    def matches(self, token):
        if self.case_insensitive:
            return token.lower() in tuple(word.lower() for word in self.words)
        return token in self.words


@dataclass(frozen=True)
class PairAlignmentConfig:
    """Everything that actually differs between language pairs."""

    pair: str
    model_dir_name: Optional[str] = None
    src_ignore_rules: Tuple[TokenIgnoreRule, ...] = ()
    trg_ignore_rules: Tuple[TokenIgnoreRule, ...] = ()
    src_word_sep: str = " "
    trg_word_sep: str = " "
    align_layer: int = 8
    threshold: float = 4e-7
    max_word_len: int = 3
    fallback_model_name: str = "bert-base-multilingual-cased"

    @property
    def la1(self):
        return self.pair.split("_")[0]

    @property
    def la2(self):
        return self.pair.split("_")[1]


@dataclass
class PairLanguageComponents:
    """The per-language callables a pair plugs into the shared flow."""

    src_morphological_batch: Callable
    trg_morphological_batch: Callable
    src_normalizer: Callable
    trg_normalizer: Callable
    exceptions: Sequence[Sequence[str]] = field(default_factory=list)


def load_exceptions(path):
    with open(path, "r") as f:
        return list(csv.reader(f, delimiter=","))


def build_ignore_indexes(tokens, pos_codes, rules):
    """Collect token indexes that the alignment should skip."""
    ignored = set()
    for rule in rules:
        for index, token in enumerate(tokens):
            if not rule.matches(token):
                continue
            if rule.require_next_pos is not None:
                if index + 1 >= len(tokens):
                    continue
                if pos_codes[index + 1] != rule.require_next_pos:
                    continue
            ignored.add(index + rule.offset)
    return ignored


def group_aligned_words(sorted_indexes, tokens, pos_codes, word_sep):
    """Collapse the content-bearing span of an alignment group into one unit.

    Indexes carrying a POS code are "independent" words. Everything between the
    first and last of them belongs to the same lexical unit and gets joined;
    tokens outside that span stay separate.
    """
    independent = [
        position
        for position, index in enumerate(sorted_indexes)
        if pos_codes[index] != ""
    ]

    if len(independent) <= 1:
        return (
            [tokens[index] for index in sorted_indexes],
            [pos_codes[index] for index in sorted_indexes],
        )

    head = slice(None, independent[0])
    if independent[-1] + 1 < len(sorted_indexes):
        span = slice(independent[0], independent[-1] + 1)
        tail = slice(independent[-1] + 1, None)
    else:
        span = slice(independent[0], None)
        tail = slice(0, 0)

    words = (
        [tokens[index] for index in sorted_indexes[head]]
        + [word_sep.join(tokens[index] for index in sorted_indexes[span])]
        + [tokens[index] for index in sorted_indexes[tail]]
    )
    tags = (
        [pos_codes[index] for index in sorted_indexes[head]]
        + ["/".join(pos_codes[index] for index in sorted_indexes[span])]
        + [pos_codes[index] for index in sorted_indexes[tail]]
    )
    return words, tags


def last_independent_index(pos_codes):
    """Index of the last content-bearing token, or -1 when there is none."""
    independent_index = -1
    for index, pos_tag in enumerate(pos_codes):
        if pos_tag != "":
            independent_index = index
    return independent_index


def merge_subword_alignments(align_subwords, sub2word_map_src, sub2word_map_trg):
    """Group subword-level alignment points into connected word-index pairs."""
    index_pair_list = []
    for i_tmp, j_tmp in align_subwords:
        i = sub2word_map_src[i_tmp]
        j = sub2word_map_trg[j_tmp]
        src_index = -1
        trg_index = -1
        for position, index_pair in enumerate(index_pair_list):
            if i in index_pair[0]:
                src_index = position
            if j in index_pair[1]:
                trg_index = position
        if src_index == -1 and trg_index == -1:
            index_pair_list.append([{i}, {j}])
        elif src_index != -1 and trg_index != -1 and src_index != trg_index:
            # NOTE: kept as-is from the original per-pair implementations. The
            # set unions here are computed and discarded, so two groups that
            # should merge stay separate. Changing it changes counting output,
            # so it is a behaviour fix rather than a refactor.
            index_pair_list[src_index][0] | index_pair_list[trg_index][0]
            index_pair_list[src_index][1] | index_pair_list[trg_index][1]
        elif src_index != -1:
            index_pair_list[src_index][0].add(i)
            index_pair_list[src_index][1].add(j)
        elif trg_index != -1:
            index_pair_list[trg_index][0].add(i)
            index_pair_list[trg_index][1].add(j)
    return index_pair_list


class AwesomePairAligner:
    """The `alignment` / `alignment_batch` contract, driven by a config."""

    def __init__(self, config, module_file, components):
        self.config = config
        self.components = components
        self.model_selection = resolve_awesome_model_selection(
            module_file,
            config.model_dir_name,
            pair_name=config.pair,
            fallback_model_name=config.fallback_model_name,
        )
        self.backend_name = f"alignment.{config.pair}"
        self._model = None
        self._tokenizer = None
        self._smoke_model_warned = False
        self._encoder_truncated = False

    # -- naming helper so timing keys stay stable per pair -----------------

    def _timer(self, suffix, **kwargs):
        return timed(f"{self.backend_name}.{suffix}", **kwargs)

    # -- model ------------------------------------------------------------

    def get_model_and_tokenizer(self):
        ensure_production_model(
            self.model_selection,
            backend_name=self.backend_name,
            pair_name=self.config.pair,
        )
        if not self._smoke_model_warned:
            warn_on_smoke_model(self.model_selection, backend_name=self.backend_name)
            self._smoke_model_warned = True

        if self._model is None or self._tokenizer is None:
            with self._timer("load_model_bundle"):
                model, tokenizer = load_awesome_model_and_tokenizer(
                    self.model_selection.model_spec
                )
            self._model = model.to(self.device)
            self._tokenizer = tokenizer

        if get_bool_env("LTC_BERT_EARLY_STOP", False) and not self._encoder_truncated:
            # Only hidden_states[align_layer] is read, so the layers above it
            # are pure cost. Truncating keeps that index pointing at the same
            # tensor because it becomes the last layer.
            with self._timer("bert_early_stop_truncate"):
                self._model.encoder.layer = self._model.encoder.layer[
                    : self.config.align_layer
                ]
            self._encoder_truncated = True

        return self._model, self._tokenizer

    @property
    def device(self):
        import torch

        if not hasattr(self, "_device"):
            self._device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        return self._device

    def runtime_check(self):
        check_awesome_model_runtime(
            self.model_selection.model_spec, backend_name=self.backend_name
        )

    def runtime_metadata(self):
        selection = self.model_selection
        return {
            "model_spec": selection.model_spec,
            "resolution_source": selection.resolution_source,
            "model_profile": selection.profile,
            "production_ready": selection.production_ready,
            "registry_dir": selection.registry_dir,
            "model_metadata": selection.metadata,
            "note": selection.note,
        }

    # -- alignment core ---------------------------------------------------

    def _preprocess_sentence_pair(self, sent_src, sent_trg):
        _, tokenizer = self.get_model_and_tokenizer()
        ids_src, sub2word_map_src = build_awesome_input_ids_and_subword_map(
            tokenizer, sent_src
        )
        ids_trg, sub2word_map_trg = build_awesome_input_ids_and_subword_map(
            tokenizer, sent_trg
        )
        return ids_src, ids_trg, sub2word_map_src, sub2word_map_trg

    def postprocess_alignment(
        self,
        softmax_inter,
        sub2word_map_src,
        sub2word_map_trg,
        pos_src,
        pos_trg,
        sent_src,
        sent_trg,
    ):
        import torch

        config = self.config
        index_pair_list = merge_subword_alignments(
            torch.nonzero(softmax_inter, as_tuple=False),
            sub2word_map_src,
            sub2word_map_trg,
        )

        index_src_ignore = build_ignore_indexes(
            sent_src, pos_src, config.src_ignore_rules
        )
        index_trg_ignore = build_ignore_indexes(
            sent_trg, pos_trg, config.trg_ignore_rules
        )

        alignmented_l = []
        for index_pair in index_pair_list:
            if not index_pair[0].isdisjoint(index_src_ignore):
                continue
            if not index_pair[1].isdisjoint(index_trg_ignore):
                continue

            src_words, src_tags = group_aligned_words(
                sorted(index_pair[0]), sent_src, pos_src, config.src_word_sep
            )
            trg_words, trg_tags = group_aligned_words(
                sorted(index_pair[1]), sent_trg, pos_trg, config.trg_word_sep
            )
            alignmented_l.append((src_tags, src_words, trg_tags, trg_words))
        return alignmented_l

    def align_from_morphs_batch(self, sent_srcs, pos_srcs, sent_trgs, pos_trgs):
        import torch

        batch_size = len(sent_srcs)

        with self._timer("tokenize_prepare_batch", items=batch_size):
            ids_srcs, ids_trgs, sub2word_map_srcs, sub2word_map_trgs = zip(
                *[
                    self._preprocess_sentence_pair(sent_src, sent_trg)
                    for sent_src, sent_trg in zip(sent_srcs, sent_trgs)
                ]
            )

        with self._timer("padding_src_batch", items=batch_size):
            padded_ids_src = self._pad(ids_srcs)
        with self._timer("padding_trg_batch", items=batch_size):
            padded_ids_trg = self._pad(ids_trgs)

        model, _ = self.get_model_and_tokenizer()
        model.eval()

        align_layer = self.config.align_layer
        with torch.no_grad():
            with self._timer("model_forward_src_batch", items=batch_size):
                out_srcs = model(
                    padded_ids_src.to(self.device), output_hidden_states=True
                )[2][align_layer].to("cpu")
            with self._timer("model_forward_trg_batch", items=batch_size):
                out_trgs = model(
                    padded_ids_trg.to(self.device), output_hidden_states=True
                )[2][align_layer].to("cpu")

        softmax_inter_list = []
        with self._timer("softmax_intersections_batch", items=batch_size):
            for ids_src, ids_trg, out_src, out_trg in zip(
                ids_srcs, ids_trgs, out_srcs, out_trgs
            ):
                out_src = out_src[1 : ids_src.size(0) - 1]
                out_trg = out_trg[1 : ids_trg.size(0) - 1]

                dot_prod = torch.matmul(out_src, out_trg.transpose(-1, -2))
                softmax_srctrg = torch.nn.Softmax(dim=-1)(dot_prod)
                softmax_trgsrc = torch.nn.Softmax(dim=-2)(dot_prod)
                threshold = self.config.threshold
                softmax_inter_list.append(
                    (softmax_srctrg > threshold) * (softmax_trgsrc > threshold)
                )

        with self._timer("awesome_postprocessing_batch", items=batch_size):
            return [
                self.postprocess_alignment(
                    softmax_inter,
                    sub2word_map_src,
                    sub2word_map_trg,
                    pos_src,
                    pos_trg,
                    sent_src,
                    sent_trg,
                )
                for (
                    softmax_inter,
                    sub2word_map_src,
                    sub2word_map_trg,
                    pos_src,
                    pos_trg,
                    sent_src,
                    sent_trg,
                ) in zip(
                    softmax_inter_list,
                    sub2word_map_srcs,
                    sub2word_map_trgs,
                    pos_srcs,
                    pos_trgs,
                    sent_srcs,
                    sent_trgs,
                )
            ]

    def _pad(self, ids_list):
        import torch

        _, tokenizer = self.get_model_and_tokenizer()
        pad_token_id = tokenizer.pad_token_id
        max_size = max(ids.size(0) for ids in ids_list)
        return torch.stack(
            [
                torch.cat(
                    [
                        ids,
                        torch.tensor(
                            [pad_token_id] * (max_size - ids.size(0)),
                            dtype=torch.long,
                            device=ids.device,
                        ),
                    ]
                )
                for ids in ids_list
            ]
        )

    # -- relation extraction ----------------------------------------------

    def _lookup_side(
        self,
        word_l,
        independent_index,
        independent_pos_tag,
        normalizer,
        word_sep,
        wordlist,
        test,
    ):
        normalized_dict = {}
        pos_tag_s = set()
        found = False
        word = ""

        max_len = min(len(word_l), self.config.max_word_len)
        for length in range(max_len, 0, -1):
            for combination in itertools.combinations(range(len(word_l)), length):
                if independent_index in combination:
                    word = word_sep.join(word_l[index] for index in combination)
                    for pos_tag in independent_pos_tag:
                        if test:
                            normalized = normalizer(word, pos_tag, wordlist, test)
                            word_id = -1
                        else:
                            word_id, normalized = normalizer(
                                word, pos_tag, wordlist, test
                            )
                        if word_id is not None:
                            normalized_dict.setdefault(word, {})[pos_tag] = {
                                "id": int(word_id),
                                "word_normalized": normalized,
                            }
                            found = True
                            pos_tag_s.add(pos_tag)
                if found:
                    break
            else:
                continue
            break
        return found, word, pos_tag_s, normalized_dict

    def extract_relations(self, alignmented, wordlist, test=False):
        components = self.components
        output_l = []

        for src_tags, src_words, trg_tags, trg_words in alignmented:
            independent_index_src = last_independent_index(src_tags)
            independent_index_trg = last_independent_index(trg_tags)

            independent_pos_tag = set(src_tags[independent_index_src].split("/")) & set(
                trg_tags[independent_index_trg].split("/")
            )
            independent_pos_tag.discard("")

            found_src, src_word, src_pos_tag_s, src_normalized = self._lookup_side(
                src_words,
                independent_index_src,
                independent_pos_tag,
                components.src_normalizer,
                self.config.src_word_sep,
                wordlist,
                test,
            )
            found_trg, trg_word, trg_pos_tag_s, trg_normalized = self._lookup_side(
                trg_words,
                independent_index_trg,
                independent_pos_tag,
                components.trg_normalizer,
                self.config.trg_word_sep,
                wordlist,
                test,
            )

            if not (found_src and found_trg):
                continue

            for pos_tag in src_pos_tag_s & trg_pos_tag_s:
                pair = [
                    src_normalized[src_word][pos_tag]["word_normalized"],
                    trg_normalized[trg_word][pos_tag]["word_normalized"],
                ]
                if pair in components.exceptions:
                    continue
                output_l.append(
                    [
                        pos_tag,
                        str(src_normalized[src_word][pos_tag]["id"]),
                        src_word,
                        str(trg_normalized[trg_word][pos_tag]["id"]),
                        trg_word,
                    ]
                )
        return output_l

    # -- public contract --------------------------------------------------

    def alignment_batch(self, corpus_rows, wordlist, test=False):
        batch_size = len(corpus_rows)
        sub_batch_size = get_positive_int_env("LTC_ALIGNMENT_BATCH_SIZE", 10)

        with self._timer("corpus_preprocess_batch", items=batch_size):
            corpus_rows = [preprocess_corpus_row(row) for row in corpus_rows]

        with self._timer("awesome_alignment_batch_total", items=batch_size):
            sentence_srcs = [row[1] for row in corpus_rows]
            sentence_trgs = [row[2] for row in corpus_rows]
            with self._timer("morphological_src_batch", items=batch_size):
                sent_srcs, pos_srcs = self.components.src_morphological_batch(
                    sentence_srcs
                )
            with self._timer("morphological_trg_batch", items=batch_size):
                sent_trgs, pos_trgs = self.components.trg_morphological_batch(
                    sentence_trgs
                )

            alignmented_ls = []
            for start in range(0, batch_size, sub_batch_size):
                end = min(start + sub_batch_size, batch_size)
                alignmented_ls.extend(
                    self.align_from_morphs_batch(
                        sent_srcs[start:end],
                        pos_srcs[start:end],
                        sent_trgs[start:end],
                        pos_trgs[start:end],
                    )
                )

        assert len(alignmented_ls) == len(corpus_rows)

        with self._timer("relation_postprocess_batch", items=batch_size):
            return [
                self.extract_relations(alignmented, wordlist, test)
                for alignmented in alignmented_ls
            ]

    def alignment(self, corpus_row, wordlist, test=False):
        return self.alignment_batch([corpus_row], wordlist, test)[0]


def preprocess_corpus_row(corpus_row):
    """Strip the `@` markers the corpus files use around candidate words."""
    corpus_row = list(corpus_row)
    corpus_row[1] = corpus_row[1].replace("@", "")
    corpus_row[2] = corpus_row[2].replace("@", "")
    return corpus_row


def build_pair_aligner(
    config,
    module_file,
    src_morphological_batch,
    trg_morphological_batch,
    src_normalizer,
    trg_normalizer,
    exceptions_path,
):
    """Construct the aligner a pair module re-exports.

    Returns the aligner itself; pair modules bind `alignment`,
    `alignment_batch`, `runtime_check` and `runtime_metadata` from it so the
    module keeps satisfying `ltc.runtime.load_alignment_runtime`.
    """
    components = PairLanguageComponents(
        src_morphological_batch=src_morphological_batch,
        trg_morphological_batch=trg_morphological_batch,
        src_normalizer=src_normalizer,
        trg_normalizer=trg_normalizer,
        exceptions=load_exceptions(exceptions_path),
    )
    return AwesomePairAligner(config, module_file, components)
