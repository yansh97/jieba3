"""Dictionary and HMM segmentation using precompiled log probability scores."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from pickle import load
from re import compile as compile_regex
from typing import (
    Callable,
    Dict,
    Final,
    Iterable,
    List,
    Literal,
    Optional,
    Pattern,
    Tuple,
)

NEGATIVE_INFINITY: Final[float] = float("-inf")
MIN_LOG_PROBABILITY: Final[float] = -3.14e100

ASCII_ALPHANUMERICS: Final[str] = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
ALPHANUMERIC_PATTERN: Final[Pattern[str]] = compile_regex(r"([a-zA-Z0-9]+(?:\.\d+)?%?)")
HAN_PATTERN: Final[Pattern[str]] = compile_regex(r"([\u4E00-\u9FD5]+)")
TEXT_PATTERN: Final[Pattern[str]] = compile_regex(
    r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)"
)

ModelName = Literal["base", "small", "large"]
# Nodes store (log probability score, child index); zero marks a nonword prefix.
BigramNode = Tuple[float, Optional[Dict[str, float]]]
UnigramNode = Tuple[float, Dict[str, BigramNode]]
StateScores = Tuple[float, float, float, float]

UNKNOWN_EMISSION_SCORES: Final[StateScores] = (
    MIN_LOG_PROBABILITY,
    MIN_LOG_PROBABILITY,
    MIN_LOG_PROBABILITY,
    MIN_LOG_PROBABILITY,
)


@dataclass
class DictionaryModel:
    __slots__ = ("prefix_index", "unknown_score")

    prefix_index: Dict[str, UnigramNode]
    unknown_score: float

    def contains_word(self, word: str) -> bool:
        first_node = self.prefix_index.get(word[0])
        if first_node is None:
            return False
        length = len(word)
        if length == 1:
            return bool(first_node[0])
        second_node = first_node[1].get(word[1])
        if second_node is None:
            return False
        if length == 2:
            return bool(second_node[0])
        return second_node[1] is not None and bool(second_node[1].get(word[2:]))


@lru_cache(maxsize=None)
def _load_dictionary(model: ModelName) -> DictionaryModel:
    with (Path(__file__).parent / "data" / f"model.{model}.pickle").open("rb") as file:
        return DictionaryModel(**load(file))


@dataclass
class HMMModel:
    __slots__ = ("emission_scores", "start_scores", "transition_scores")

    start_scores: StateScores
    transition_scores: Tuple[float, ...]
    emission_scores: Dict[str, StateScores]


@lru_cache(maxsize=None)
def _load_hmm() -> HMMModel:
    with (Path(__file__).parent / "data" / "hmm.pickle").open("rb") as file:
        return HMMModel(**load(file))


def _find_route(block: str, length: int, model: DictionaryModel) -> List[int]:
    scores = [0.0] * (length + 1)
    route = [0] * (length + 1)
    characters = list(block)
    get_first_node = model.prefix_index.get
    unknown_score = model.unknown_score
    # Scan backwards so each outgoing edge can be scored immediately.
    for start in range(length - 1, -1, -1):
        first_node = get_first_node(characters[start])
        if first_node is None:
            scores[start] = unknown_score + scores[start + 1]
            route[start] = start + 1
            continue
        word_score = first_node[0]
        best_score = word_score + scores[start + 1] if word_score else NEGATIVE_INFINITY
        best_end = start + 1 if word_score else 0
        suffix_start = start + 2
        if start + 1 < length:
            second_node = first_node[1].get(characters[start + 1])
            if second_node is not None:
                word_score = second_node[0]
                if word_score:
                    score = word_score + scores[suffix_start]
                    if score >= best_score:
                        best_score = score
                        best_end = suffix_start
                suffix_scores = second_node[1]
                if suffix_scores is not None:
                    get_suffix_score = suffix_scores.get
                    for end in range(start + 3, length + 1):
                        word_score = get_suffix_score(block[suffix_start:end])
                        if word_score is None:
                            break
                        if word_score:
                            score = word_score + scores[end]
                            if score >= best_score:
                                best_score = score
                                best_end = end
        if not best_end:
            best_end = start + 1
            best_score = unknown_score + scores[best_end]
        scores[start] = best_score
        route[start] = best_end
    return route


def _cut_block_without_hmm(block: str, model: DictionaryModel) -> Iterable[str]:
    length = len(block)
    route = _find_route(block, length, model)
    start = 0
    buffer_start = 0
    while start < length:
        end = route[start]
        word = block[start:end]
        if end - start == 1 and word in ASCII_ALPHANUMERICS:
            start = end
            continue
        if buffer_start < start:
            yield block[buffer_start:start]
        yield word
        buffer_start = start = end
    if buffer_start < length:
        yield block[buffer_start:]


def _cut_hmm_block(block: str) -> Iterable[str]:
    model = _load_hmm()
    length = len(block)
    predecessors = bytearray(length)
    get_emission_scores = model.emission_scores.get
    begin_emission, middle_emission, end_emission, single_emission = (
        get_emission_scores(block[0], UNKNOWN_EMISSION_SCORES)
    )
    begin_score, middle_score, end_score, single_score = (
        model.start_scores[0] + begin_emission,
        model.start_scores[1] + middle_emission,
        model.start_scores[2] + end_emission,
        model.start_scores[3] + single_emission,
    )
    (
        end_to_begin,
        single_to_begin,
        middle_to_middle,
        begin_to_middle,
        begin_to_end,
        middle_to_end,
        single_to_single,
        end_to_single,
    ) = model.transition_scores
    # States B/M/E/S are 0/1/2/3. Four two-bit predecessors fit in one byte.
    # Compare after adding emissions to preserve floating-point ties.
    # Equal scores select the lexicographically later predecessor, like jieba.
    for index in range(1, length):
        begin_emission, middle_emission, end_emission, single_emission = (
            get_emission_scores(block[index], UNKNOWN_EMISSION_SCORES)
        )
        first_score = end_score + end_to_begin + begin_emission
        second_score = single_score + single_to_begin + begin_emission
        if first_score > second_score:
            next_begin_score, begin_predecessor = first_score, 2
        else:
            next_begin_score, begin_predecessor = second_score, 3
        first_score = middle_score + middle_to_middle + middle_emission
        second_score = begin_score + begin_to_middle + middle_emission
        if first_score >= second_score:
            next_middle_score, middle_predecessor = first_score, 1
        else:
            next_middle_score, middle_predecessor = second_score, 0
        first_score = begin_score + begin_to_end + end_emission
        second_score = middle_score + middle_to_end + end_emission
        if first_score > second_score:
            next_end_score, end_predecessor = first_score, 0
        else:
            next_end_score, end_predecessor = second_score, 1
        first_score = single_score + single_to_single + single_emission
        second_score = end_score + end_to_single + single_emission
        if first_score >= second_score:
            next_single_score, single_predecessor = first_score, 3
        else:
            next_single_score, single_predecessor = second_score, 2
        predecessors[index] = (
            begin_predecessor
            | (middle_predecessor << 2)
            | (end_predecessor << 4)
            | (single_predecessor << 6)
        )
        begin_score, middle_score, end_score, single_score = (
            next_begin_score,
            next_middle_score,
            next_end_score,
            next_single_score,
        )
    path = bytearray(length)
    state = 2 if end_score > single_score else 3
    for index in range(length - 1, 0, -1):
        path[index] = state
        state = (predecessors[index] >> (state * 2)) & 3
    path[0] = state
    start = 0
    for index, state in enumerate(path):
        if state == 0:
            start = index
        elif state == 2:
            yield block[start : index + 1]
            start = index + 1
        elif state == 3:
            yield block[index]
            start = index + 1


def _cut_hmm(text: str) -> Iterable[str]:
    for block_index, block in enumerate(HAN_PATTERN.split(text)):
        if block_index & 1:
            yield from _cut_hmm_block(block)
        elif block:
            for word in ALPHANUMERIC_PATTERN.split(block):
                if word:
                    yield word


def _cut_block_with_hmm(block: str, model: DictionaryModel) -> Iterable[str]:
    length = len(block)
    route = _find_route(block, length, model)
    start = 0
    buffer_start = 0
    while start < length:
        end = route[start]
        if end - start == 1:
            start = end
            continue
        buffer_length = start - buffer_start
        if buffer_length == 1:
            yield block[buffer_start]
        elif buffer_length > 1:
            buffer = block[buffer_start:start]
            if not model.contains_word(buffer):
                yield from _cut_hmm(buffer)
            else:
                yield from buffer
        yield block[start:end]
        buffer_start = start = end
    buffer_length = length - buffer_start
    if buffer_length == 1:
        yield block[buffer_start]
    elif buffer_length > 1:
        buffer = block[buffer_start:]
        if not model.contains_word(buffer):
            yield from _cut_hmm(buffer)
        else:
            yield from buffer


def _cut_text(sentence: str, model: ModelName, use_hmm: bool) -> Iterable[str]:
    dictionary = _load_dictionary(model)
    if use_hmm:
        cut_block: Callable[[str, DictionaryModel], Iterable[str]] = _cut_block_with_hmm
    else:
        cut_block = _cut_block_without_hmm
    for block_index, block in enumerate(TEXT_PATTERN.split(sentence)):
        if not block:
            continue
        if block_index & 1:
            yield from cut_block(block, dictionary)
        else:
            # Outside TEXT_PATTERN, only CRLF is kept as a multi-character token.
            for span_index, span in enumerate(block.split("\r\n")):
                if span_index:
                    yield "\r\n"
                yield from span


def _cut_query(sentence: str, model: ModelName, use_hmm: bool) -> Iterable[str]:
    get_first_node = _load_dictionary(model).prefix_index.get
    for word in _cut_text(sentence, model=model, use_hmm=use_hmm):
        length = len(word)
        if length > 2:
            characters = list(word)
            for start in range(length - 1):
                first_node = get_first_node(characters[start])
                if first_node is not None:
                    second_node = first_node[1].get(characters[start + 1])
                    if second_node is not None and second_node[0]:
                        yield word[start : start + 2]
        if length > 3:
            for start in range(length - 2):
                first_node = get_first_node(characters[start])
                if first_node is not None:
                    second_node = first_node[1].get(characters[start + 1])
                    if (
                        second_node is not None
                        and second_node[1] is not None
                        and second_node[1].get(characters[start + 2])
                    ):
                        yield word[start : start + 3]
        yield word
