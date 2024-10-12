import re
from math import log
from pathlib import Path
from typing import Callable, Final, Iterable, Literal, TypeAlias, get_args

from pydantic import BaseModel

NEG_INF: Final[float] = float("-inf")
LARGE_NEG: Final[float] = -3.14e100

PURE_ENG: Final[str] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
RE_ENG: Final[re.Pattern[str]] = re.compile(pattern=r"([a-zA-Z0-9]+(?:\.\d+)?%?)")
RE_PURE_HAN: Final[re.Pattern[str]] = re.compile(pattern=r"([\u4E00-\u9FD5]+)")
RE_HAN: Final[re.Pattern[str]] = re.compile(
    pattern=r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)"
)
RE_SKIP: Final[re.Pattern[str]] = re.compile(pattern=r"(\r\n|\s)")


class ModelParams(BaseModel):
    freq: dict[str, int]
    total: int


BASE_MODEL_PARAMS: Final[ModelParams] = ModelParams.model_validate_json(
    json_data=(Path(__file__).parent / "model.base.json").read_text(encoding="utf-8")
)
BASE_MODEL_FREQ: Final[dict[str, int]] = BASE_MODEL_PARAMS.freq
BASE_MODEL_TOTAL: Final[int] = BASE_MODEL_PARAMS.total

SMALL_MODEL_PARAMS: Final[ModelParams] = ModelParams.model_validate_json(
    json_data=(Path(__file__).parent / "model.small.json").read_text(encoding="utf-8")
)
SMALL_MODEL_FREQ: Final[dict[str, int]] = SMALL_MODEL_PARAMS.freq
SMALL_MODEL_TOTAL: Final[int] = SMALL_MODEL_PARAMS.total

LARGE_MODEL_PARAMS: Final[ModelParams] = ModelParams.model_validate_json(
    json_data=(Path(__file__).parent / "model.large.json").read_text(encoding="utf-8")
)
LARGE_MODEL_FREQ: Final[dict[str, int]] = LARGE_MODEL_PARAMS.freq
LARGE_MODEL_TOTAL: Final[int] = LARGE_MODEL_PARAMS.total


State: TypeAlias = Literal["B", "M", "E", "S"]
STATES: tuple[State, ...] = get_args(tp=State)


class HMMParams(BaseModel):
    state_prob: dict[State, float]
    char_prob: dict[State, dict[str, float]]
    trans_prob: dict[State, dict[State, float]]
    prev_states: dict[State, tuple[State, ...]]


HMM_PARAMS: Final[HMMParams] = HMMParams.model_validate_json(
    json_data=(Path(__file__).parent / "hmm.json").read_text(encoding="utf-8")
)

HMM_STATE_PROB: Final[dict[State, float]] = HMM_PARAMS.state_prob
HMM_CHAR_PROB: Final[dict[State, dict[str, float]]] = HMM_PARAMS.char_prob
HMM_TRANS_PROB: Final[dict[State, dict[State, float]]] = HMM_PARAMS.trans_prob
HMM_PREV_STATES: Final[dict[State, tuple[State, ...]]] = HMM_PARAMS.prev_states


def _dp(
    block: str, n: int, MODEL_FREQ: dict[str, int], MODEL_TOTAL: int, /
) -> list[int]:
    dag: list[list[tuple[int, int]]] = [None] * n  # type: ignore
    for i in range(n):
        indexes: list[tuple[int, int]] = []
        for j in range(i + 1, n + 1):
            freq: int | None = MODEL_FREQ.get(block[i:j])
            if freq is None:
                break
            if freq > 0:
                indexes.append((j, freq))
        if len(indexes) == 0:
            indexes.append((i + 1, 1))
        dag[i] = indexes
    probs: list[float] = [0] * (n + 1)
    routes: list[int] = [0] * (n + 1)
    log_total: float = log(MODEL_TOTAL)
    for i in range(n - 1, -1, -1):
        max_prob: float = NEG_INF
        for j, freq in dag[i]:
            prob: float = log(freq) - log_total + probs[j]
            if prob >= max_prob:
                max_prob = prob
                probs[i] = max_prob
                routes[i] = j
    return routes


def _cut_block_without_hmm(
    block: str, MODEL_FREQ: dict[str, int], MODEL_TOTAL: int, /
) -> Iterable[str]:
    n: int = len(block)
    route: list[int] = _dp(block, n, MODEL_FREQ, MODEL_TOTAL)
    i: int = 0
    buffer: str = ""
    while i < n:
        j: int = route[i]
        span: str = block[i:j]
        if j - i == 1 and span in PURE_ENG:
            buffer += span
            i = j
            continue
        if buffer:
            yield buffer
            buffer = ""
        yield span
        i = j
    if buffer:
        yield buffer
        buffer = ""


def _hmm_cut_block(block: str, /) -> Iterable[str]:
    V: list[dict[State, float]] = [{}]
    paths: dict[State, list[State]] = {}
    for s in STATES:
        V[0][s] = HMM_STATE_PROB[s] + HMM_CHAR_PROB[s].get(block[0], LARGE_NEG)
        paths[s] = [s]
    for j in range(1, len(block)):
        V.append({})
        cur_paths: dict[State, list[State]] = {}
        for s in STATES:
            chat_prob: float = HMM_CHAR_PROB[s].get(block[j], LARGE_NEG)
            max_prob: float = NEG_INF
            max_state: State = "B"
            for s0 in HMM_PREV_STATES[s]:
                prob: float = (
                    V[j - 1][s0] + HMM_TRANS_PROB[s0].get(s, LARGE_NEG) + chat_prob
                )
                if prob > max_prob:
                    max_prob = prob
                    max_state = s0
                elif prob == max_prob:
                    max_state = max(max_state, s0)
            V[j][s] = max_prob
            cur_paths[s] = paths[max_state] + [s]
        paths = cur_paths
    path: list[State] = paths["E" if V[-1]["E"] > V[-1]["S"] else "S"]
    i: int = 0
    for j, char in enumerate(iterable=block):
        state = path[j]
        if state == "B":
            i = j
        elif state == "E":
            yield block[i : j + 1]
            i = j + 1
        elif state == "S":
            yield char
            i = j + 1


def _hmm_cut(buffer: str, /) -> Iterable[str]:
    for block in RE_PURE_HAN.split(string=buffer):
        if RE_PURE_HAN.match(string=block):
            yield from _hmm_cut_block(block)
        else:
            yield from (x for x in RE_ENG.split(string=block) if x)


def _cut_block_with_hmm(
    block: str, MODEL_FREQ: dict[str, int], MODEL_TOTAL: int, /
) -> Iterable[str]:
    n: int = len(block)
    route: list[int] = _dp(block, n, MODEL_FREQ, MODEL_TOTAL)
    i: int = 0
    buffer: str = ""
    while i < n:
        j: int = route[i]
        span: str = block[i:j]
        if j - i == 1:
            buffer += span
            i = j
            continue
        m: int = len(buffer)
        if m == 1:
            yield buffer
        elif m > 1:
            if not MODEL_FREQ.get(buffer):
                words: Iterable[str] = _hmm_cut(buffer)
                for word in words:
                    yield word
            else:
                yield from buffer
        buffer = ""
        yield span
        i = j
    m = len(buffer)
    if m == 1:
        yield buffer
    elif m > 1:
        if not MODEL_FREQ.get(buffer):
            words = _hmm_cut(buffer)
            for word in words:
                yield word
        else:
            yield from buffer


def _cut_text(
    sentence: str, /, *, model: Literal["base", "small", "large"], use_hmm: bool
) -> Iterable[str]:
    match model:
        case "base":
            MODEL_FREQ: dict[str, int] = BASE_MODEL_FREQ
            MODEL_TOTAL: int = BASE_MODEL_TOTAL
        case "small":
            MODEL_FREQ = SMALL_MODEL_FREQ
            MODEL_TOTAL = SMALL_MODEL_TOTAL
        case "large":
            MODEL_FREQ = LARGE_MODEL_FREQ
            MODEL_TOTAL = LARGE_MODEL_TOTAL
    if use_hmm:
        cut_block: Callable[[str, dict[str, int], int], Iterable[str]] = (
            _cut_block_with_hmm
        )
    else:
        cut_block = _cut_block_without_hmm
    blocks: list[str] = RE_HAN.split(string=sentence)
    for block in blocks:
        if not block:
            continue
        if RE_HAN.match(string=block):
            for word in cut_block(block, MODEL_FREQ, MODEL_TOTAL):
                yield word
        else:
            spans: list[str] = RE_SKIP.split(string=block)
            for span in spans:
                if not span:
                    continue
                if RE_SKIP.match(string=span):
                    yield span
                else:
                    yield from span


def _cut_query(
    sentence: str, /, *, model: Literal["base", "small", "large"], use_hmm: bool
) -> Iterable[str]:
    match model:
        case "base":
            MODEL_FREQ: dict[str, int] = BASE_MODEL_FREQ
        case "small":
            MODEL_FREQ = SMALL_MODEL_FREQ
        case "large":
            MODEL_FREQ = LARGE_MODEL_FREQ
    words: Iterable[str] = _cut_text(sentence, model=model, use_hmm=use_hmm)
    for w in words:
        n: int = len(w)
        if n > 2:
            for i in range(n - 1):
                gram2: str = w[i : i + 2]
                if MODEL_FREQ.get(gram2):
                    yield gram2
        if n > 3:
            for i in range(n - 2):
                gram3: str = w[i : i + 3]
                if MODEL_FREQ.get(gram3):
                    yield gram3
        yield w
