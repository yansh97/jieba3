"""Compile reference JSON models into portable, ready-to-load pickle data."""

from __future__ import annotations

from collections import defaultdict
from json import loads
from math import log
from pathlib import Path
from pickle import dump
from typing import DefaultDict, Dict, Set

from jieba3.tok import MIN_LOG_PROBABILITY, StateScores, UnigramNode

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "assets" / "models"
OUTPUT_DIR = PROJECT_DIR / "src" / "jieba3" / "data"


def build_dictionary(source_path: Path) -> Dict[str, object]:
    source = loads(source_path.read_text(encoding="utf-8"))
    total_log_frequency = log(source["total"])
    frequencies = source["freq"]
    shared_strings: Dict[str, str] = {}
    suffix_index: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
    prefix_index: Dict[str, UnigramNode] = {}
    for word, frequency in frequencies.items():
        length = len(word)
        if length == 2:
            continue
        # Integer zero is a shared sentinel for prefixes that are not words.
        score = log(frequency) - total_log_frequency if frequency else 0
        if length == 1:
            prefix_index[shared_strings.setdefault(word, word)] = (score, {})
        elif length > 2:
            prefix = word[:2]
            suffix = word[2:]
            suffix_index[prefix][shared_strings.setdefault(suffix, suffix)] = score
    for word, frequency in frequencies.items():
        if len(word) == 2:
            score = log(frequency) - total_log_frequency if frequency else 0
            first_character, second_character = word
            prefix_index[first_character][1][
                shared_strings.setdefault(second_character, second_character)
            ] = (score, suffix_index.get(word))
    return {"prefix_index": prefix_index, "unknown_score": -total_log_frequency}


def build_hmm(source_path: Path) -> Dict[str, object]:
    source = loads(source_path.read_text(encoding="utf-8"))
    state_scores = source["state_prob"]
    character_scores = source["char_prob"]
    state_transition_scores = source["trans_prob"]
    start_scores = tuple(state_scores[state] for state in ("B", "M", "E", "S"))
    transition_scores = tuple(
        state_transition_scores[previous_state][current_state]
        for previous_state, current_state in (
            ("E", "B"),
            ("S", "B"),
            ("M", "M"),
            ("B", "M"),
            ("B", "E"),
            ("M", "E"),
            ("S", "S"),
            ("E", "S"),
        )
    )
    characters: Set[str] = set().union(*character_scores.values())
    emission_scores: Dict[str, StateScores] = {
        character: (
            character_scores["B"].get(character, MIN_LOG_PROBABILITY),
            character_scores["M"].get(character, MIN_LOG_PROBABILITY),
            character_scores["E"].get(character, MIN_LOG_PROBABILITY),
            character_scores["S"].get(character, MIN_LOG_PROBABILITY),
        )
        for character in sorted(characters)
    }
    return {
        "start_scores": start_scores,
        "transition_scores": transition_scores,
        "emission_scores": emission_scores,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    builders = [
        (f"model.{model}", build_dictionary) for model in ("base", "small", "large")
    ]
    builders.append(("hmm", build_hmm))
    for name, build_model in builders:
        output_path = OUTPUT_DIR / f"{name}.pickle"
        with output_path.open("wb") as file:
            dump(build_model(SOURCE_DIR / f"{name}.json"), file, protocol=4)
        print(
            f"{output_path.relative_to(PROJECT_DIR)}: {output_path.stat().st_size:,} bytes"
        )


if __name__ == "__main__":
    main()
