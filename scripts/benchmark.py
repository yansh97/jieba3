"""Compare segmentation results and warm-run performance against jieba."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from statistics import median
from subprocess import run
from sys import executable
from time import perf_counter
from typing import Callable, List, Literal, Tuple

from jieba import initialize, lcut, lcut_for_search

from jieba3 import jieba3

REPETITIONS = 10
CORPUS_DIR = Path(__file__).resolve().parents[1] / "assets" / "corpora"
HMM_MODES = ((False, "关闭 HMM"), (True, "开启 HMM"))
Tokenize = Callable[[str], List[str]]


def measure_pair(
    jieba_measure: Callable[[], float], jieba3_measure: Callable[[], float]
) -> Tuple[float, float]:
    jieba_samples: List[float] = []
    jieba3_samples: List[float] = []
    measurements = ((jieba_measure, jieba_samples), (jieba3_measure, jieba3_samples))
    for repetition in range(REPETITIONS):
        order = measurements[::-1] if repetition % 2 else measurements
        for measure, samples in order:
            samples.append(measure())
    return median(jieba_samples), median(jieba3_samples)


def measure_loading(implementation: Literal["jieba", "jieba3"], use_hmm: bool) -> float:
    # The unknown word also triggers lazy HMM loading when enabled.
    if implementation == "jieba":
        statement = f"from jieba import lcut\nlcut('杭研大厦', HMM={use_hmm!r})\n"
    else:
        statement = (
            "from jieba3 import jieba3\n"
            f"jieba3(use_hmm={use_hmm!r}).cut_text('杭研大厦')\n"
        )
    script = (
        "from time import perf_counter\n"
        "start_time = perf_counter()\n"
        f"{statement}"
        "print(perf_counter() - start_time)\n"
    )
    result = run([executable, "-c", script], check=True, capture_output=True, text=True)
    return float(result.stdout)


def benchmark_loading() -> None:
    for use_hmm, mode in HMM_MODES:
        jieba_load_time, jieba3_load_time = measure_pair(
            partial(measure_loading, "jieba", use_hmm),
            partial(measure_loading, "jieba3", use_hmm),
        )
        print(
            f"首次加载（{mode}，含导入和首次分词）: "
            f"jieba: {jieba_load_time:.3f} 秒，"
            f"jieba3: {jieba3_load_time:.3f} 秒，"
            f"加载速度比: {jieba_load_time / jieba3_load_time:.1f}x"
        )


def measure_segmentation(tokenize: Tokenize, lines: List[str]) -> float:
    start_time = perf_counter()
    for line in lines:
        tokenize(line)
    return perf_counter() - start_time


def benchmark_mode(
    lines: List[str],
    size_mib: float,
    mode: str,
    jieba_tokenize: Tokenize,
    jieba3_tokenize: Tokenize,
) -> None:
    # Validate and warm both tokenizers before timing entire corpus passes.
    for line_number, line in enumerate(lines, 1):
        assert jieba_tokenize(line) == jieba3_tokenize(line), (
            f"{mode}: 第 {line_number} 行分词结果不一致"
        )
    jieba_time, jieba3_time = measure_pair(
        partial(measure_segmentation, jieba_tokenize, lines),
        partial(measure_segmentation, jieba3_tokenize, lines),
    )
    print(
        f"测试模式: {mode}，"
        f"jieba: {jieba_time:.2f} 秒，{size_mib / jieba_time:.2f} MiB/s，"
        f"jieba3: {jieba3_time:.2f} 秒，{size_mib / jieba3_time:.2f} MiB/s，"
        f"加速比: {jieba_time / jieba3_time:.1f}x"
    )


def benchmark_dataset(corpus_path: Path) -> None:
    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    size_mib = corpus_path.stat().st_size / 1024 / 1024
    print(f"测试数据集: {corpus_path.name}")
    for use_hmm, hmm_mode in HMM_MODES:
        tokenizer = jieba3(use_hmm=use_hmm)
        for mode, jieba_tokenize, jieba3_tokenize in (
            ("文档模式", lcut, tokenizer.cut_text),
            ("查询模式", lcut_for_search, tokenizer.cut_query),
        ):
            benchmark_mode(
                lines,
                size_mib,
                f"{mode}（{hmm_mode}）",
                partial(jieba_tokenize, HMM=use_hmm),
                jieba3_tokenize,
            )


def main() -> None:
    initialize()
    benchmark_loading()
    for corpus_name in (
        "as_test.utf8",
        "cityu_test.utf8",
        "msr_test.utf8",
        "pku_test.utf8",
        "围城.utf8",
    ):
        benchmark_dataset(CORPUS_DIR / corpus_name)


if __name__ == "__main__":
    main()
