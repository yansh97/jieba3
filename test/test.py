import time
from functools import partial
from pathlib import Path
from typing import Callable

import jieba
import jieba3

jieba.initialize()
jieba3_w_h = jieba3.jieba3()
jieba3_wo_h = jieba3.jieba3(use_hmm=False)


def timeit(
    callable: Callable[[str], list[str]], text: str, /
) -> tuple[list[str], float]:
    start: float = time.perf_counter()
    result: list[str] = callable(text)
    end: float = time.perf_counter()
    return result, end - start


def test_mode(
    path: Path,
    mode: str,
    jieba_callable: Callable[[str], list[str]],
    jieba3_callable: Callable[[str], list[str]],
    /,
) -> None:
    jieba_total_time: float = 0
    jieba3_total_time: float = 0
    size: float = path.stat().st_size / 1024 / 1024
    lines: list[str] = path.read_text().splitlines()
    N = 10
    for _ in range(N):
        for line in lines:
            jieba_result, jieba_time = timeit(jieba_callable, line)
            jieba3_result, jieba3_time = timeit(jieba3_callable, line)
            assert jieba_result == jieba3_result
            jieba_total_time += jieba_time
            jieba3_total_time += jieba3_time
    jieba_total_time = jieba_total_time / N
    jieba3_total_time = jieba3_total_time / N
    print(
        f"测试模式: {mode}，"
        f"jieba: {jieba_total_time:.2f} 秒，{size / jieba_total_time:.2f} MB/s，"
        f"jieba3: {jieba3_total_time:.2f} 秒，{size / jieba3_total_time:.2f} MB/s，"
        f"加速比: {1-jieba3_total_time / jieba_total_time:.2f}"
    )


def test_dataset(path: Path, /) -> None:
    print(f"测试数据集: {path.name}")
    test_mode(
        path,
        f"文档模式（关闭 HMM）",
        partial(jieba.lcut, HMM=False),
        jieba3_wo_h.cut_text,
    )
    test_mode(path, f"文档模式（开启 HMM）", jieba.lcut, jieba3_w_h.cut_text)
    test_mode(
        path,
        f"查询模式（关闭 HMM）",
        partial(jieba.lcut_for_search, HMM=False),
        jieba3_wo_h.cut_query,
    )
    test_mode(
        path, f"查询模式（开启 HMM）", jieba.lcut_for_search, jieba3_w_h.cut_query
    )


if __name__ == "__main__":
    test_dataset(Path("test/data/as_test.utf8"))
    test_dataset(Path("test/data/cityu_test.utf8"))
    test_dataset(Path("test/data/msr_test.utf8"))
    test_dataset(Path("test/data/pku_test.utf8"))
    test_dataset(Path("test/data/围城.utf8"))
