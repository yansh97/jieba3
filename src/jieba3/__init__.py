"""“结巴 3”中文分词：做最好的 Modern Python 3 中文分词组件"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from jieba3.tok import ModelName, _cut_query, _cut_text

__version__ = "1.1.0"


@dataclass(init=False)
class jieba3:
    __slots__ = ("model", "use_hmm")

    model: ModelName
    use_hmm: bool

    def __init__(self, model: ModelName = "base", use_hmm: bool = True) -> None:
        self.model = model
        self.use_hmm = use_hmm
        if self.model not in ("base", "small", "large"):
            raise ValueError("model must be 'base', 'small', or 'large'")
        if not isinstance(self.use_hmm, bool):
            raise TypeError("use_hmm must be a bool")

    def cut_text(self, sentence: str) -> List[str]:
        return list(_cut_text(sentence, model=self.model, use_hmm=self.use_hmm))

    def cut_query(self, sentence: str) -> List[str]:
        return list(_cut_query(sentence, model=self.model, use_hmm=self.use_hmm))
