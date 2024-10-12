"""“结巴 3”中文分词：做最好的 Modern Python 3 中文分词组件"""

from typing import Literal

from pydantic import BaseModel

from jieba3.tok import _cut_query, _cut_text

__version__ = "1.0.2"


class jieba3(BaseModel):
    model: Literal["base", "small", "large"] = "base"
    use_hmm: bool = True

    def cut_text(self, sentence: str, /) -> list[str]:
        return list(_cut_text(sentence, model=self.model, use_hmm=self.use_hmm))

    def cut_query(self, sentence: str, /) -> list[str]:
        return list(_cut_query(sentence, model=self.model, use_hmm=self.use_hmm))
