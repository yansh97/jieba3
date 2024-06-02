# jieba3

“结巴 3”中文分词：做最好的 Modern Python 3 中文分词组件

# 与 jieba 的区别

jieba3 是 [jieba](https://github.com/fxsjy/jieba) 分词模块的 Modern Python 3 重构版本

- 删除 Python 2 兼容代码，支持 type hints 等 Modern Python 3 特性
- 重构分词模块，在纯 Python 实现前提下，提高约 **20%** 的性能，且与 jieba 分词结果对齐
- 暂不支持除分词外的其他 jieba 功能，如关键词提取、词性标注等

# 安装说明

jieba3 仅支持 Python 3.10+ 版本

```bash
pip install jieba3
```

# 算法

- 基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图 (DAG)
- 采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合
- 对于未登录词，采用了基于汉字成词能力的 HMM 模型，使用了 Viterbi 算法

# 主要功能

构建 `jieba3.jieba3` 分词器实例，支持以下参数：

- `model: Literal["base", "small", "large"] = "base"`
  - 分词模型选项，可选值为 `small`、`base`、`large`，默认为 `base`
  - `base` 模型是 jieba 提供的默认模型
  - `small` 模型是 jieba 提供的占用内存较小的模型
  - `large` 模型是 jieba 支持繁体分词更好的模型
- `use_hmm: bool = True`
  - 是否开启 HMM 新词发现，可选值为 `True`、`False`，默认为 `True`

示例如下：

```python
import jieba3

tokenizer = jieba3.jieba3()  # 默认为 base 模型，开启 HMM 新词发现
tokenizer = jieba3.jieba3(model="small")  # 使用 small 模型
tokenizer = jieba3.jieba3(model="base")  # 使用 base 模型
tokenizer = jieba3.jieba3(model="large")  # 使用 large 模型
tokenizer = jieba3.jieba3(use_hmm=False)  # 关闭 HMM 新词发现
tokenizer = jieba3.jieba3(use_hmm=True)  # 开启 HMM 新词发现
```

## 文档模式

试图将句子最精确地切开，适合文档分析

> 当使用默认的 `base` 模型时，jieba3 文档模式与 jieba 精确模式的分词结果完全一致

```python
import jieba3
import jieba

# 开启 HMM 新词发现

tokenizer = jieba3.jieba3()
tokenizer.cut_text("小明硕士毕业于中国科学院计算所")
# ["小明", "硕士", "毕业", "于", "中国科学院", "计算所"]

jieba.lcut("小明硕士毕业于中国科学院计算所")
# ["小明", "硕士", "毕业", "于", "中国科学院", "计算所"]

# 关闭 HMM 新词发现

tokenizer = jieba3.jieba3(use_hmm=False)
tokenizer.cut_text("小明硕士毕业于中国科学院计算所")
# ["小", "明", "硕士", "毕业", "于", "中国科学院", "计算所"]

jieba.lcut("小明硕士毕业于中国科学院计算所", HMM=False)
# ["小", "明", "硕士", "毕业", "于", "中国科学院", "计算所"]
```

## 查询模式

在文档模式的基础上，对长词再次切分，提高召回率，适合查询分析

> 当使用默认的 `base` 模型时，jieba3 查询模式与 jieba 搜索引擎模式的分词结果完全一致

```python
import jieba3
import jieba

# 开启 HMM 新词发现

tokenizer = jieba3.jieba3()
tokenizer.cut_query("小明硕士毕业于中国科学院计算所")
# ["小明", "硕士", "毕业", "于", "中国", "科学", "学院", "科学院", "中国科学院", "计算", "计算所"]

jieba.lcut_for_search("小明硕士毕业于中国科学院计算所")
# ["小明", "硕士", "毕业", "于", "中国", "科学", "学院", "科学院", "中国科学院", "计算", "计算所"]

# 关闭 HMM 新词发现

tokenizer = jieba3.jieba3(use_hmm=False)
tokenizer.cut_query("小明硕士毕业于中国科学院计算所")
# ["小", "明", "硕士", "毕业", "于", "中国", "科学", "学院", "科学院", "中国科学院", "计算", "计算所"]

jieba.lcut_for_search("小明硕士毕业于中国科学院计算所", HMM=False)
# ["小", "明", "硕士", "毕业", "于", "中国", "科学", "学院", "科学院", "中国科学院", "计算", "计算所"]
```

# 性能测试

jieba3 均使用默认的 `base` 模型，与 jieba 的默认模型对比

测试环境：MacBookPro18,3，macOS 14.5，Apple M1 Pro @ 3.20 GHz，16 GB

## SIGHAN Bakeoff 2005 测试集（逐行分词）

### `as_test.utf8`（繁体）

| 模式                 | jieba 耗时 | jieba 速度 | jieba3 耗时 | jieba3 速度 | 性能提升 |
| -------------------- | ---------- | ---------- | ----------- | ----------- | -------- |
| 文档模式（关闭 HMM） | 0.26 秒    | 2.28 MB/s  | 0.20 秒     | 2.94 MB/s   | 22%      |
| 文档模式（开启 HMM） | 0.60 秒    | 0.98 MB/s  | 0.48 秒     | 1.23 MB/s   | 20%      |
| 查询模式（关闭 HMM） | 0.27 秒    | 2.17 MB/s  | 0.21 秒     | 2.79 MB/s   | 22%      |
| 查询模式（开启 HMM） | 0.63 秒    | 0.93 MB/s  | 0.51 秒     | 1.15 MB/s   | 20%      |

### `cityu_test.utf8`（繁体）

| 模式                 | jieba 耗时 | jieba 速度 | jieba3 耗时 | jieba3 速度 | 性能提升 |
| -------------------- | ---------- | ---------- | ----------- | ----------- | -------- |
| 文档模式（关闭 HMM） | 0.09 秒    | 2.22 MB/s  | 0.07 秒     | 2.87 MB/s   | 23%      |
| 文档模式（开启 HMM） | 0.21 秒    | 0.93 MB/s  | 0.17 秒     | 1.16 MB/s   | 20%      |
| 查询模式（关闭 HMM） | 0.09 秒    | 2.11 MB/s  | 0.07 秒     | 2.71 MB/s   | 22%      |
| 查询模式（开启 HMM） | 0.21 秒    | 0.90 MB/s  | 0.17 秒     | 1.12 MB/s   | 20%      |

### `msr_test.utf8`（简体）

| 模式                 | jieba 耗时 | jieba 速度 | jieba3 耗时 | jieba3 速度 | 性能提升 |
| -------------------- | ---------- | ---------- | ----------- | ----------- | -------- |
| 文档模式（关闭 HMM） | 0.26 秒    | 2.06 MB/s  | 0.20 秒     | 2.69 MB/s   | 24%      |
| 文档模式（开启 HMM） | 0.30 秒    | 1.79 MB/s  | 0.24 秒     | 2.25 MB/s   | 20%      |
| 查询模式（关闭 HMM） | 0.28 秒    | 1.91 MB/s  | 0.22 秒     | 2.47 MB/s   | 23%      |
| 查询模式（开启 HMM） | 0.32 秒    | 1.67 MB/s  | 0.26 秒     | 2.08 MB/s   | 20%      |

### `pku_test.utf8`（简体）

| 模式                 | jieba 耗时 | jieba 速度 | jieba3 耗时 | jieba3 速度 | 性能提升 |
| -------------------- | ---------- | ---------- | ----------- | ----------- | -------- |
| 文档模式（关闭 HMM） | 0.25 秒    | 1.91 MB/s  | 0.20 秒     | 2.48 MB/s   | 23%      |
| 文档模式（开启 HMM） | 0.30 秒    | 1.64 MB/s  | 0.24 秒     | 2.04 MB/s   | 20%      |
| 查询模式（关闭 HMM） | 0.26 秒    | 1.85 MB/s  | 0.20 秒     | 2.41 MB/s   | 23%      |
| 查询模式（开启 HMM） | 0.33 秒    | 1.48 MB/s  | 0.27 秒     | 1.82 MB/s   | 19%      |

## 《围城》（全文分词）

| 模式                 | jieba 耗时 | jieba 速度 | jieba3 耗时 | jieba3 速度 | 性能提升 |
| -------------------- | ---------- | ---------- | ----------- | ----------- | -------- |
| 文档模式（关闭 HMM） | 0.35 秒    | 1.85 MB/s  | 0.28 秒     | 2.32 MB/s   | 20%      |
| 文档模式（开启 HMM） | 0.51 秒    | 1.25 MB/s  | 0.42 秒     | 1.52 MB/s   | 18%      |
| 查询模式（关闭 HMM） | 0.33 秒    | 1.93 MB/s  | 0.26 秒     | 2.45 MB/s   | 21%      |
| 查询模式（开启 HMM） | 0.55 秒    | 1.17 MB/s  | 0.45 秒     | 1.42 MB/s   | 18%      |
