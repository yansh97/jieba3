# jieba3

“结巴 3”中文分词：做最好的 Modern Python 3 中文分词组件

## 1.1.0 更新

相较于 [1.0.2](https://pypi.org/project/jieba3/1.0.2/)：

- **精简依赖**：移除 pydantic，运行时仅依赖 Python 标准库。
- **兼容增强**：最低版本从 Python 3.10 降至 Python 3.8，支持 Python 3.8+。
- **加载速度**：默认配置下，首次加载速度为上一版本的 **5.8x**。
- **分词速度**：五份语料合计，文档模式为上一版本的 **2.0x**，查询模式为 **1.9x**。

以上速度均在同一环境下测量，使用默认 `base` 模型并开启 HMM。

# 与 jieba 的区别

jieba3 是 [jieba](https://github.com/fxsjy/jieba) 分词模块的 Modern Python 3 重构版本

- 删除 Python 2 兼容代码，支持 type hints 等 Modern Python 3 特性
- 纯 Python 实现，本次基准中分词速度为 jieba 的 **2.2–3.0x**
- 所有模型均与 jieba 使用对应词典及相同 HMM 设置时的分词结果对齐
- 暂不支持除分词外的其他 jieba 功能，如关键词提取、词性标注等

# 安装说明

jieba3 支持 Python 3.8+ 版本

运行时仅依赖 Python 标准库，无需额外依赖。

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
  - `base` 对应 jieba 默认词典 `dict.txt`
  - `small` 对应 jieba 的 `dict.txt.small`，占用内存较小
  - `large` 对应 jieba 的 `dict.txt.big`，对繁体分词支持更好
- `use_hmm: bool = True`
  - 是否开启 HMM 新词发现，可选值为 `True`、`False`，默认为 `True`

`use_hmm` 需传入布尔值，不会自动转换字符串或数字。

与 jieba 比较时需选择对应词典并保持 HMM 设置一致；不同词典之间的分词结果可能不同。

示例如下：

```python
from jieba3 import jieba3

tokenizer = jieba3()  # 默认为 base 模型，开启 HMM 新词发现
tokenizer = jieba3(model="small")  # 使用 small 模型
tokenizer = jieba3(model="base")  # 使用 base 模型
tokenizer = jieba3(model="large")  # 使用 large 模型
tokenizer = jieba3(use_hmm=False)  # 关闭 HMM 新词发现
tokenizer = jieba3(use_hmm=True)  # 开启 HMM 新词发现
```

## 文档模式

试图将句子最精确地切开，适合文档分析

> 使用对应词典和相同 HMM 设置时，三个模型的文档模式均与 jieba 精确模式的分词结果对齐

```python
from jieba import lcut
from jieba3 import jieba3

# 开启 HMM 新词发现

tokenizer = jieba3()
tokenizer.cut_text("小明硕士毕业于中国科学院计算所")
# ["小明", "硕士", "毕业", "于", "中国科学院", "计算所"]

lcut("小明硕士毕业于中国科学院计算所")
# ["小明", "硕士", "毕业", "于", "中国科学院", "计算所"]

# 关闭 HMM 新词发现

tokenizer = jieba3(use_hmm=False)
tokenizer.cut_text("小明硕士毕业于中国科学院计算所")
# ["小", "明", "硕士", "毕业", "于", "中国科学院", "计算所"]

lcut("小明硕士毕业于中国科学院计算所", HMM=False)
# ["小", "明", "硕士", "毕业", "于", "中国科学院", "计算所"]
```

## 查询模式

在文档模式的基础上，对长词再次切分，提高召回率，适合查询分析

> 使用对应词典和相同 HMM 设置时，三个模型的查询模式均与 jieba 搜索引擎模式的分词结果对齐

```python
from jieba import lcut_for_search
from jieba3 import jieba3

# 开启 HMM 新词发现

tokenizer = jieba3()
tokenizer.cut_query("小明硕士毕业于中国科学院计算所")
# ["小明", "硕士", "毕业", "于", "中国", "科学", "学院", "科学院", "中国科学院", "计算", "计算所"]

lcut_for_search("小明硕士毕业于中国科学院计算所")
# ["小明", "硕士", "毕业", "于", "中国", "科学", "学院", "科学院", "中国科学院", "计算", "计算所"]

# 关闭 HMM 新词发现

tokenizer = jieba3(use_hmm=False)
tokenizer.cut_query("小明硕士毕业于中国科学院计算所")
# ["小", "明", "硕士", "毕业", "于", "中国", "科学", "学院", "科学院", "中国科学院", "计算", "计算所"]

lcut_for_search("小明硕士毕业于中国科学院计算所", HMM=False)
# ["小", "明", "硕士", "毕业", "于", "中国", "科学", "学院", "科学院", "中国科学院", "计算", "计算所"]
```

# 性能测试

使用 `base` 模型与 jieba 默认词典，覆盖文档、查询模式及 HMM 开关组合，五份语料均逐行分词并校验结果一致。

## 测试环境与方法

测试环境：Apple M4 / 24 GB 内存，macOS 15.8.3，CPython 3.14.7；jieba3 1.1.0、jieba 0.42.1。测试日期：2026-11-14。

每种配置测量 10 次，轮换执行顺序，取中位数。分词计时前完成预热，计时不包含语料读取、模型首次加载和结果校验；吞吐量按原文件字节数换算为 MiB/s。

首次加载在新进程中测量，从模块导入计时至首次分词完成，包含 HMM 的首次加载；jieba 使用预先生成的默认磁盘缓存。

速度比为“jieba 耗时 ÷ jieba3 耗时”，由未舍入的耗时计算并保留一位小数。[语料来源](assets/corpora/README.md)与[原始采样数据](assets/benchmarks/1.1.0.json)可用于核对结果。

## 与 jieba 对比

### 首次加载

| 配置 | jieba 耗时 | jieba3 耗时 | 加载速度比 |
| --- | --- | --- | --- |
| 关闭 HMM | 0.247 秒 | 0.053 秒 | 4.7x |
| 开启 HMM | 0.254 秒 | 0.056 秒 | 4.6x |

### `as_test.utf8`（繁体）

| 模式 | jieba 耗时 | jieba 速度 | jieba3 耗时 | jieba3 速度 | 分词速度比 |
| --- | --- | --- | --- | --- | --- |
| 文档模式（关闭 HMM） | 0.153 秒 | 3.84 MiB/s | 0.061 秒 | 9.71 MiB/s | 2.5x |
| 文档模式（开启 HMM） | 0.345 秒 | 1.71 MiB/s | 0.116 秒 | 5.08 MiB/s | 3.0x |
| 查询模式（关闭 HMM） | 0.150 秒 | 3.93 MiB/s | 0.065 秒 | 9.14 MiB/s | 2.3x |
| 查询模式（开启 HMM） | 0.351 秒 | 1.68 MiB/s | 0.124 秒 | 4.74 MiB/s | 2.8x |

### `cityu_test.utf8`（繁体）

| 模式 | jieba 耗时 | jieba 速度 | jieba3 耗时 | jieba3 速度 | 分词速度比 |
| --- | --- | --- | --- | --- | --- |
| 文档模式（关闭 HMM） | 0.053 秒 | 3.64 MiB/s | 0.020 秒 | 9.39 MiB/s | 2.6x |
| 文档模式（开启 HMM） | 0.113 秒 | 1.71 MiB/s | 0.038 秒 | 5.09 MiB/s | 3.0x |
| 查询模式（关闭 HMM） | 0.054 秒 | 3.58 MiB/s | 0.022 秒 | 8.65 MiB/s | 2.4x |
| 查询模式（开启 HMM） | 0.114 秒 | 1.68 MiB/s | 0.040 秒 | 4.75 MiB/s | 2.8x |

### `msr_test.utf8`（简体）

| 模式 | jieba 耗时 | jieba 速度 | jieba3 耗时 | jieba3 速度 | 分词速度比 |
| --- | --- | --- | --- | --- | --- |
| 文档模式（关闭 HMM） | 0.143 秒 | 3.73 MiB/s | 0.060 秒 | 8.86 MiB/s | 2.4x |
| 文档模式（开启 HMM） | 0.161 秒 | 3.31 MiB/s | 0.069 秒 | 7.76 MiB/s | 2.3x |
| 查询模式（关闭 HMM） | 0.161 秒 | 3.32 MiB/s | 0.073 秒 | 7.31 MiB/s | 2.2x |
| 查询模式（开启 HMM） | 0.172 秒 | 3.11 MiB/s | 0.079 秒 | 6.75 MiB/s | 2.2x |

### `pku_test.utf8`（简体）

| 模式 | jieba 耗时 | jieba 速度 | jieba3 耗时 | jieba3 速度 | 分词速度比 |
| --- | --- | --- | --- | --- | --- |
| 文档模式（关闭 HMM） | 0.139 秒 | 3.51 MiB/s | 0.059 秒 | 8.29 MiB/s | 2.4x |
| 文档模式（开启 HMM） | 0.157 秒 | 3.09 MiB/s | 0.066 秒 | 7.36 MiB/s | 2.4x |
| 查询模式（关闭 HMM） | 0.143 秒 | 3.40 MiB/s | 0.066 秒 | 7.38 MiB/s | 2.2x |
| 查询模式（开启 HMM） | 0.171 秒 | 2.84 MiB/s | 0.077 秒 | 6.30 MiB/s | 2.2x |

### 《围城》（简体）

| 模式 | jieba 耗时 | jieba 速度 | jieba3 耗时 | jieba3 速度 | 分词速度比 |
| --- | --- | --- | --- | --- | --- |
| 文档模式（关闭 HMM） | 0.190 秒 | 3.68 MiB/s | 0.082 秒 | 8.57 MiB/s | 2.3x |
| 文档模式（开启 HMM） | 0.284 秒 | 2.46 MiB/s | 0.111 秒 | 6.32 MiB/s | 2.6x |
| 查询模式（关闭 HMM） | 0.191 秒 | 3.66 MiB/s | 0.085 秒 | 8.24 MiB/s | 2.3x |
| 查询模式（开启 HMM） | 0.300 秒 | 2.33 MiB/s | 0.123 秒 | 5.68 MiB/s | 2.4x |
