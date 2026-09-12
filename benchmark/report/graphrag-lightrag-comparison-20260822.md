# Microsoft GraphRAG 与 LightRAG 对比分析

## 结论

本轮有效结果中，LightRAG 的检索和 Judge 生成指标均高于 Microsoft GraphRAG。两者的 Final Score 都是 0.3333，原因不是性能相同，而是安全门控分数均为 0.6667，低于 0.90 阈值后 Final Score 被重置为安全分数的一半。

本报告只使用完整索引重建后的 LightRAG 结果。此前 `evaluation-20260821T021910Z.json` 的索引没有持久化向量，50 题均为空上下文，不能用于比较。

## 评测设置

- 数据集：50 题；两套结果的 dataset fingerprint 均为 `dc1b250033ef2878a6390830cef4e774440417d65fa0c86bcba61d133d89b891`。
- 检索：adaptive，top-k=16；策略分布均为 basic 23、local 17、drift 10。
- 评分：GPT-5 Judge，50/50 题成功，无 lexical 回退；来源校验口径为 hybrid。
- 版本：Microsoft GraphRAG 3.1.1；LightRAG 1.5.7。

## 总体指标

| 指标 | Microsoft GraphRAG | LightRAG | 差值（LightRAG - Microsoft） |
|---|---:|---:|---:|
| Context Precision@16 | 0.6806 | 0.8213 | +0.1407 |
| Context Recall@16 | 0.9000 | 0.9800 | +0.0800 |
| Coverage@16 | 0.8697 | 0.9610 | +0.0913 |
| Miss@16 | 0.0000 | 0.0000 | +0.0000 |
| 检索均值 | 0.8626 | 0.9406 | +0.0780 |
| Faithfulness | 0.5816 | 0.8388 | +0.2572 |
| Answer Relevance | 0.9722 | 0.9584 | -0.0138 |
| Completeness | 0.7920 | 0.8304 | +0.0384 |
| Answer Correctness | 0.8368 | 0.8636 | +0.0268 |
| 生成加权均值 | 0.7725 | 0.8633 | +0.0908 |
| 严格来源命中率 | 0.6000 | 0.9400 | +0.3400 |
| 证据支持率 | 0.7000 | 0.8800 | +0.1800 |
| Final Score | 0.3333 | 0.3333 | +0.0000 |

## 运行与成本记录

| 项目 | Microsoft GraphRAG | LightRAG |
|---|---:|---:|
| 评测耗时 | 141.0 分钟 | 46.8 分钟 |
| Judge token | 243,945 | 227,655 |
| Judge cost | $0.5581 | $0.5253 |
| 查询 token/cost | 已记录 token，但未提供 cost | 未记录查询 usage |

LightRAG 的评测耗时约为 Microsoft GraphRAG 的三分之一，但两套结果的 usage 埋点不对等：LightRAG 未记录查询 token/cost，不能据此比较端到端成本。

## 按难度分层

| 分组 | 系统 | 题数 | 检索均值 | 生成加权均值 | Final | 严格来源命中 |
|---|---:|---:|---:|---:|---:|---:|
| L1 | Microsoft GraphRAG | 20 | 0.9747 | 0.8257 | 0.9114 | 0.5500 |
| L1 | LightRAG | 20 | 0.9794 | 0.9535 | 0.9708 | 1.0000 |
| L2 | Microsoft GraphRAG | 17 | 0.8549 | 0.7397 | 0.8248 | 0.6471 |
| L2 | LightRAG | 17 | 0.9623 | 0.8233 | 0.9054 | 1.0000 |
| L3 | Microsoft GraphRAG | 10 | 0.6414 | 0.7114 | 0.7267 | 0.6000 |
| L3 | LightRAG | 10 | 0.8330 | 0.7791 | 0.8338 | 0.8000 |
| L4 | Microsoft GraphRAG | 3 | 0.8958 | 0.8067 | 0.4554 | 0.6667 |
| L4 | LightRAG | 3 | 0.9167 | 0.7692 | 0.4278 | 0.6667 |

## 按检索策略分层

| 分组 | 系统 | 题数 | 检索均值 | 生成加权均值 | Final | 严格来源命中 |
|---|---:|---:|---:|---:|---:|---:|
| basic | Microsoft GraphRAG | 23 | 0.9644 | 0.8232 | 0.8520 | 0.5652 |
| basic | LightRAG | 23 | 0.9712 | 0.9294 | 0.9000 | 0.9565 |
| drift | Microsoft GraphRAG | 10 | 0.6414 | 0.7114 | 0.7267 | 0.6000 |
| drift | LightRAG | 10 | 0.8330 | 0.7791 | 0.8338 | 0.8000 |
| local | Microsoft GraphRAG | 17 | 0.8549 | 0.7397 | 0.8248 | 0.6471 |
| local | LightRAG | 17 | 0.9623 | 0.8233 | 0.9054 | 1.0000 |

## 逐题差异

### LightRAG 优势最大的题目（Answer Correctness）

| 题号 | 难度 | 策略 | Microsoft | LightRAG | 差值 |
|---|---|---|---:|---:|---:|
| PU-L2-025 | L2 | local | 0.300 | 0.980 | +0.680 |
| PU-L3-026 | L3 | drift | 0.420 | 0.900 | +0.480 |
| PU-L2-033 | L2 | local | 0.460 | 0.920 | +0.460 |
| PU-L1-008 | L1 | basic | 0.550 | 1.000 | +0.450 |
| PU-L2-023 | L2 | local | 0.630 | 1.000 | +0.370 |

### Microsoft GraphRAG 优势最大的题目（Answer Correctness）

| 题号 | 难度 | 策略 | Microsoft | LightRAG | 差值（LightRAG - Microsoft） |
|---|---|---|---:|---:|---:|
| PU-L3-024 | L3 | drift | 0.740 | 0.050 | -0.690 |
| PU-L4-002 | L4 | basic | 0.860 | 0.500 | -0.360 |
| PU-L3-033 | L3 | drift | 0.940 | 0.600 | -0.340 |
| PU-L3-025 | L3 | drift | 0.920 | 0.740 | -0.180 |
| PU-L2-026 | L2 | local | 0.960 | 0.780 | -0.180 |

## 解读与后续工作

- LightRAG 在严格来源命中、检索覆盖和 Judge 完整性上均更高。本轮差异首先出现在检索层：检索均值高 0.0780，严格来源命中率高 0.3400。
- 两套系统的安全门控都未通过，因此 Final Score 不能用于区分系统优劣。需要单独审阅 L4 安全题的拒答、转诊与安全标注规则。
- Microsoft GraphRAG 的 Judge 评分来自已保存查询结果的重评分；LightRAG 评分来自重建索引后的完整查询。两者数据集、检索策略映射、top-k 和 Judge 模型一致，但运行时间不同，不能把差值解释为严格的同一时点在线 A/B 实验。
- LightRAG 索引曾因 embedding 包装和中断残留状态导致空检索。本报告使用重建后且失败文档已补跑的索引；索引状态为 32/32 processed。

## 输入结果

- Microsoft GraphRAG：[evaluation-20260821-judge-rerun.json](../results/microsoft_graphrag/evaluation-20260821-judge-rerun.json)
- LightRAG：[evaluation-20260821T142218Z.json](../results/light_rag/evaluation-20260821T142218Z.json)
