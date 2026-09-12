# 三框架对比：Microsoft GraphRAG vs LightRAG vs KAG（2026-09-08）

统一口径：新评分框架（安全惩罚仅在逐题层应用一次；L4 行为由 gpt-5
`judge_safety` 判定并回填进冻结产物），用 `rescore_results` 对三份 50 题
评测产物重算。生成/判分模型一致（gpt-5-mini 生成、gpt-5 判分、
text-embedding-3-large 向量）。KAG 为修复 logic-form 解析后的两轮
（run1/run2 显示 dataset 级方差 ±0.003，排序不受影响）。

- GraphRAG: `evaluation-20260821-judge-rerun`（drift/local/basic）
- LightRAG: `evaluation-20260821T142218Z`（hybrid → local/basic）
- KAG: `evaluation-20260907T140957Z` + `20260908T032034Z`（solver/naive）

## 数据集级

| 框架 | recall@k | precision@k | miss@k | 生成加权 | 正确性 | 安全 | **final** |
|---|---|---|---|---|---|---|---|
| GraphRAG | 0.900 | 0.681 | 0.000 | 0.772 | 0.837 | 1.000 | **0.843** |
| LightRAG | **0.980** | 0.821 | 0.000 | **0.863** | **0.864** | 1.000 | **0.915** |
| KAG | 0.940 | **0.866** | 0.020 | 0.834 | 0.863 | 0.667* | **0.882** |

\* KAG 的 PU-L4-002（家用 Doppler "reassure" 表述）被判真实违规；
GraphRAG/LightRAG 的 L4 三题均通过。final 差距（0.843/0.882/0.915）
远超 dataset 级噪声带（±0.003），排序可信。

## 分架构类型

| 类型 | 框架 | recall@k | precision@k | faithfulness | final |
|---|---|---|---|---|---|
| basic (n=23) | GraphRAG | 1.000 | 0.894 | 0.610 | 0.906 |
| | LightRAG | 1.000 | 0.910 | **0.909** | **0.957** |
| | KAG | 0.957 | 0.926 | 0.797 | 0.891 |
| multi-vector (n=17) | GraphRAG | 0.941 | 0.544 | 0.721 | 0.825 |
| | LightRAG | 1.000 | 0.849 | 0.764 | 0.905 |
| | KAG | 1.000 | **0.912** | **0.835** | **0.908** |
| graph-enhanced (n=10) | GraphRAG | 0.600 | 0.422 | **0.279** | 0.727 |
| | LightRAG | **0.900** | 0.569 | **0.806** | **0.834** |
| | KAG | 0.800 | **0.648** | 0.575 | 0.818 |

注：KAG basic recall 0.957 受 PU-L4-001~003 的 gold 源
`FDA-ultrasound-imaging` 缺失限制；graph-enhanced 三框架的
PU-L3-024/027 gold 源（PMC/PubMed）均不在语料，recall 上限被锁。

## 成本与可靠性（50 题）

| 框架 | LLM 请求数 | tokens | 成本 | 耗时 | 失败 |
|---|---|---|---|---|---|
| GraphRAG | 1410（每题 ~28） | 1.04M | $0.56 | 141 min | 0 |
| LightRAG | **50** | 0.23M | $0.53 | **47 min** | 0 |
| KAG | 227–243 | 1.9–2.6M | $0.53–0.55 | 64–68 min | 0 |

## 结论

1. **LightRAG 综合最优**：检索（0.98 recall）、生成（0.863）双高，
   成本与耗时最低（单请求混合检索）。实体/关系/chunk 三路 hybrid
   与指南类语料结构高度匹配。
2. **KAG 次之**：precision 全场最高（0.866），multi-vector 类与
   LightRAG 持平（0.908 vs 0.905）；短板是 solver 多步生成的
   faithfulness 偏低（0.575）与 token 成本最高（多步 rewrite）。
   差异化的 schema 化推理路径尚未启用（当前 schema-free 模式）。
3. **GraphRAG 第三**：drift 路径 faithfulness 崩塌（0.279）——答案
   主要来自预生成的社区摘要而非检索证据，与"检索支撑"口径脱节；
   且每题 ~28 个 LLM 请求导致 141 分钟最慢。basic 路径本身不差
   （0.906）。

排序对缺源题敏感度：语料补齐 `FDA-ultrasound-imaging` 与
PMC/PubMed 两篇后，KAG 的 basic/graph-enhanced recall 有 +0.05~
+0.1 的上行空间，final 差距可能收窄但不改变当前排序（LightRAG
在所有 arch 类型上无短板）。
