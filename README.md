# Prenatal GraphRAG Benchmark

使用 Microsoft GraphRAG 对产前超声指南语料建库，并用
`benchmark/qa/dataset/sample_questions.json` 执行离线基线评测。

## 运行

```bash
uv sync

# 1. 提取 benchmark/data/raw 下的 PDF，并初始化 GraphRAG 项目
uv run python -m benchmark.baseline.microsoft_graphrag_client.benchmark prepare

# 2. 在生成的文件中配置模型密钥
# benchmark/data/microsoft_graphrag/.env
# GRAPHRAG_API_KEY=<your-api-key>

# 3. 不发起远程请求的本地预检
uv run python -m benchmark.baseline.microsoft_graphrag_client.benchmark preflight

# 4. 构建知识图谱及向量索引
uv run python -m benchmark.baseline.microsoft_graphrag_client.benchmark index

# 5. 逐题查询并生成评测报告
uv run python -m benchmark.baseline.microsoft_graphrag_client.benchmark evaluate
```

也可在配置密钥后一次运行全部阶段：

```bash
uv run python -m benchmark.baseline.microsoft_graphrag_client.benchmark run
```

默认评测模式为 `adaptive`：`basic` 题使用基础向量检索，
`multi-vector` 题使用 local search，`graph-enhanced` 题使用 DRIFT search。
结果写入 `benchmark/results/microsoft_graphrag/`，每题同时增量写入 JSONL。

## 数据约束

- 当前 33 份 PDF 去重后得到 32 份 GraphRAG 输入文档。
- 原始语料覆盖 50 道题中的 45 道；缺少
  `FDA-ultrasound-imaging`、`PMC-3410507`、`PubMed-24258515`。
- 数据集中 40 道扩充题仍为 `pending` 标注，L3/L4 结果需要医学专家复核。
- 自动生成指标使用 `benchmark/qa/scoring.py` 的词法近似算法，不等同于
  LLM-as-Judge 或临床有效性结论。
