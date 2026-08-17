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

默认只使用 lexical baseline，不会额外调用 Judge。需要启用医学语义评审时，
显式指定 Judge 模型和密钥环境变量：

```bash
uv run python -m benchmark.baseline.microsoft_graphrag_client.benchmark evaluate \
    --judge-mode optional \
    --judge-model gpt-5 \
    --judge-api-key-env JUDGE_API_KEY
```

每题 JSONL 会同时写入 lexical/Judge 评分、`telemetry` 事件以及
completion/embedding/Judge 的 prompt、completion、total tokens 和成本；供应商未返回
价格时成本保持 `null`。来源校验默认使用 `hybrid`，保留 exact 命中并可在
`benchmark/qa/dataset/scoring_config.yaml` 配置权威来源等价组。

也可在配置密钥后一次运行全部阶段：

```bash
uv run python -m benchmark.baseline.microsoft_graphrag_client.benchmark run
```

默认评测模式为 `adaptive`：`basic` 题使用基础向量检索，
`multi-vector` 题使用 local search，`graph-enhanced` 题使用 DRIFT search。
结果写入 `benchmark/results/microsoft_graphrag/`，每题同时增量写入 JSONL。

## LLM 模型配置

模型与供应商在 `benchmark/data/microsoft_graphrag/.env` + `settings.yaml`
中配置。`.env.example` 为不含密钥的模板。

```dotenv
# .env
GRAPHRAG_API_KEY=<your-api-key>
GRAPHRAG_API_BASE=https://your-gateway.example.com/v1   # 必须含 /v1
```

- 默认 completion 模型 `gpt-4.1`、embedding 模型 `text-embedding-3-large`
  （`settings.yaml` 中引用 `${GRAPHRAG_API_KEY}` / `${GRAPHRAG_API_BASE}`）。
- 切换模型或供应商时，可在命令行**运行时覆盖**（不改写 `settings.yaml`）：

```bash
# index / evaluate / run 通用：
# --model <completion模型>  --embedding-model <向量模型>
# --model-provider <openai|azure>  --api-base <含/v1>
# --api-key-env <环境变量名>
uv run python -m benchmark.baseline.microsoft_graphrag_client.benchmark evaluate \
    --model gpt-4o-mini \
    --embedding-model text-embedding-3-small
```

`vectorize` 支持 `--embedding-model / --model-provider / --api-base /
--api-key-env` 覆盖向量模型。

### 自定义 `.env` 环境变量名

默认使用 `GRAPHRAG_API_KEY` / `GRAPHRAG_API_BASE` /
`GRAPHRAG_COMPLETION_MODEL` / `GRAPHRAG_EMBEDDING_MODEL`，均可改为自定义
变量名（如 `MY_API_KEY`、`MY_MODEL`），completion 与 embedding 也可分别使用
不同变量：

```bash
# 持久化到 settings.yaml（在 .env 中定义对应变量即可）：
uv run python -m benchmark.baseline.microsoft_graphrag_client.benchmark prepare \
    --model-env MY_COMPLETION_MODEL \
    --embedding-model-env MY_EMBEDDING_MODEL \   # 缺省沿用 --model-env
    --api-key-env MY_API_KEY \
    --api-base-env MY_API_BASE \
    --embedding-api-key-env MY_EMBED_KEY \       # 缺省沿用 --api-key-env
    --embedding-api-base-env MY_EMBED_BASE       # 缺省沿用 --api-base-env

# 或运行时覆盖（不改写配置）：
uv run python -m benchmark.baseline.microsoft_graphrag_client.benchmark evaluate \
    --model-env MY_COMPLETION_MODEL --api-key-env MY_API_KEY
```

对应的 `.env`：

```dotenv
MY_COMPLETION_MODEL=gpt-4.1
MY_EMBEDDING_MODEL=text-embedding-3-large
MY_API_KEY=<your-api-key>
MY_API_BASE=https://your-gateway.example.com/v1
MY_EMBED_KEY=<your-embedding-key>
MY_EMBED_BASE=https://your-gateway.example.com/v1
```

## 语料向量化

```bash
uv run python -m benchmark.baseline.microsoft_graphrag_client.vectorize \
    --project-dir benchmark/data/microsoft_graphrag \
    --output-dir benchmark/data/proceed
```

输出 `corpus_vectors.parquet` / `corpus_vectors.jsonl` /
`vectorize_manifest.json`（支持增量续传，可自定义分块与批量大小）。

## 数据约束

- 当前 33 份 PDF 去重后得到 32 份 GraphRAG 输入文档。
- 原始语料覆盖 50 道题中的 45 道；缺少
  `FDA-ultrasound-imaging`、`PMC-3410507`、`PubMed-24258515`。
- 数据集中 40 道扩充题仍为 `pending` 标注，L3/L4 结果需要医学专家复核。
- lexical 指标是回归 baseline；Judge 开启后 selected 指标才使用 LLM-as-Judge，
  两者均保留，不能替代 L3/L4 医学专家复核或临床有效性结论。

## LightRAG 基线

仓库内置的 LightRAG 版本位于 `benchmark/baseline/libs/light_rag`，客户端和评测入口位于
`benchmark/baseline/light_rag_client`。配置 `LLM_BINDING`、`LLM_MODEL`、
`LLM_BINDING_HOST`、`LLM_BINDING_API_KEY`、`EMBEDDING_MODEL` 和 `EMBEDDING_DIM` 后运行：

```bash
uv run python \
  -m benchmark.baseline.light_rag_client.benchmark prepare
uv run python \
  -m benchmark.baseline.light_rag_client.benchmark index
uv run python \
  -m benchmark.baseline.light_rag_client.benchmark evaluate
```

`LightRAGClient` 支持注入 `llm_model_func` 和 `embedding_func`，便于离线集成测试。
`drift_search` 映射到 LightRAG 的 `hybrid` 检索，返回统一的 `QueryResult`。
