# Prenatal GraphRAG Benchmark

使用 Microsoft GraphRAG、LightRAG、PathRAG 和本地 OpenSPG KAG 对产前超声指南
语料建库，并用 `benchmark/qa/dataset/sample_questions.json` 执行离线基线评测。

## 代码架构

```
benchmark/
├── .env                  # 统一 LLM 供应商配置（RAG_* 变量，五个基线共用）
├── common/               # 五个基线共享的评测逻辑（不重复造轮子）
│   ├── corpus.py         # PDF 提取、来源 ID 映射、manifest 写入
│   ├── unified_corpus.py # 统一语料目录（data/corpus）提取与读取
│   ├── usage.py          # token/成本统计聚合
│   ├── answers.py        # 回答归一化、拒答/转诊检测、语句支持判定
│   ├── retrieval.py      # 检索上下文抽取（DataFrame/list 双兼容）
│   └── scoring_options.py# 来源校验策略（scoring_config.yaml）
├── config/               # 框架无关的环境变量解析
├── qa/                   # 数据集 schema、三层评分、LLM-as-Judge
├── baseline/
│   ├── microsoft_graphrag_client/   # Microsoft GraphRAG 客户端 + 评测入口
│   ├── light_rag_client/            # LightRAG 客户端 + 评测入口
│   ├── pathrag_client/              # PathRAG 客户端 + 评测入口
│   ├── kag_client/                  # 本地 OpenSPG KAG 客户端 + 评测入口
│   └── hippo_rag_client/            # HippoRAG 2 客户端 + 评测入口
├── data/
│   ├── corpus/           # ★ 统一输入语料：input/ + corpus_manifest.json
│   ├── microsoft_graphrag/  # GraphRAG 专属索引产物（output/、cache/）
│   ├── light_rag/           # LightRAG 专属索引产物（rag_storage/）
│   ├── path_rag/            # PathRAG 专属索引产物（pathrag_storage/）
│   ├── kag/                 # KAG 专属索引产物（kag_storage/ckpt/）
│   └── hipporag/            # HippoRAG 专属索引产物（KG、OpenIE、embedding）
└── results/              # 评测结果（按基线分子目录）
tests/                    # pytest 测试（与源码结构对应）
├── conftest.py           # sys.path 与 vendored LightRAG/PathRAG/KAG 引导
├── test_common_*.py      # 共享模块单元测试
└── baseline/             # 各基线的行为/冒烟测试
```

### 统一输入输出

- **输入统一**：`benchmark/data/corpus/input/` 是唯一语料目录。`prepare`
  对四个基线均提取 PDF 到此处；Microsoft GraphRAG 通过
  `settings.yaml` 的 `input_storage.base_dir` 指向该目录，LightRAG、PathRAG 与 KAG
  在索引时自动回退到该目录（各基线自己的 `input/` 若存在则优先）。
- **清单统一**：`data/corpus/corpus_manifest.json` 为唯一来源清单，
  评测时的 `SourceResolver` 均从统一清单解析。
- **输出统一**：评测结果统一写 `benchmark/results/<baseline>/`，
  向量化产物写 `benchmark/data/proceed/`，索引产物留在各基线目录。

运行测试：`uv run --with pytest python -m pytest tests/ -v`

## 运行

```bash
uv sync

# 1. 提取 benchmark/data/raw 下的 PDF，并初始化 GraphRAG 项目
uv run python -m benchmark.baseline.microsoft_graphrag_client.benchmark prepare

# 2. 在生成的文件中配置模型密钥
# benchmark/.env
# RAG_API_KEY=<your-api-key>

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

模型与供应商统一由 `benchmark/.env` 解析。Microsoft GraphRAG 的
`settings.yaml` 与 KAG 运行时生成的 `kag_config.yaml` 都从环境变量读取密钥；
所有基准均只读取 `benchmark/.env`，`.env.example` 为不含密钥的模板。

```dotenv
# .env
# benchmark/.env（Microsoft GraphRAG、LightRAG、PathRAG 与 KAG 共用）
RAG_API_KEY=<your-api-key>
RAG_API_BASE=https://your-gateway.example.com/v1   # OpenAI 兼容端点通常含 /v1

# 使用 Azure OpenAI 时：
# RAG_MODEL_PROVIDER=azure
# RAG_API_BASE=https://<resource>.openai.azure.com
# RAG_API_VERSION=2024-10-21
```

- 默认 completion 模型 `gpt-4.1`、embedding 模型 `text-embedding-3-large`
  （`settings.yaml` 中引用 `${RAG_API_KEY}` / `${RAG_API_BASE}`）。
- 切换模型或供应商时，可在命令行**运行时覆盖**。KAG 会在运行目录生成非敏感模型
  设置，但密钥始终只保留为环境变量引用：

```bash
# index / evaluate / run 通用：
# --model <completion模型>  --embedding-model <向量模型>
# --model-provider <openai|azure>  --api-base <端点>
# --api-version <Azure API 版本>  --api-key-env <环境变量名>
uv run python -m benchmark.baseline.microsoft_graphrag_client.benchmark evaluate \
    --model gpt-4o-mini \
    --embedding-model text-embedding-3-small
```

`vectorize` 支持 `--embedding-model / --model-provider / --api-base /
--api-key-env` 覆盖向量模型。

### 自定义 `.env` 环境变量名

默认使用 `RAG_API_KEY` / `RAG_API_BASE` / `RAG_API_VERSION` /
`RAG_COMPLETION_MODEL` / `RAG_EMBEDDING_MODEL`。Microsoft GraphRAG、LightRAG、
PathRAG 与 KAG 不再使用各自独立的环境变量：

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
    --corpus-dir benchmark/data/corpus \
    --output-dir benchmark/data/proceed
```

输出 `corpus_vectors.parquet` / `corpus_vectors.jsonl` /
`vectorize_manifest.json`（支持增量续传，可自定义分块与批量大小）。

## 数据约束

现有 50 题中安全题仅 3 题、跨指南题仅 1 题，结果仅支持技术选型，不能证明临床可用。
新增独立挑战草案包含 6 道安全题和 6 道跨指南题，均待专家审核；来源核实、隐藏验收集和上线判定要求见
[上线前评测与审核方案](benchmark/qa/dataset/PRECLINICAL_REVIEW.md)。
2026-09-12 补跑后，五框架增量评估已有 60 份成功答案（历史超时记录保留），发现跨指南无依据排序及评分流程问题；
详见[增量评估报告](benchmark/report/preclinical-incremental-20260911.md)。
2026-09-12 首次补跑仍超时（见[历史记录](benchmark/report/preclinical-retry-20260912.md)）；修复 KAG 请求与向量异常处理后，第 9 题已完成，但检索上下文为空，不能据此认定临床可用。见[修复验证记录](benchmark/report/preclinical-fix-20260912.md)。

- 当前 33 份 PDF 去重后得到 32 份 GraphRAG 输入文档。
- 原始语料覆盖 50 道题中的 45 道；缺少
  `FDA-ultrasound-imaging`、`PMC-3410507`、`PubMed-24258515`
  （对应 PU-L4-001~003 与 PU-L3-024/027 的 recall 上限被语料锁定）。
- 数据集中 40 道扩充题仍为 `pending` 标注，L3/L4 结果需要医学专家复核。
- L4 安全扩展集 `sample_questions_l4_safety.json`（6 题，全部锚定语料内来源）
  为独立数据集，避免改动 `sample_questions.json` 的 dataset fingerprint
  破坏历史产物的 rescore；运行 `evaluate --dataset` 指定即可。
- lexical 指标是回归 baseline；Judge 开启后 selected 指标才使用 LLM-as-Judge，
  两者均保留，不能替代 L3/L4 医学专家复核或临床有效性结论。

评分口径（2026-09-08 起）：安全惩罚只在逐题层应用一次，数据集总分 =
逐题均值（此前 dataset 层的 `mean_safety < 0.90` 二次折半已移除，单题
短语缺失不再使总分在 0.87/0.42 间跳变）；L4 行为判定（violates_flags /
requires_referral / referred / refused）由 `judge_safety` 用 gpt-5 判定并
冻结进评测产物，短语词典仅作 fallback；`rescore_results` 会自动恢复
行内 verdict，旧产物无该字段时回退词典口径。

## LightRAG 基线

仓库内置的 LightRAG 版本位于 `benchmark/baseline/libs/light_rag`，客户端和评测入口位于
`benchmark/baseline/light_rag_client`。配置统一的 `RAG_MODEL_PROVIDER`、`RAG_COMPLETION_MODEL`、
`RAG_EMBEDDING_MODEL`、`RAG_API_BASE`、`RAG_API_KEY` 和
`RAG_EMBEDDING_DIMENSION` 后运行：

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

## PathRAG 基线

仓库内置的 PathRAG 版本位于 `benchmark/baseline/libs/path_rag`（BUPT-GAMMA/PathRAG
checkout），客户端和评测入口位于 `benchmark/baseline/pathrag_client`，索引产物写入
`benchmark/data/path_rag/pathrag_storage/`。配置与其他基线相同的 `RAG_MODEL_PROVIDER`、
`RAG_COMPLETION_MODEL`、`RAG_EMBEDDING_MODEL`、`RAG_API_BASE`、`RAG_API_KEY` 和
`RAG_EMBEDDING_DIMENSION` 后运行：

```bash
uv run python -m benchmark.baseline.pathrag_client.benchmark prepare
uv run python -m benchmark.baseline.pathrag_client.benchmark preflight
uv run python -m benchmark.baseline.pathrag_client.benchmark index
uv run python -m benchmark.baseline.pathrag_client.benchmark evaluate

# 或在配置密钥后一次完成 prepare、index 和 evaluate：
uv run python -m benchmark.baseline.pathrag_client.benchmark run
```

PathRAG 引擎只暴露一条 hybrid 检索链路（local 与 global 关键词并行构图检索），
`evaluate --method adaptive` 下所有 `rag_arch_type` 均映射到 `hybrid`，与显式
`--method hybrid` 等价。`index` 默认复用已有存储；PathRAG 按内容哈希去重，
`index --no-cache`（或 `run --no-cache`）会删除 KV/向量/图存储文件后完全重建。
`PathRAGClient` 支持注入 `llm_model_func` 和 `embedding_func`，便于离线集成测试。
PathRAG 检索结果只给出原始 chunk 文本（无文件级出处），`SourceResolver` 先用
chunk 内嵌的 `SOURCE_ID:` 头，再回退到统一语料文本的包含匹配。

## KAG 基线

仓库内置的 KAG 版本位于 `benchmark/baseline/libs/kag`，客户端和评测入口位于
`benchmark/baseline/kag_client`。KAG 复用统一语料目录，并把本地内存图的持久化
checkpoint 写入 `benchmark/data/kag/kag_storage/ckpt/MemoryGraphWriter`；该模式不需要
单独启动 OpenSPG 服务。配置与其他基线相同的 `RAG_MODEL_PROVIDER`、
`RAG_COMPLETION_MODEL`、`RAG_EMBEDDING_MODEL`、`RAG_API_BASE`、`RAG_API_KEY` 和
`RAG_EMBEDDING_DIMENSION` 后运行；Azure OpenAI 还需要设置 `RAG_API_VERSION`：

```bash
uv run python -m benchmark.baseline.kag_client.benchmark prepare
uv run python -m benchmark.baseline.kag_client.benchmark preflight
uv run python -m benchmark.baseline.kag_client.benchmark index
uv run python -m benchmark.baseline.kag_client.benchmark evaluate

# 或在配置密钥后一次完成 prepare、index 和 evaluate：
uv run python -m benchmark.baseline.kag_client.benchmark run
```

`cache` 默认复用完全相同的语料与建库配置；客户端会在文档内容、分块、workspace、
LLM 或向量模型变化时自动重建 checkpoint。若需要强制完全重建，可用
`index --no-cache`（或 `run --no-cache`）。

KAG 的 `adaptive` 评测会将 `basic` 映射为基础向量检索（`naive` 的公开别名），
将 `multi-vector` 映射为 `naive`，将 `graph-enhanced` 映射为逻辑形式 `solver`。
结果写入 `benchmark/results/kag/`，每题同时增量写入 JSONL；来源解析使用统一
`corpus_manifest.json`，将 KAG 的 `<文件名>_split_N` chunk 标题映射回稳定来源 ID。

## HippoRAG 2 基线

仓库内置的 HippoRAG 2 版本位于 `benchmark/baseline/libs/HippoRAG`
（OSU-NLP-Group/HippoRAG checkout），客户端和评测入口位于
`benchmark/baseline/hippo_rag_client`，索引产物（知识图、OpenIE 结果、
chunk/entity/fact embedding 与 LLM 缓存）写入 `benchmark/data/hipporag/`。
配置与其他基线相同的 `RAG_COMPLETION_MODEL`、`RAG_EMBEDDING_MODEL`、
`RAG_API_BASE`、`RAG_API_KEY` 后运行：

```bash
uv run python -m benchmark.baseline.hippo_rag_client.benchmark preflight
uv run python -m benchmark.baseline.hippo_rag_client.benchmark index
uv run python -m benchmark.baseline.hippo_rag_client.benchmark evaluate
```

说明：

* `index` 复用统一语料 `data/corpus/input/`，客户端按约 1200 token
  （段落优先）切分并携带 manifest `source_id`（HippoRAG 默认每文档一个
  chunk，长指南文档不适用）；加载时剥离 `## Page N` 页码标记并过滤
  小于 50 字符的碎片（当前语料切出 742 chunk / 去重后 688）。引擎按
  内容哈希去重，OpenIE 结果与 LLM 响应持久化在 save_dir，中断后重跑
  `index` 即增量续建；若索引处于半完成态（embedding 已有、图缺失），
  用 `index --force` 显式授权重建图结构（LLM 缓存仍可命中）。
* HippoRAG 只有一条检索链路（事实检索 → 识别记忆 → 个性化 PageRank），
  `evaluate` 对所有 `rag_arch_type` 统一使用 `hipporag` 方法。
* 检索保持框架原样；答案生成使用固定的临床引用 prompt（与生成隔离
  实验一致），因为 HippoRAG 自带 QA prompt 是 Wikipedia CoT 模板，
  不适用于指南引用场景。
* gpt-5-mini 等推理模型经 OpenAI 兼容端点做 OpenIE 时，默认的 512/2048
  输出预算会被推理 token 耗尽导致空响应；客户端已将预算提升为
  `HIPPO_NER_MAX_TOKENS=8192` / `HIPPO_TRIPLE_MAX_TOKENS=16384`，并对 OpenIE
  响应做形状校验：空响应退避重试、裸 JSON 数组重包装、不可解析响应以
  `store=True` 绕过缓存重新采样，彻底失败时该 chunk 降级为空实体列表，
  保证批量抽取不再整体中断。
* 结果写入 `benchmark/results/hippo_rag/`，JSONL 逐题增量落盘；
  L4 行为由 `judge_safety` 判定，`safety_verdict` 随行持久化，
  与其他基线共用统一评分器。
