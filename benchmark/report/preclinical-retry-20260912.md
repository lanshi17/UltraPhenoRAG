# KAG 缺失题补跑（2026-09-12）

PU-L3-PRE009 补跑仍未成功：等待上限由 900 秒放宽到 1800 秒，实际运行 1800.2 秒后超时，未生成可评分答案。汇总保持 59/60 份成功答案，该题继续单列为失败。

日志记录到 10 个不同请求 ID 的 HTTP 429 限流错误，影响问题改写、关系检索与向量化；向量化失败后还出现了空向量导致的 `TypeError: must be real number, not NoneType`。本次未记录到上轮的 `AttributeError` 型 NER 解析异常。限流是已观察到的运行问题，不能据此断定全部耗时均由限流造成，也不能把无答案视为医学错误或安全通过。

未切换模型、重建索引或更换检索策略。此次无成功答案，因此没有新增正确性或安全判分。原 900 秒超时记录和补跑前报告均保留。

- [完整运行日志](../results/preclinical-20260912/kag_PRE009.log)
- [命令、超时与耗时](../results/preclinical-20260912/retry.status.json)
- [运行诊断](../results/preclinical-20260912/diagnostics.json)
- [更新后的增量评估报告](preclinical-incremental-20260911.md)
- [补跑前报告快照](../results/preclinical-20260912/before_retry/preclinical-incremental-20260911.md)

下一次尝试应先排查模型服务限流与调用重试策略；单纯重复放宽超时尚未补齐该题。
