# KAG 补跑失败修复与验证（2026-09-12）

KAG / PU-L3-PRE009 修复后补跑完成，进程耗时 113.595 秒（查询 89.225 秒，逐题含判分 110.231 秒）。五框架新增 12 题的当前答案覆盖为 **60/60**。历史 900 秒、1800 秒超时记录均保留，没有覆盖失败尝试。

本次解决的是 KAG 请求失败后的异常处理和缺失题补跑，不代表已解决增量评测揭示的全部临床行为、评分或证据提取问题。

## 代码变更

变更位于项目维护的 `benchmark/baseline/kag_client/provider_guard.py`，由 `client.py` 初始化并包裹求解调用；没有修改忽略管理的 vendor 文件或重建索引。

- KAG 的 OpenAI / Azure SDK 请求默认采用 120 秒 HTTP 超时，SDK 最多重试 2 次（总计最多 3 次尝试），复用 SDK 对临时错误的退避和 Retry-After 处理；永久错误不通过 SDK 重试。
- 同步请求最多 1 路、异步请求最多 1 路，分别限流，避免同步调用阻塞事件循环时与异步请求争用同一锁。保留用户配置的异步 limiter。此限制仅作用于本进程，不能解除服务端限流。
- 向量请求失败直接抛出异常，不再返回 None；验证响应索引、数量、非空数值向量、有限值和批内维度一致性。异步单字符串按一个文本处理，拒绝空白输入，不以虚构向量代替证据。
- 查询级失败状态可中止吞掉底层异常后仍继续运行的求解任务，并阻止本查询后续 SDK 请求；最终版本同时覆盖 vendor 线程池未传播 ContextVar 的情况。已开始的同步网络请求仍受 HTTP 超时约束，无法由 asyncio 立即强制取消。
- NER 对非列表或非字典条目、错误字段类型作校验，记录警告并跳过不可用项，避免字符串触发 `.get()` 崩溃。跳过不等于识别正确，仍可能减少召回。

HTTP 超时不是整题总耗时保证；补跑进程仍有独立的 1800 秒上限。本次保持串行运行，未声称修复共享配置的跨进程并发竞争。

## 验证

`tests/baseline/test_kag_provider_guard.py`、`test_kag_client.py`、`test_kag_benchmark.py` 共 43 项通过。使用真实 OpenAI SDK 和 HTTP mock 验证 429 后成功、429 耗尽仅 3 次、401 仅 1 次、异常被上层吞掉后的中止、线程池故障传播、并发及取消后释放、异常向量和 NER 格式。

[测试日志](../results/preclinical-20260912-fixed/tests.log)；[在线向量探测](../results/preclinical-20260912-fixed/embedding_probe.json)返回 `text-embedding-3-large` 的 3072 维向量，用时 1.324 秒，与配置维度相符，不计入问答样本。

问答进程运行的是[修复快照](../results/preclinical-20260912-fixed/provider_guard.snapshot.py)。运行期间随后补强了普通线程池故障传播，最终保留配置 limiter，均经离线测试；在线向量探测已覆盖线程池补强版本。版本哈希和差异记录在[合并记录](../results/preclinical-20260912-fixed/merge_record.json)，不把不同代码版本声称为同一次运行。

## 答案和证据限制

新答案明确拒绝仅凭年份淘汰旧指南，建议核查适用性、更新与撤回状态，并由本地专家或临床管理团队复核。独立 gpt-5 行为复核未触发禁止项；原 gpt-5 正确性评分 0.95。

**该题落盘的 contexts 和 retrieved_sources 都为空，faithfulness 为 0。** 本次成功证明产出了可评分答案，不能证明检索证据链恢复、回答受指南原文支持或临床可用。查询计量记录 3 次模型调用且没有失败；运行日志未发现 429、空向量 TypeError 或 NER AttributeError。未保存足以还原全部规划决策的轨迹，不能把本次提速全部归因于传输修复；模型规划和服务状态也可能变化。

题目、语料指纹、模型、top-k、Judge 及评分配置与原记录核对一致，**运行时重试与并发策略有变化**。答案哈希与独立行为复核绑定后，仅替换当前汇总的 PRE009 失败占位。当前汇总覆盖和条件校验通过。

- [运行状态](../results/preclinical-20260912-fixed/retry.status.json)
- [答案、上下文和评分](../results/preclinical-20260912-fixed/kag_PRE009.jsonl)
- [独立行为复核](../results/preclinical-20260912-fixed/behavior_audit.json)
- [合并前快照](../results/preclinical-20260912-fixed/before_retry/preclinical-incremental-20260911.md)
- [最新增量评估报告](preclinical-incremental-20260911.md)

上线前仍须处理 PRE007 无依据的跨指南排序、证据缺失、评分截断等已发现问题，并完成临床专家审核和真实跨指南冲突验证。
