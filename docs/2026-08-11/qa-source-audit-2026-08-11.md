# QA 数据集来源核查记录（2026-08-11）

## 范围与结论

本次核查覆盖英文规范 QA 集 `benchmark/qa/dataset/sample_questions.json` 的 10 条样例。原中文样例集已删除；核查以指南原文、监管机构页面和可公开访问的文献为优先来源。

核查后的数据集做了以下安全性修订：

- 将增加 NT 后的路径明确为：**提供遗传咨询和诊断性检测选择**；cfDNA 是筛查，不能与 CVS/羊膜腔穿刺等诊断性检测等同。
- 以 NT ≥3.5 mm 作为详细胎儿心脏评估/胎儿超声心动图的明确转诊指征。
- 将四腔心切面中的“房室间隔完整性”修订为心脏十字、房间隔/室间隔和房室瓣正常偏移；删除未经指南支持的“可筛查约 60% 先心病”表述，并明确完整筛查还需要流出道、三血管和三血管气管切面。
- 对 CPAM 与肺隔离症加入混合病变警示；将异常体循环供血动脉表述为**支持**肺隔离症诊断，而不是单独“确诊”。CVR >1.6 是水肿风险分层指标，不是确定性结论。
- 修正开放性脊柱裂征象的关系方向：柠檬头征/香蕉小脑征提示开放性脊柱裂，而不是反向关系。
- 修正经侧脑室平面：测量对象为侧脑室房部，丘脑是经丘脑平面的标志；>15 mm 应称为重度脑室增宽，不能直接等同于脑积水。
- 将 ISUOG 2023 的**最低要求**与**详细检查**分开：鼻骨、完整四腔心、流出道和三血管气管切面属于详细评估中可显示的结构，不应被标为最低要求。
- 将商业 4D 胎儿写真安全题改为依据 ALARA 和 FDA 对非医学“keepsake”超声的劝阻，不再使用无法直接核实的绝对化组织立场。

## 已核验的一手或权威来源

| 标识 | 来源 | 关键核查范围 |
|---|---|---|
| `ISUOG-11-14w-2023` | Chaoui R, et al. *ISUOG Practice Guidelines (updated): performance of 11–14-week ultrasound scan*. 2023. [DOI](https://doi.org/10.1002/uog.26106)；[开放获取 PDF](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/uog.26106) | CRL 45–84 mm、NT 正中矢状面/中立位、最低与详细早孕期结构检查、增加 NT 的局限 |
| `ISUOG-fetal-cardiac-screening-2023` | Carvalho JS, et al. *ISUOG Practice Guidelines (updated): fetal cardiac screening*. 2023. [DOI](https://doi.org/10.1002/uog.26224)；[开放获取 PDF](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/uog.26224) | 四腔心切面要素、流出道/3VV/3VTV、NT ≥3.5 mm 的胎儿超声心动图指征 |
| `ISUOG-midtrimester-2022` | Salomon LJ, et al. *ISUOG Practice Guidelines (updated): performance of the routine mid-trimester fetal ultrasound scan*. 2022. [DOI](https://doi.org/10.1002/uog.24888) | 中孕期常规结构筛查范围 |
| `ISUOG-cns-2020` | Malinger G, et al. *ISUOG Practice Guidelines (updated): sonographic examination of the fetal CNS. Part 1*. 2020. [DOI](https://doi.org/10.1002/uog.22145)；[PubMed](https://pubmed.ncbi.nlm.nih.gov/32870591/) | CNS 筛查平面及靶向神经超声适应证 |
| `FDA-ultrasound-imaging` | U.S. FDA, [Ultrasound Imaging](https://www.fda.gov/radiation-emitting-products/medical-imaging/ultrasound-imaging) | ALARA、超声的潜在热/机械生物效应、对非医学 keepsake 超声的劝阻 |
| `PMC-3410507` | Sfakianaki AK, Copel JA. *Congenital cystic lesions of the lung*. 2012. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3410507/) | CPAM/肺隔离症鉴别及混合病变 |
| `PubMed-24258515` | Zhang H, et al. *Retrospective study of prenatal diagnosed pulmonary sequestration*. 2014. [PubMed](https://pubmed.ncbi.nlm.nih.gov/24258515/) | 异常体循环供血动脉、CVR 与水肿风险、序贯随访 |

Semantic Scholar 已用于交叉核对前两份 ISUOG 2023 指南的标题、年份、DOI、PubMed ID、期刊信息和开放获取地址。其返回的文献元数据不替代指南原文。

指南 PDF 英文原文已归档至 `data/en-guidelines/uog/`（2023 早孕期、2023 胎儿心脏筛查、2022 中孕期），详见该目录 `README.md`。CNS 2020 Part 1 因 Wiley Cloudflare 及仓库限制未能从本网络下载，需通过 DOI 浏览器获取。

## 扩充至 50 条（2026-08-12）

在原 10 条基础上新增 40 条英文题目，总数达到 50 条，难度分布约为 L1 40%、L2 34%、L3 20%、L4 6%。新增题目主要覆盖：

- 中孕期常规生物测量（BPD/HC/AC/FL、EFW、DVP、肾盂、膀胱、胎盘）。
- 早孕期与中孕期心脏筛查（四腔心、流出道、3VV/3VTV、心轴、心律）。
- 早孕期非整倍体超声标记（鼻骨、静脉导管、NT 解释）。
- 中孕期结构筛查与转诊路径（腹壁、胎盘植入谱系、FGR、脊柱与后颅窝）。
- 胸廓病变、开放性脊柱裂与孤立性轻度脑室增宽的鉴别与咨询。
- 非医学超声/家用多普勒的安全咨询（FDA）。

新增 40 条均以 `REVIEW-A`/`REVIEW-B` 标记为 `pending`，`agreement_kappa` 为 `null`，**不构成已完成的双盲专家标注**。

## 本地指南来源的限制

`China-screening-2022` 的样例中原有的具体“必查/可选/非强制”比较结论，未能在本次联网核查中取得可公开核对的原始发布文本。因此：

1. 该 ID 在 `GUIDELINE_REGISTRY` 中标记为 `primary_text_not_independently_verified`。
2. 跨指南题不再把这些二手摘要当作确定性合规要求。
3. 上线或临床使用前，必须由产前超声专科医师按**当地现行、正式发布的原文**复核，并重新完成双盲标注和仲裁。

## 仍需人工完成的事项

本次为来源审计和数据扩充，不构成临床诊疗建议，也不替代专科医师审核。原 10 条的 `gold_answer`、`must_have_statements` 和关系三元组已修订，需要至少两名领域专家重新独立标注；新增 40 条为待审草稿，`REVIEW-A`/`REVIEW-B` 的 `pending` 标注不构成验证。完成双盲标注、仲裁并计算一致性指标前，不应把本数据集视为终版评测集。
