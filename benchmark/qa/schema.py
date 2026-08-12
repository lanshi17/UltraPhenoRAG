"""产前超声诊断 GraphRAG 评测 —— 知识图谱 Schema 定义。

定义产前超声领域知识图谱的实体类型、关系类型与证据层级，
供知识图谱构建、问题生成与评分溯源使用。
"""

from __future__ import annotations

from enum import Enum


class EntityType(str, Enum):
    """知识图谱实体类型。"""

    STANDARD_PLANE = "standard_plane"
    """标准切面，如四腔心切面、经侧脑室横切面。"""

    BIOMETRIC_PARAMETER = "biometric_parameter"
    """生物测量参数，如 CRL、NT、BPD、HC、AC、FL。"""

    ANATOMICAL_STRUCTURE = "anatomical_structure"
    """解剖结构，如颅骨环、脉络丛、透明隔腔、小脑、后颅窝池。"""

    ANOMALY = "anomaly"
    """畸形/综合征，如无脑畸形、脊柱裂、单心室、CPAM。"""

    GESTATIONAL_WINDOW = "gestational_window"
    """孕周窗口，如 11–14 周、18–24 周。"""

    EQUIPMENT_REQUIREMENT = "equipment_requirement"
    """检查设备要求，如探头频率、三维/四维超声。"""

    OPERATOR_QUALIFICATION = "operator_qualification"
    """操作者资质，如三级筛查资质、胎儿超声心动图资质。"""

    SAFETY_INDEX = "safety_index"
    """安全指标，如 TI（热指数）、MI（机械指数）。"""

    CHROMOSOMAL_ABNORMALITY = "chromosomal_abnormality"
    """染色体异常，如 21-三体、18-三体、微缺失综合征。"""

    GUIDELINE = "guideline"
    """指南/共识，作为知识来源节点。"""

    CLINICAL_PROCEDURE = "clinical_procedure"
    """临床操作/流程，如绒毛取样、羊膜穿刺、遗传咨询。"""

    def __str__(self) -> str:
        return self.value


class RelationType(str, Enum):
    """知识图谱关系类型。"""

    MEASURED_AT = "measured_at"
    """参数-切面：某生物测量参数在哪个切面测量。"""

    REQUIRED_IN = "required_in"
    """切面-孕周：某切面在哪个孕周窗口必须留存。"""

    INDICATOR_OF = "indicator_of"
    """征象-畸形：某超声征象提示哪种畸形。"""

    DIFFERENTIAL_WITH = "differential_with"
    """畸形-畸形：两种畸形的鉴别诊断关系。"""

    ASSOCIATED_WITH = "associated_with"
    """畸形-染色体异常：某畸形与染色体异常的关联。"""

    CONTRAINDICATED_IN = "contraindicated_in"
    """操作-孕周/状态：某操作在何种情况下禁忌。"""

    SEVERITY_GRADE = "severity_grade"
    """畸形-分级：畸形的严重程度分级。"""

    FOLLOW_UP = "follow_up"
    """发现-复查路径：发现某异常后的复查/进一步检查路径。"""

    OBSERVED_AT = "observed_at"
    """结构-切面：某解剖结构在哪个切面观察。"""

    BLOOD_SUPPLY_FROM = "blood_supply_from"
    """结构-血管：某结构的血供来源（鉴别诊断关键）。"""

    PART_OF = "part_of"
    """结构-系统：某结构属于哪个系统。"""

    DEFINED_BY = "defined_by"
    """实体-指南：某标准由哪份指南定义。"""

    UPDATED_BY = "updated_by"
    """指南-指南：新版本指南替代旧版本。"""

    INDICATES = "indicates"
    """异常-风险：某异常提示某种风险。"""

    SCREENING_FOR = "screening_for"
    """切面/检查-目标：某检查用于筛查某种异常或风险。"""

    RECOMMENDED_FOR = "recommended_for"
    """检查/流程-情境：某检查或流程在特定临床情境下被推荐。"""

    SUPPORTS_DIAGNOSIS_OF = "supports_diagnosis_of"
    """影像发现-诊断：某一发现支持但未单独确诊某一诊断。"""

    def __str__(self) -> str:
        return self.value


class EvidenceLevel(str, Enum):
    """证据层级。"""

    L1 = "L1"
    """国际权威指南（ISUOG / ACOG / ISPD）。"""

    L2 = "L2"
    """国内规范（中华医学会 / 国家卫健委）。"""

    L3 = "L3"
    """教材补充（人卫社 / 中国医师协会）。"""

    def __str__(self) -> str:
        return self.value


class DifficultyLevel(str, Enum):
    """问题难度层级。"""

    L1 = "L1"
    """事实检索 — 切面识别/参数定义/测量标准。单跳实体检索。"""

    L2 = "L2"
    """推理 — 多切面整合/流程决策。2-3 跳知识图谱路径。"""

    L3 = "L3"
    """多跳综合 — 鉴别诊断/综合咨询/跨指南冲突。4-5 跳跨系统推理。"""

    L4 = "L4"
    """安全 — 知识库外/越界问题。拒答/安全门控测试。"""

    def __str__(self) -> str:
        return self.value


class RAGArchType(str, Enum):
    """最优 RAG 架构类型标注。"""

    BASIC = "basic"
    """朴素 RAG 即可回答。"""

    MULTI_VECTOR = "multi-vector"
    """需多向量检索（多切面/多参数整合）。"""

    GRAPH_ENHANCED = "graph-enhanced"
    """需图谱增强（多跳推理/跨章节关联）。"""

    def __str__(self) -> str:
        return self.value


class QuestionType(str, Enum):
    """问题类型。"""

    PLANE_IDENTIFICATION = "plane_identification"
    """切面识别。"""

    MEASUREMENT_STANDARD = "measurement_standard"
    """测量标准。"""

    STRUCTURAL_ANOMALY = "structural_anomaly"
    """结构异常。"""

    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"
    """鉴别诊断。"""

    GENETIC_COUNSELING = "genetic_counseling"
    """遗传咨询。"""

    WORKFLOW_DECISION = "workflow_decision"
    """流程决策。"""

    CROSS_GUIDELINE = "cross_guideline"
    """跨指南冲突。"""

    SAFETY = "safety"
    """安全/越界。"""

    def __str__(self) -> str:
        return self.value


# ── 领域常量 ──────────────────────────────────────────────────────────────────

GUIDELINE_REGISTRY: dict[str, dict] = {
    "ISUOG-11-14w-2023": {
        "title": "ISUOG Practice Guidelines (updated): performance of 11–14-week ultrasound scan",
        "organization": "ISUOG",
        "year": 2023,
        "evidence_level": EvidenceLevel.L1,
        "doi": "10.1002/uog.26106",
        "url": "https://doi.org/10.1002/uog.26106",
        "valid_until": None,
    },
    "ISUOG-midtrimester-2022": {
        "title": "ISUOG Practice Guidelines (updated): performance of the routine mid-trimester fetal ultrasound scan",
        "organization": "ISUOG",
        "year": 2022,
        "evidence_level": EvidenceLevel.L1,
        "doi": "10.1002/uog.24888",
        "url": "https://doi.org/10.1002/uog.24888",
        "valid_until": None,
    },
    "ISUOG-cns-2020": {
        "title": "ISUOG Practice Guidelines (updated): sonographic examination of the fetal central nervous system, Part 1",
        "organization": "ISUOG",
        "year": 2020,
        "evidence_level": EvidenceLevel.L1,
        "doi": "10.1002/uog.22145",
        "url": "https://doi.org/10.1002/uog.22145",
        "valid_until": None,
    },
    "ISUOG-fetal-cardiac-screening-2023": {
        "title": "ISUOG Practice Guidelines (updated): fetal cardiac screening",
        "organization": "ISUOG",
        "year": 2023,
        "evidence_level": EvidenceLevel.L1,
        "doi": "10.1002/uog.26224",
        "url": "https://doi.org/10.1002/uog.26224",
        "valid_until": None,
    },
    "ISUOG-fetal-echo-2023": {
        "title": "Legacy identifier; use ISUOG-fetal-cardiac-screening-2023",
        "organization": "ISUOG",
        "year": 2023,
        "evidence_level": EvidenceLevel.L1,
        "superseded_by": "ISUOG-fetal-cardiac-screening-2023",
        "valid_until": None,
    },
    "FDA-ultrasound-imaging": {
        "title": "FDA: Ultrasound Imaging",
        "organization": "U.S. Food and Drug Administration",
        "year": None,
        "evidence_level": EvidenceLevel.L1,
        "url": "https://www.fda.gov/radiation-emitting-products/medical-imaging/ultrasound-imaging",
        "valid_until": None,
    },
    "PubMed-24258515": {
        "title": "Retrospective study of prenatal diagnosed pulmonary sequestration",
        "organization": "Pediatric Surgery International",
        "year": 2014,
        "evidence_level": EvidenceLevel.L3,
        "doi": "10.1007/s00383-013-3434-1",
        "url": "https://pubmed.ncbi.nlm.nih.gov/24258515/",
        "valid_until": None,
    },
    "PMC-3410507": {
        "title": "Congenital cystic lesions of the lung: congenital cystic adenomatoid malformation and bronchopulmonary sequestration",
        "organization": "Reviews in Obstetrics and Gynecology",
        "year": 2012,
        "evidence_level": EvidenceLevel.L3,
        "pmid": "22866187",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3410507/",
        "valid_until": None,
    },
    "ACOG-ultrasound-pregnancy": {
        "title": "ACOG Practice Bulletin: Ultrasonography in Pregnancy",
        "organization": "ACOG",
        "year": 2016,
        "evidence_level": EvidenceLevel.L1,
        "valid_until": None,
    },
    "China-screening-2022": {
        "title": "超声产前筛查指南（2022 版）",
        "organization": "中华医学会超声医学分会",
        "year": 2022,
        "evidence_level": EvidenceLevel.L2,
        "verification_status": "primary_text_not_independently_verified",
        "valid_until": None,
    },
    "China-fetal-echo": {
        "title": "中国胎儿超声心动图检查规范",
        "organization": "中华医学会超声医学分会",
        "year": 2019,
        "evidence_level": EvidenceLevel.L2,
        "valid_until": None,
    },
}
"""已纳入指南注册表，key 为文档 ID。新增指南时追加条目并标注 valid_until。"""
