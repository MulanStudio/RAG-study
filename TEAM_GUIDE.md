# 🛢️ 油田服务 RAG 比赛 - 团队分工指南

## 团队架构（5人，任务独立并行）

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG 系统架构                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐                                                   │
│   │ 📂 数据层   │ ←── 👤 成员A: 数据工程师                           │
│   └─────────────┘     负责多模态数据加载                             │
│          ↓                                                          │
│   ┌─────────────┐                                                   │
│   │ 🔍 检索层   │ ←── 👤 成员B: 检索工程师                           │
│   └─────────────┘     负责检索策略 + 🆕 LLM Query 改写               │
│          ↓                                                          │
│   ┌─────────────┐                                                   │
│   │ 🤖 生成层   │ ←── 👤 成员C: Prompt 工程师                        │
│   └─────────────┘     负责生成优化 + 🆕 LLM 答案后处理               │
│          ↓                                                          │
│   ┌─────────────┐                                                   │
│   │ 📊 评测层   │ ←── 👤 成员D: 评测工程师                           │
│   └─────────────┘     负责评测 + 🆕 语义相似度评分                   │
│          ↓                                                          │
│   ┌─────────────┐                                                   │
│   │ 🖥️ 系统层   │ ←── 👤 成员E: 系统工程师                           │
│   └─────────────┘     负责集成和配置管理                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 👤 成员A：数据工程师（Data Engineer）

### 负责模块
```
src/member_a_data/
├── loaders/                # 多模态数据加载 (PDF/Excel/Word/MD/PPT/Images)
├── pdf_table_extractor.py  # PDF 表格提取
├── metadata_cleaner.py     # 元数据清洗与标准化
└── chunk_summarizer.py     # 文本块摘要生成
```

### 核心任务
1. **优化数据解析**：确保各类文件格式正确解析
2. **表格提取**：PDF/Excel 表格转结构化文本
3. **图片理解**：调用 VLM 生成图片描述
4. **元数据清洗**：标准化元数据字段，自动提取日期/公司名/季度等
5. **文本块摘要**：为 chunk 生成精炼摘要，增强检索效果

### 关键指标
- 数据加载成功率 > 99%
- 表格结构保留率
- 图片描述质量

### 工作接口
```python
from src.member_a_data.loaders import load_all_documents
docs = load_all_documents("data/")  # 返回 List[Document]
```

---

## 👤 成员B：检索工程师（Retrieval Engineer）

### 负责模块
```
src/member_b_retrieval/
├── retrieval/              # 检索核心 + 🆕 LLM Query 改写
├── rrf_fusion.py
├── hyde_retrieval.py
├── query_decomposition.py
└── text_processing.py
```

### 核心任务
1. **混合检索策略**：向量 + BM25 + HyDE
2. **分组召回**：按文档类型分组检索
3. **RRF 融合**：科学融合多路结果
4. **重排序优化**：调整 Reranker 参数
5. **🆕 LLM Query 改写**：用 LLM 处理中文音译、同义词、寒暄语过滤

### 🆕 LLM Query 改写详解

**问题背景**：
- 用户可能用中文音译（"哈利伯顿" vs "Halliburton"）
- 用户可能带寒暄语（"你好，SLB 的营收是多少？"）
- 用户可能使用非标准表达

**解决方案**：使用 LLM 改写 Query，而非 hardcode 同义词表
```python
# 在 retrieval/__init__.py 中实现
def _normalize_query_with_llm(self, query: str) -> str:
    """用 LLM 改写 Query，处理中文音译、同义词等"""
    prompt = self.config.get_prompt("query_normalization").format(query=query)
    response = self.llm.invoke(prompt)
    return response.content.strip()
```

**优点**：
- 通用性强，无需维护同义词表
- 自动处理中文音译（哈利伯顿 → Halliburton）
- 自动过滤寒暄语（"你好" 等）
- 自动修正拼写错误

### 关键指标
- Recall@10（召回率）
- MRR（平均倒数排名）
- 检索延迟 < 2s

### 工作接口
```python
from src.member_b_retrieval.retrieval import OilfieldRetriever
retriever = OilfieldRetriever(vectorstore, config, llm)
docs, debug_info, avg_score, max_score = retriever.retrieve(query, top_k=5)
```

---

## 👤 成员C：Prompt 工程师（Prompt Engineer）

### 负责模块
```
src/member_c_generation/
└── generation/             # Prompt/CRAG/回答格式 + 🆕 LLM 答案后处理
```

### 核心任务
1. **Prompt 优化**：针对 AI 评分优化 Prompt
2. **CRAG 实现**：检索质量评估 + 查询重写
3. **答案格式化**：确保输出格式符合评分标准
4. **多语言支持**：中英文问答
5. **常识拒答策略**：材料外常识类问题统一拒答
6. **🆕 LLM 答案后处理**：用 LLM 提取核心答案、格式化选择题

### 🆕 LLM 答案后处理详解

**问题背景**：
- LLM 可能在答案中加寒暄语（"很高兴为您解答..."）
- LLM 可能加引用说明（"根据文档..."）
- 选择题格式可能不规范（"选项是 A" vs "A. xxx"）

**解决方案 1：LLM 核心答案提取**
```python
def _extract_core_answer(self, answer: str, query: str) -> str:
    """用 LLM 移除寒暄语、引用、解释，只保留核心答案"""
    prompt = self.prompts.format("answer_extraction", answer=answer, query=query)
    response = self.llm.invoke(prompt)
    return response.content.strip()
```

**解决方案 2：LLM 选择题格式化**
```python
def _llm_format_choice(self, answer: str, options: List[Tuple[str, str]]) -> str:
    """用 LLM 确保选择题输出 "A. xxx" 标准格式"""
    prompt = self.prompts.format("format_correction", answer=answer, options=options_text)
    response = self.llm.invoke(prompt)
    return response.content.strip()
```

### 🆕 简化的置信度检查（大模型策略）

**问题背景**：
- 之前有多层保护机制（关键词覆盖、忠实度验证、对齐检查）
- 这些对小模型有用，但对 GPT-4/5 反而会误拒正确答案

**解决方案**：简化检查，信任大模型输出
```python
# 仅在检索分数极低时拒绝
if retrieval_score < 0.1:
    return self._refusal_message(query)
# 其他情况：信任 LLM 输出
```

### 关键指标
- 答案准确率
- 答案完整度
- 格式规范性

### 工作接口
```python
from src.member_c_generation.generation import AnswerGenerator
generator = AnswerGenerator(llm, prompts)
answer, debug_info = generator.generate(query, documents, retrieval_score=0.5)
```

---

## 👤 成员D：评测工程师（Evaluation Engineer）

### 负责模块
```
src/member_d_evaluation/
└── rag_eval.py             # 问题+标准答案+提交答案评分 + 语义相似度
```

### 核心任务
1. **构建测试集**：收集各类问题的测试用例
2. **评测指标**：实现多维度评分
3. **A/B 测试**：对比不同配置的效果
4. **错误分析**：分析失败 case，反馈给其他成员
5. **🆕 语义相似度评测**：基于 embedding 的相似度评分

### 🆕 语义相似度评测详解

**问题背景**：
- 关键词匹配评测过于严格（"33.1 Billion USD" vs "33.1B" 被判错误）
- LLM-as-Judge 成本高且不稳定

**解决方案**：SemanticEvaluator
```python
class SemanticEvaluator:
    """基于 embedding 的语义相似度评测"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def evaluate(self, answer: str, reference: str) -> float:
        # 1. 先检查数值精确匹配
        if self._numerical_match(answer, reference):
            return 1.0
        # 2. 计算语义相似度
        emb1 = self.model.encode([answer])
        emb2 = self.model.encode([reference])
        return float(cosine_similarity(emb1, emb2)[0][0])
```

### 关键指标
- 测试覆盖率
- 各类问题准确率
- 评测效率

### 工作接口
```bash
python src/member_d_evaluation/rag_eval.py

# 输出报告
{
  "overall_score": 8.5,
  "semantic_similarity": 0.92,
  "by_category": {
    "financial": 9.0,
    "technical": 8.2,
    "choice": 10.0
  }
}
```

---

## 👤 成员E：系统工程师（System Engineer）

### 负责模块
```
src/member_e_system/
├── app.py                   # 主应用入口
├── batch_answer_excel.py    # Excel 批量问答
└── azure_openai_client.py   # Azure OpenAI 适配
config/config.yaml           # 统一配置
README.md                    # 使用文档
scripts/prebuild_index.py    # 预构建索引脚本
```

### 核心任务
1. **系统集成**：整合各模块
2. **配置管理**：统一管理所有参数
3. **环境搭建**：确保各成员环境一致
4. **文档编写**：使用说明、API 文档
5. **性能优化**：向量索引持久化、并行 Embedding

### 关键指标
- 系统稳定性
- 一键部署成功率
- 配置灵活性

### 工作接口
```bash
# 一键运行
python src/member_e_system/app.py --question "水力压裂是什么？"

# 批量问答
python src/member_e_system/batch_answer_excel.py --input data/questions.xlsx
```

---

## 🆕 LLM 驱动优化 vs Hardcode 对比

| 场景 | 旧方案（Hardcode） | 新方案（LLM 驱动） |
|------|-------------------|-------------------|
| 中文音译 | 维护同义词表 `{"哈利伯顿": "Halliburton"}` | LLM 自动改写 |
| 寒暄语过滤 | 正则匹配 `r"^(你好|您好).*"` | LLM 提取核心问题 |
| 答案格式化 | 正则替换 `r"根据.*?[,，]"` | LLM 提取核心答案 |
| 选择题格式 | 正则匹配选项字母 | LLM 智能格式化 |
| 置信度检查 | 多层规则判断 | 简化为检索分数阈值 |

**优点**：
- ✅ 无需维护规则和同义词表
- ✅ 泛化能力强，覆盖更多边缘情况
- ✅ 随 LLM 能力提升自动优化

**适用条件**：
- 使用大模型（GPT-4/5）时推荐
- 小模型（qwen2.5:3b）可能不稳定，需权衡

---

## 📅 协作流程

### 第一阶段：独立开发（并行）
```
Day 1-3:
  成员A: 完成数据加载模块
  成员B: 完成检索模块 + LLM Query 改写
  成员C: 完成生成模块 + LLM 答案后处理
  成员D: 准备测试用例 + 语义相似度评测
  成员E: 搭建项目框架
```

### 第二阶段：集成测试
```
Day 4-5:
  成员E: 集成各模块
  成员D: 运行评测
  全员: 根据评测结果优化各自模块
```

### 第三阶段：调优冲刺
```
Day 6-7:
  成员D: 分析错题，分配给相关成员
  全员: 针对性优化
  成员E: 最终集成，准备提交
```

---

## 🔧 Git 协作规范

```bash
# 分支策略
main          # 稳定版本
├── dev       # 开发分支
├── feat/loader    # 成员A
├── feat/retrieval # 成员B
├── feat/generation # 成员C
├── feat/evaluation # 成员D
└── feat/system     # 成员E

# 合并流程
1. 各成员在自己的 feat/ 分支开发
2. 完成后 PR 到 dev 分支
3. 成员E 负责 code review 和合并
4. 测试通过后合并到 main
```

---

## 📞 沟通机制

1. **每日站会**（15分钟）
   - 昨天完成了什么
   - 今天计划做什么
   - 有什么阻塞

2. **接口约定**
   - 各模块的输入输出格式提前约定
   - 变更接口需通知相关成员

3. **共享文档**
   - 错题本：记录失败 case 和原因
   - 优化日志：记录每次优化的效果

---

## 📋 核心 Prompt 模板

### Query Normalization Prompt（成员B）
```
将以下用户问题改写为标准化的检索查询：
1. 将中文音译转为英文（如：哈利伯顿 → Halliburton）
2. 移除寒暄语（如：你好、请问）
3. 保持核心问题不变

原始问题：{query}
改写后：
```

### Answer Extraction Prompt（成员C）
```
从以下回答中提取核心答案，移除所有：
- 寒暄语（如：很高兴为您解答）
- 引用说明（如：根据文档）
- 不确定性表达（如：可能、大概）

原始回答：{answer}
问题：{query}
核心答案：
```

### Format Correction Prompt（成员C）
```
将以下答案格式化为标准选择题格式 "X. 选项内容"：
答案：{answer}
选项：{options}
标准格式答案：
```
