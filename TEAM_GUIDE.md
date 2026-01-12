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
│   └─────────────┘     负责检索策略优化                               │
│          ↓                                                          │
│   ┌─────────────┐                                                   │
│   │ 🤖 生成层   │ ←── 👤 成员C: Prompt 工程师                        │
│   └─────────────┘     负责 Prompt 和生成优化                         │
│          ↓                                                          │
│   ┌─────────────┐                                                   │
│   │ 📊 评测层   │ ←── 👤 成员D: 评测工程师                           │
│   └─────────────┘     负责评测和参数调优                             │
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
src/loaders/
├── pdf_loader.py      # PDF 表格提取
├── excel_loader.py    # Excel/CSV 加载
├── word_loader.py     # Word 结构化加载
├── image_loader.py    # 图片 VLM 处理
├── markdown_loader.py # Markdown 加载
└── video_loader.py    # 视频处理（如果需要）
```

### 核心任务
1. **优化数据解析**：确保各类文件格式正确解析
2. **表格提取**：PDF/Excel 表格转结构化文本
3. **图片理解**：调用 VLM 生成图片描述
4. **语义增强**：给文档添加元数据标签

### 关键指标
- 数据加载成功率 > 99%
- 表格结构保留率
- 图片描述质量

### 工作接口
```python
# 其他成员调用你的代码
from src.loaders import load_all_documents
docs = load_all_documents("data/")  # 返回 List[Document]
```

---

## 👤 成员B：检索工程师（Retrieval Engineer）

### 负责模块
```
src/retrieval/
├── vector_search.py       # 向量检索
├── bm25_search.py         # BM25 关键词检索
├── hyde.py                # HyDE 假设文档检索
├── query_decomposition.py # 问题分解
├── rrf_fusion.py          # RRF 多路融合
└── reranker.py            # 重排序
```

### 核心任务
1. **混合检索策略**：向量 + BM25 + HyDE
2. **分组召回**：按文档类型分组检索
3. **RRF 融合**：科学融合多路结果
4. **重排序优化**：调整 Reranker 参数

### 关键指标
- Recall@10（召回率）
- MRR（平均倒数排名）
- 检索延迟 < 2s

### 工作接口
```python
# 其他成员调用你的代码
from src.retrieval import retrieve_documents
docs = retrieve_documents(query, top_k=5)  # 返回最相关的文档
```

---

## 👤 成员C：Prompt 工程师（Prompt Engineer）

### 负责模块
```
src/generation/
├── prompts.py         # Prompt 模板管理
├── crag.py            # 自我修正 RAG
├── answer_verify.py   # 答案验证
└── citation.py        # 来源引用
```

### 核心任务
1. **Prompt 优化**：针对 AI 评分优化 Prompt
2. **CRAG 实现**：检索质量评估 + 查询重写
3. **答案格式化**：确保输出格式符合评分标准
4. **多语言支持**：中英文问答

### 关键指标
- 答案准确率
- 答案完整度
- 格式规范性

### 工作接口
```python
# 其他成员调用你的代码
from src.generation import generate_answer
answer = generate_answer(query, context_docs, llm)
```

### Prompt 模板示例
```yaml
# config/prompts.yaml
generation:
  system: |
    You are a Senior Oil & Gas Technical Expert.
    Answer based ONLY on the provided context.
    If unsure, say "I don't have enough information."
    
  format: |
    Structure your answer:
    1. Direct answer to the question
    2. Supporting evidence from context
    3. Key data points (if applicable)
```

---

## 👤 成员D：评测工程师（Evaluation Engineer）

### 负责模块
```
src/evaluation/
├── benchmark.py       # 评测框架
├── metrics.py         # 评测指标
├── test_cases.py      # 测试用例管理
└── analysis.py        # 结果分析
```

### 核心任务
1. **构建测试集**：收集各类问题的测试用例
2. **评测指标**：实现多维度评分
3. **A/B 测试**：对比不同配置的效果
4. **错误分析**：分析失败 case，反馈给其他成员

### 关键指标
- 测试覆盖率
- 各类问题准确率
- 评测效率

### 工作接口
```python
# 运行评测
python benchmark.py --config config/config.yaml

# 输出报告
{
  "overall_score": 8.5,
  "by_category": {
    "financial": 9.0,
    "technical": 8.2,
    "image": 7.8
  },
  "failed_cases": [...]
}
```

---

## 👤 成员E：系统工程师（System Engineer）

### 负责模块
```
├── app.py             # 主应用入口
├── config/
│   └── config.yaml    # 统一配置
├── scripts/
│   ├── setup.sh       # 环境安装
│   └── run.sh         # 一键运行
└── README.md          # 使用文档
```

### 核心任务
1. **系统集成**：整合各模块
2. **配置管理**：统一管理所有参数
3. **环境搭建**：确保各成员环境一致
4. **文档编写**：使用说明、API 文档

### 关键指标
- 系统稳定性
- 一键部署成功率
- 配置灵活性

### 工作接口
```bash
# 一键运行
./scripts/run.sh

# 或
python app.py --data_dir data/ --config config/config.yaml
```

---

## 📅 协作流程

### 第一阶段：独立开发（并行）
```
Day 1-3:
  成员A: 完成数据加载模块
  成员B: 完成检索模块
  成员C: 完成生成模块
  成员D: 准备测试用例
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
