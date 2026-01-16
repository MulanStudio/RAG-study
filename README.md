# 🛢️ 油田服务 RAG 系统

一个专为油田服务领域设计的多模态 RAG（检索增强生成）问答系统。

## 🚀 快速开始

### 1. 准备数据

把组委会提供的数据放到 `data/` 文件夹：

```
data/
├── *.pdf          # PDF 文档（支持表格提取）
├── *.xlsx         # Excel 数据
├── *.csv          # CSV 数据
├── *.docx         # Word 文档
├── *.md           # Markdown 文档
├── *.pptx         # PowerPoint
├── *.png / *.jpg  # 图片（VLM 自动生成描述）
└── subfolders/    # 支持子文件夹
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动 Ollama

```bash
# 终端 1: 启动 Ollama 服务
ollama serve

# 终端 2: 下载模型
ollama pull qwen2.5:3b
ollama pull llava  # 可选，用于图片理解
```

### 4. 运行系统

```bash
# 方式 1: 命令行交互
python app.py

# 方式 2: Web UI
python app.py --mode web
# 或
streamlit run streamlit_app.py

# 方式 3: 直接提问
python app.py --question "SLB 的 2023 年营收是多少？"
```

---

## 📁 项目结构

```
RAG-study/
├── data/                    # 👈 把数据放这里！
│   ├── documents/
│   └── images/
├── config/
│   └── config.yaml          # 统一配置文件
├── src/
│   ├── loaders/             # 数据加载模块
│   ├── retrieval/           # 检索模块
│   ├── generation/          # 生成模块
│   └── evaluation/          # 评测模块
├── app.py                   # 主应用入口
├── streamlit_app.py         # Web UI
├── benchmark_challenge.py   # 评测脚本
├── requirements.txt         # 依赖清单
├── TEAM_GUIDE.md           # 团队分工指南
└── README.md
```

---

## ⚙️ 配置说明

编辑 `config/config.yaml` 来调整系统行为：

```yaml
# 模型配置
models:
  llm:
    model_name: "qwen2.5:3b"  # 可以换成其他模型
  embedding:
    model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 检索配置
retrieval:
  k_excel: 10      # Excel 数据召回数量
  k_word: 5        # Word 文档召回数量

# Prompt 配置（可自定义）
prompts:
  generation: |
    You are a Senior Technical Expert...
```

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户问题                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Query Decomposition (问题分解)                              │
│  复杂问题 → 多个子问题                                        │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  向量检索    │      │   BM25      │      │   HyDE      │
│  (分组召回)  │      │  关键词     │      │  假设文档    │
└─────────────┘      └─────────────┘      └─────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   RRF 融合      │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Reranker 精排  │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  CRAG 自我修正  │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  LLM 生成回答   │
                    └─────────────────┘
```

---

## 🔧 核心优化

| 优化项 | 说明 | 效果 |
|--------|------|------|
| **Query Decomposition** | 复杂问题拆分为子问题 | 提高复杂问题覆盖率 |
| **HyDE** | 生成假设文档后检索 | 提高语义匹配准确度 |
| **RRF 融合** | 多路召回按排名融合 | 综合多种检索优势 |
| **PDF 表格提取** | 表格转结构化文本 | 精准检索表格数据 |
| **Parent-Child 分块** | 小块检索，大块生成 | 检索精准 + 上下文完整 |

---

## 👥 团队分工

详见 [TEAM_GUIDE.md](TEAM_GUIDE.md)

| 角色 | 负责模块 |
|------|---------|
| 成员A | 数据加载 (`src/loaders/`) |
| 成员B | 检索优化 (`src/retrieval/`) |
| 成员C | Prompt 工程 (`src/generation/`) |
| 成员D | 评测调优 (`src/evaluation/`) |
| 成员E | 系统集成 (`app.py`, `config/`) |

---

## 📊 评测

```bash
# 运行评测
python benchmark_challenge.py

# 输出示例
🧪 Testing: 比较类问题
❓ Question: Compare the revenue of Latin America vs Middle East...
🤖 Answer: Based on the Q3 2024 data, Latin America showed 12% growth...
📝 Score: 9/10
```

---

## 🐛 常见问题

**Q: Ollama 连接失败**
```bash
# 确保 Ollama 正在运行
ollama serve
```

**Q: 图片没有描述**
```bash
# 下载 VLM 模型
ollama pull llava
```

**Q: 检索效果不好**
- 检查 `config/config.yaml` 中的召回参数
- 尝试调整 `rrf.weights` 中的权重

---

## 📝 License

MIT
