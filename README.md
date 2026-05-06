# ScholarRAG

🔬 **面向顶会论文的模块化检索增强生成系统**  
A Production-Grade Academic RAG Framework for CVPR/IEEE/NeurIPS Literature

---

## 📖 项目简介

**ScholarRAG** 是一款专为科研人员与 AI 开发者打造的高性能学术文献检索增强生成系统。项目深度参考顶级综述《Retrieval-Augmented Generation for Large Language Models: A Survey》，针对顶会论文排版复杂、表格公式密集、知识更新快等核心痛点，构建了一套从“结构化解析 → 混合检索 → 迭代生成 → 引用溯源”的完整工程闭环。

系统支持本地大模型与在线 API 无缝路由，通过引入**RF-Mem（Recollection-Familiarity Memory Retrieval）双通路记忆检索机制**，灵感来源于人类认知科学中的双重记忆理论，结合迭代式 RAG Agent 与多轮对话记忆压缩，让大模型真正读懂论文、精准提取实验指标、降低幻觉输出。

---

## 🌟 核心亮点

### 1. 面向顶会论文的原子化分块策略
针对学术论文双栏排版、三线表密集、LaTeX 公式易断裂等问题，提出**Small2Big+表格原子化**的分块架构。PDF 解析阶段自动检测双栏布局、提取表格为 Markdown、保护公式结构；分块阶段采用父子块策略——子块（800 tokens）用于精准向量检索，父块（1500 tokens）保留完整章节上下文用于生成；表格无论长短均保持结构完整，超长表格按行切分时强制保留表头于每个子块。每个 Chunk 绑定丰富的元数据（doc_id/year/venue/section/chunk_type），支持按会议/年份/章节的精准过滤检索。

### 2. RF-Mem双通路记忆检索 + 迭代式RAG Agent
系统通过计算查询与记忆库的相似度分布熵和峰值信号，动态门控选择检索路径：高熟悉度场景直接Top-K返回，低熟悉度场景启动Beam Search + KMeans聚簇迭代扩展检索。结合**LLM 驱动的信息充分性自检**，每轮检索后自动判断上下文是否完备，不足时指出缺失信息并触发下一轮补充检索。

### 3. 表格查询专用管线 + 上下文记忆压缩
当检测到实验指标对比类查询时，系统对结构化数据执行精确计算，避免 LLM 数字幻觉。对话记忆方面，维护多轮对话历史并支持自动摘要压缩，超阈值时由 LLM 生成 Session Summary 替代冗长历史；上下文注入时采用 **智能优先级排序**（摘要+100、表格+50、指标+30），摘要和表格永不截断，普通文本安全截断，确保关键信息完整性的同时防止上下文窗口溢出。

### 4. 混合检索管线 + 多会话持久化
检索层融合**BM25 关键词 + Dense Vector 语义 + CrossEncoder 重排序** 三路混合策略，可选 HyDE 假设文档生成与查询重写扩展。系统支持 `fast/balanced/accurate` 三档检索策略和 `local/online` 双模型路由，8GB 显存即可流畅运行。对话历史采用 **MySQL + JSON 双存储**，支持多会话新建/切换/重命名/删除，Web UI 提供 SSE 流式输出、引用溯源、主题切换等完整交互功能。

---

## 🛠️ 技术栈

| 模块 | 技术选型 |
|:---|:---|
| **核心框架** | `LangChain`, `ChromaDB`, `PyMuPDF`, `FastAPI` |
| **模型路由** | `Ollama` (qwen3-embedding / gemma4), `智谱 GLM-4` |
| **检索增强** | `BM25`, `CrossEncoder` (ms-marco), `HyDE`, `RF-Mem` |
| **记忆机制** | `Conversation Memory`, `Auto Summarization`, `Context Compression` |
| **表格处理** | `pandas`, `LLM Code Generation`, `Sandbox Execution`, `Self-Correction` |
| **工程优化** | `Small2Big Chunking`, `Table Atomicity`, `Citation Validation`, `Vector Cache` |
| **持久化** | `MySQL` (pymysql), `JSON` (file-based backup) |

---

### Web 界面

- **多会话管理**：新建、切换、重命名、删除历史对话
- **快捷操作**：一键生成论文摘要、方法对比、局限性分析
- **实时流式输出**：SSE 协议支持，逐 Token 显示生成过程
- **引用溯源**：每个回答附带引用文献列表，点击查看详情
- **主题切换**：Dark / Light 双主题
- **检索模式切换**：Fast / Balanced / Accurate 三档
- **模型切换**：Online (GLM-4) / Local (Ollama)

## 快速开始

### 环境准备
```bash
# 克隆项目并安装依赖
git clone https://github.com/lxcshine/ScholarRAG.git
cd ScholarRAG
pip install -r requirements.txt

# 启动Ollama并拉取学术嵌入模型
ollama pull bge-m3
ollama serve

# 将 CVPR/IEEE 等 PDF 放入 ./papers 目录
mkdir papers
# 重建向量索引（自动执行版面解析与分块）
python main.py --rebuild --mode fast
# 启动交互式问答系统
python main.py --mode fast

