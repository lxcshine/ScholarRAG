# ScholarRAG

🔬 **面向顶会论文的模块化检索增强生成系统**  
A Production-Grade Academic RAG Framework for CVPR/IEEE/NeurIPS Literature

---

## 📖 项目简介

**ScholarRAG** 是一款专为科研人员与 AI 开发者打造的高性能学术文献检索增强生成（RAG）系统。项目深度参考顶级综述《Retrieval-Augmented Generation for Large Language Models: A Survey》，针对顶会论文排版复杂、表格公式密集、知识更新快等核心痛点，构建了一套从“结构化解析 → 混合检索 → 迭代生成 → 引用溯源”的完整工程闭环。

系统支持本地大模型与在线 API 无缝路由，旨在让大模型真正“读懂”论文、精准提取实验指标、杜绝幻觉输出，助力科研写作、文献调研与指标对比自动化。

---

## 🌟 核心亮点

### 1. Advanced + Modular 混合检索架构
摒弃传统 Naive RAG 的单一检索链路，采用 `BM25 关键词 + Dense Vector 语义 + CrossEncoder 重排序` 三路混合检索管线。结合 **HyDE 假设文档生成** 与 **多视角查询重写**，显著提升长尾学术概念与专业术语的召回率。内置多轮迭代检索与 LLM 驱动的**“信息充分性自检”机制**，系统可自主判断上下文是否完备并动态补全缺失指标，复杂汇总类查询命中率提升 30%+。

### 2. 面向顶会论文的原子化分块与强溯源防幻觉
针对双栏排版、三线表与 LaTeX 公式易断裂的问题，提出 `Small2Big + Metadata Attachments + Structural Index` 分块策略，实现**表格原子化保留**、章节上下文绑定与元数据路由。生成阶段采用 Token 级安全压缩与 `[DocID]` 强制引用校验模块，无效或越界引用自动拦截替换。从源头切断大模型“胡编乱造”，输出结果结构清晰、来源可溯，可直接粘贴至论文实验对比或 Related Work 章节。

### 3. 本地/在线双模路由与消费级显卡友好设计
提供 `fast / balanced / accurate` 三档检索策略一键切换，兼顾响应速度与推理精度。创新支持**生成模型热切换**（本地 Ollama `gemma4` / 在线 `GLM-4`），嵌入模型固定使用高性能本地 `bge-m3`，彻底解决大模型显存瓶颈。内置 LaTeX/控制符深度清洗、阶梯重试降级与向量缓存机制，**8GB 显存设备即可流畅运行**，单次查询平均响应 <25 秒，满足日常科研流水线的高频、稳定调用需求。

---

## 🛠️ 技术栈

| 模块 | 技术选型 |
|:---|:---|
| **核心框架** | `LangChain`, `ChromaDB`, `PyMuPDF` |
| **模型路由** | `Ollama` (bge-m3 / gemma4), `智谱 GLM-4` / OpenAI 兼容接口 |
| **检索增强** | `BM25`, `CrossEncoder` (ms-marco), `HyDE`, `Iterative Retrieval` |
| **工程优化** | `Small2Big Chunking`, `Token Compression`, `Citation Validation` |

---

##  快速开始

### 1. 环境准备
```bash
# 克隆项目并安装依赖
git clone https://github.com/your-username/scholar-rag.git
cd scholar-rag
pip install -r requirements.txt
