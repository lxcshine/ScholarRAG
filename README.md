# ScholarRAG

🔬 **面向顶会论文的模块化检索增强生成系统**  
A Production-Grade Academic RAG Framework for CVPR/IEEE/NeurIPS Literature

---

## 📖 项目简介

**ScholarRAG** 是一款专为科研人员与 AI 开发者打造的高性能学术文献检索增强生成系统。项目深度参考顶级综述《Retrieval-Augmented Generation for Large Language Models: A Survey》，针对顶会论文排版复杂、表格公式密集、知识更新快等核心痛点，构建了一套从“结构化解析 → 混合检索 → 迭代生成 → 引用溯源”的完整工程闭环。

系统支持本地大模型与在线 API 无缝路由，旨在让大模型真正读懂论文、精准提取实验指标、杜绝幻觉输出，助力科研写作、文献调研与指标对比自动化。

---

## 🌟 核心亮点

### 1. Advanced + Modular 混合检索架构
摒弃传统Naive RAG的单一检索链路，采用`BM25 关键词 + Dense Vector语义 + CrossEncoder重排序`三路混合检索管线。结合HyDE假设文档生成与多视角查询重写，显著提升长尾学术概念与专业术语的召回率。内置多轮迭代检索与LLM驱动的信息充分性自检机制，系统可自主判断上下文是否完备并动态补全缺失指标，复杂汇总类查询命中率提升30%+。

### 2. 面向顶会论文的原子化分块与强溯源防幻觉
针对双栏排版、三线表与LaTeX公式易断裂的问题，提出`Small2Big + Metadata Attachments + Structural Index`分块策略，实现表格原子化保留、章节上下文绑定与元数据路由。生成阶段采用Token级安全压缩与`[DocID]` 强制引用校验模块，无效或越界引用自动拦截替换。输出结果结构清晰、来源可溯。

### 3. 本地/在线双模路由与消费级显卡友好设计
提供`fast / balanced / accurate`三档检索策略一键切换，兼顾响应速度与推理精度。支持生成模型热切换（本地 Ollama `gemma4` / 在线 `GLM-4`），嵌入模型固定使用高性能本地`bge-m3`，彻底解决大模型显存瓶颈。内置LaTeX/控制符深度清洗、阶梯重试降级与向量缓存机制，8GB显存设备即可流畅运行，单次查询平均响应<25 秒，满足日常科研流水线的高频、稳定调用需求。

---

## 🛠️ 技术栈

| 模块 | 技术选型 |
|:---|:---|
| **核心框架** | `LangChain`, `ChromaDB`, `PyMuPDF` |
| **模型路由** | `Ollama` (bge-m3 / gemma4), `智谱 GLM-4` / OpenAI 兼容接口 |
| **检索增强** | `BM25`, `CrossEncoder` (ms-marco), `HyDE`, `Iterative Retrieval` |
| **工程优化** | `Small2Big Chunking`, `Token Compression`, `Citation Validation` |

---

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
