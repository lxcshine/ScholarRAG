#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科研RAG系统主入口
支持本地Ollama / 在线GLM-4.5 一键切换
嵌入模型固定使用本地 bge-m3
"""

import sys
import os
import argparse
import logging
import time
import requests
from pathlib import Path

# 确保项目根目录在 sys.path 中
from langchain_ollama import ChatOllama

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from utils.pdf_extractor import extract_pdf
from utils.chunker import chunk_documents
from core.vector_store import init_vector_store, load_vector_store
from core.reranker import Reranker
from core.llm_router import LLMRouter
from pipeline.context_manager import ContextManager
from pipeline.iterative_agent import IterativeRAGAgent
from generator.citation_validator import validate_citations

logger = logging.getLogger(__name__)


def apply_mode_preset(mode: str):
    """根据运行模式动态调整检索策略参数"""
    mode_presets = {
        "fast": {
            "MAX_ITERATIONS": 1,
            "TOP_K_RETRIEVAL": 10,
            "TOP_N_RERANK": 3,
            "ENABLE_HYDE": False,
            "ENABLE_QUERY_REWRITE": False,
            "ENABLE_HYBRID_SEARCH": True,
            "ENABLE_METADATA_FILTER": True
        },
        "balanced": {
            "MAX_ITERATIONS": 2,
            "TOP_K_RETRIEVAL": 15,
            "TOP_N_RERANK": 5,
            "ENABLE_HYDE": True,
            "ENABLE_QUERY_REWRITE": True,
            "ENABLE_HYBRID_SEARCH": True,
            "ENABLE_METADATA_FILTER": True
        },
        "accurate": {
            "MAX_ITERATIONS": 3,
            "TOP_K_RETRIEVAL": 25,
            "TOP_N_RERANK": 8,
            "ENABLE_HYDE": True,
            "ENABLE_QUERY_REWRITE": True,
            "ENABLE_HYBRID_SEARCH": True,
            "ENABLE_METADATA_FILTER": True
        }
    }
    preset = mode_presets.get(mode, mode_presets["balanced"])
    for key, value in preset.items():
        setattr(settings, key, value)
    logger.info(f"🎛️ 检索模式已切换: {mode.upper()} | 迭代: {settings.MAX_ITERATIONS} | Top-K: {settings.TOP_K_RETRIEVAL}")


def check_llm_health() -> bool:
    """检测当前配置的LLM服务是否可用"""
    mode = settings.LLM_MODE
    if mode == "local":
        try:
            resp = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if settings.LOCAL_LLM_MODEL in models:
                    logger.info(f"✅ 本地模型就绪: {settings.LOCAL_LLM_MODEL}")
                    return True
                else:
                    logger.warning(f"⚠️ 模型 {settings.LOCAL_LLM_MODEL} 未找到，可用: {models}")
                    return False
        except Exception as e:
            logger.error(f"❌ 无法连接 Ollama: {e}")
            return False
    else:  # online
        try:
            headers = {"Authorization": f"Bearer {settings.GLM_API_KEY}"}
            resp = requests.get(
                f"{settings.GLM_BASE_URL}/models",
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                logger.info(f"✅ 在线模型就绪: {settings.ONLINE_LLM_MODEL}")
                return True
            else:
                logger.error(f"❌ GLM API 返回异常: {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            logger.error(f"❌ 无法连接 GLM 服务: {e}")
            return False
    return False


def rebuild_index():
    """重建向量数据库与混合检索索引"""
    logger.info("🔄 开始重建向量索引...")
    pdf_dir = Path(settings.PDF_DIR)

    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 已创建目录: {pdf_dir}")
        logger.info("💡 请将CVPR/IEEE/NeurIPS等PDF论文放入该目录后重试")
        return

    # 解析PDF
    docs = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"⚠️ {pdf_dir} 目录下未找到PDF文件")
        return

    for p in pdf_files:
        logger.info(f"📖 解析: {p.name}")
        try:
            docs.append(extract_pdf(str(p)))
        except Exception as e:
            logger.error(f"❌ 解析失败 {p.name}: {e}")

    valid_docs = [d for d in docs if d.page_content.strip()]
    if not valid_docs:
        logger.error("❌ 未找到有效内容，索引终止")
        return

    # 高级分块
    logger.info(f"🔪 开始分块: {len(valid_docs)} 篇文献")
    chunks = chunk_documents(valid_docs)
    logger.info(f"✅ 分块完成: {len(chunks)} 个索引块")

    # 初始化向量库
    logger.info("💾 正在构建向量索引...")
    init_vector_store(chunks)
    logger.info(f"🎉 索引重建成功! 耗时: {time.time():.1f}s")


def run_interactive():
    """启动交互式问答循环"""
    # 健康检查
    if not check_llm_health():
        logger.error("❌ LLM服务未就绪，请检查配置后重试")
        return

    # 加载向量库
    vstore = load_vector_store()
    if not vstore:
        logger.info("📦 未检测到向量库，自动触发重建...")
        rebuild_index()
        vstore = load_vector_store()
        if not vstore:
            logger.error("❌ 向量库加载失败，退出")
            return

    # 初始化核心组件（使用路由获取LLM）
    if settings.LLM_MODE == "online":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=settings.ONLINE_LLM_MODEL,
            openai_api_key=settings.GLM_API_KEY,
            openai_api_base=settings.GLM_BASE_URL,
            temperature=0.1,
            max_tokens=2048,
            verbose=False
        )
    else:
        llm = ChatOllama(
            model=settings.LOCAL_LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1,
            num_ctx=16384,
            verbose=False
        )

    reranker = Reranker()
    ctx_mgr = ContextManager(vstore, reranker, llm)
    agent = IterativeRAGAgent(ctx_mgr, llm)

    # 启动信息
    print("\n" + "=" * 70)
    print(f"🚀 RAG科研系统已启动")
    print(f"   📊 模式: {settings.RETRIEVAL_MODE.upper()}")
    print(
        f"   🤖 生成模型: {settings.LLM_MODE.upper()} ({settings.ONLINE_LLM_MODEL if settings.LLM_MODE == 'online' else settings.LOCAL_LLM_MODEL})")
    print(f"   🔍 嵌入模型: {settings.EMBEDDING_MODEL} (本地)")
    print("=" * 70)
    print("💡 快捷命令:")
    print("   • mode fast|balanced|accurate  : 切换检索策略")
    print("   • model local|online           : 切换生成模型")
    print("   • exit / quit / q              : 退出系统")
    print("=" * 70)

    while True:
        try:
            user_input = input("\n🔍 你的查询: ").strip()
            if not user_input:
                continue

            # 命令: 切换检索模式
            if user_input.lower().startswith("mode "):
                parts = user_input.split()
                if len(parts) == 2 and parts[1] in ["fast", "balanced", "accurate"]:
                    apply_mode_preset(parts[1])
                    settings.RETRIEVAL_MODE = parts[1]
                    # 重新初始化agent以应用新参数
                    agent = IterativeRAGAgent(ctx_mgr, llm)
                    print(f"✅ 检索策略已切换: {parts[1].upper()}")
                else:
                    print("❌ 用法: mode fast | balanced | accurate")
                continue

            # 命令: 切换生成模型
            if user_input.lower().startswith("model "):
                parts = user_input.split()
                if len(parts) == 2 and parts[1] in ["local", "online"]:
                    if LLMRouter.switch_mode(parts[1]):
                        # 重新获取新模型的LLM实例
                        llm = LLMRouter.get_llm()
                        ctx_mgr.llm = llm
                        agent = IterativeRAGAgent(ctx_mgr, llm)
                        model_name = settings.ONLINE_LLM_MODEL if parts[1] == "online" else settings.LOCAL_LLM_MODEL
                        print(f"✅ 生成模型已切换: {parts[1].upper()} ({model_name})")
                    else:
                        print("❌ 切换失败，请检查配置")
                else:
                    print("❌ 用法: model local | online")
                continue

            # 退出命令
            if user_input.lower() in ["exit", "quit", "q", "退出"]:
                print("\n👋 感谢使用，已安全退出。")
                break

            # 执行检索与生成
            start_time = time.time()
            print(f"⏳ 正在执行智能检索 (模式: {settings.RETRIEVAL_MODE})...")

            raw_response = agent.run(user_input)
            elapsed = time.time() - start_time

            # 后处理: 引用校验
            validated_response = validate_citations(raw_response, getattr(ctx_mgr, 'last_docs', []))

            # 输出结果
            print(f"\n⏱️ 耗时: {elapsed:.2f}s")
            print("=" * 70)
            print("📝 生成结果:")
            print("=" * 70)
            print(validated_response)

            # 持久化保存
            timestamp = int(time.time())
            output_file = f"answer_{timestamp}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Query: {user_input}\n")
                f.write(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Mode: {settings.RETRIEVAL_MODE} | Model: {settings.LLM_MODE}\n\n")
                f.write(validated_response)
            print(f"\n✅ 已自动保存至: {output_file}")

        except KeyboardInterrupt:
            print("\n\n⚠️ 检测到中断信号，正在安全退出...")
            break
        except Exception as e:
            logger.error(f"❌ 运行时异常: {e}", exc_info=True)
            print(f"⚠️ 发生错误: {type(e).__name__}")
            print("💡 建议: 检查日志或重试，复杂查询可尝试 mode accurate")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="🔬 科研RAG系统 | 支持本地/在线模型混合部署",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速模式 + 在线模型 (推荐8GB显存用户)
  python main.py --mode fast --model online

  # 高精度模式 + 本地模型 (需大显存)
  python main.py --mode accurate --model local

  # 重建索引
  python main.py --rebuild
        """
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重建向量数据库与BM25索引"
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "accurate"],
        default="fast",
        help="检索策略模式 (默认: fast)"
    )
    parser.add_argument(
        "--model",
        choices=["local", "online"],
        default=None,
        help="生成模型来源 (默认: 读取.env配置)"
    )

    args = parser.parse_args()

    # 应用命令行配置
    if args.mode:
        settings.RETRIEVAL_MODE = args.mode
        apply_mode_preset(args.mode)

    if args.model:
        settings.LLM_MODE = args.model
        logger.info(f"🔧 命令行指定生成模型: {args.model.upper()}")

    # 执行主逻辑
    if args.rebuild:
        rebuild_index()
    else:
        run_interactive()


if __name__ == "__main__":
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("research_rag.log", encoding="utf-8", mode="a")
        ]
    )

    # 抑制不必要的警告
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    main()
