#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research RAG System Main Entry
Supports local Ollama / online GLM-4.5 one-click switching
Embedding model fixed to local qwen3-embedding:0.6b
"""

import sys
import os
import argparse
import logging
import time
import requests
from pathlib import Path

from langchain_ollama import ChatOllama

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from utils.pdf_extractor import extract_pdf
from utils.chunker import chunk_documents
from core.vector_store import init_vector_store, load_vector_store
from core.reranker import Reranker
from core.llm_router import LLMRouter
from core.table_processor import TableProcessor
from pipeline.context_manager import ContextManager
from pipeline.iterative_agent import IterativeRAGAgent
from pipeline.conversation_memory import ConversationMemory
from generator.citation_validator import validate_citations

logger = logging.getLogger(__name__)

table_processor = TableProcessor()


def apply_mode_preset(mode: str):
    """Dynamically adjust retrieval strategy parameters based on mode"""
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
    logger.info(f"Retrieval mode switched: {mode.upper()} | Iterations: {settings.MAX_ITERATIONS} | Top-K: {settings.TOP_K_RETRIEVAL}")


def check_llm_health() -> bool:
    """Check if the configured LLM service is available"""
    mode = settings.LLM_MODE
    if mode == "local":
        try:
            resp = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if settings.LOCAL_LLM_MODEL in models:
                    logger.info(f"Local model ready: {settings.LOCAL_LLM_MODEL}")
                    return True
                else:
                    logger.warning(f"Model {settings.LOCAL_LLM_MODEL} not found, available: {models}")
                    return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
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
                logger.info(f"Online model ready: {settings.ONLINE_LLM_MODEL}")
                return True
            else:
                logger.error(f"GLM API returned error: {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Cannot connect to GLM service: {e}")
            return False
    return False


def rebuild_index(force_rebuild: bool = False):
    """Rebuild or incrementally update vector database"""
    logger.info("Starting vector index update...")
    pdf_dir = Path(settings.PDF_DIR)

    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created: {pdf_dir}")
        logger.info("Please place CVPR/IEEE/NeurIPS PDF papers in this directory and retry")
        return

    # Check for existing vector store and indexed files
    vstore = load_vector_store() if not force_rebuild else None
    indexed_files = set()
    
    if vstore:
        try:
            all_docs = vstore.get()
            if all_docs and all_docs.get("metadatas"):
                for meta in all_docs["metadatas"]:
                    if "source_file" in meta:
                        indexed_files.add(meta["source_file"])
                logger.info(f"Found existing index with {len(indexed_files)} files")
        except Exception as e:
            logger.warning(f"Failed to read existing index metadata: {e}")
            vstore = None

    # Find new PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return

    new_files = []
    existing_files = []
    for p in pdf_files:
        if str(p) in indexed_files or p.name in indexed_files:
            existing_files.append(p)
        else:
            new_files.append(p)

    if not new_files and not force_rebuild:
        logger.info(f"All {len(pdf_files)} files already indexed. No updates needed.")
        logger.info("Use --rebuild to force full reindex, or add new PDFs to papers/ directory")
        return

    if new_files:
        logger.info(f"Found {len(new_files)} new file(s) to index")
        for p in new_files:
            logger.info(f"  New: {p.name}")
    
    if existing_files and force_rebuild:
        logger.info(f"Force rebuild: re-indexing all {len(pdf_files)} files")

    files_to_process = new_files if not force_rebuild else pdf_files

    # Parse PDFs
    docs = []
    for p in files_to_process:
        logger.info(f"Parsing: {p.name}")
        try:
            doc = extract_pdf(str(p))
            doc.metadata["source_file"] = str(p)
            doc.metadata["source_filename"] = p.name
            docs.append(doc)
        except Exception as e:
            logger.error(f"Failed to parse {p.name}: {e}")

    valid_docs = [d for d in docs if d.page_content.strip()]
    if not valid_docs:
        logger.error("No valid content found, index terminated")
        return

    # Advanced chunking
    logger.info(f"Starting chunking: {len(valid_docs)} papers")
    chunks = chunk_documents(valid_docs)
    logger.info(f"Chunking complete: {len(chunks)} index chunks")

    # Extract table schemas from documents
    logger.info("Extracting table schemas...")
    for doc in valid_docs:
        doc_id = doc.metadata.get("doc_id", "unknown")
        section = doc.metadata.get("section", "")
        try:
            schemas = table_processor.extract_tables_from_markdown(
                doc.page_content,
                doc_id=doc_id,
                section=section
            )
            if schemas:
                logger.info(f"Extracted {len(schemas)} table schemas from {doc_id}")
        except Exception as e:
            logger.warning(f"Failed to extract tables from {doc_id}: {e}")

    # Add table schema documents to vector store
    table_docs = table_processor.generate_schema_docs()
    if table_docs:
        logger.info(f"Adding {len(table_docs)} table schema documents to vector store")
        chunks.extend(table_docs)

    # Initialize or update vector store
    if vstore and not force_rebuild and new_files:
        logger.info(f"Adding {len(chunks)} new chunks to existing index...")
        from core.embedder import get_embedder
        embedder = get_embedder()
        
        batch_size = settings.EMBEDDING_BATCH_SIZE
        total = len(chunks)
        logger.info(f"Embedding new content: {total} chunks in batches of {batch_size}")
        
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            vstore.add_documents(batch)
            processed = min(i + batch_size, total)
            progress = processed / total * 100
            logger.info(f"Embedding progress: {processed}/{total} ({progress:.1f}%)")
        
        logger.info(f"Incremental index update complete! Total files: {len(pdf_files)}")
    else:
        logger.info("Building vector index...")
        init_vector_store(chunks)
        logger.info(f"Full index rebuild complete!")
    
    logger.info(f"Index update finished. Time: {time.time():.1f}s")


def run_interactive():
    """Start interactive Q&A loop"""
    # Health check
    if not check_llm_health():
        logger.error("LLM service not ready, please check configuration and retry")
        return

    # Load vector store
    vstore = load_vector_store()
    if not vstore:
        logger.info("Vector store not detected, auto-triggering rebuild...")
        rebuild_index()
        vstore = load_vector_store()
        if not vstore:
            logger.error("Vector store loading failed, exiting")
            return

    # Initialize core components
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
    conv_memory = ConversationMemory(
        max_turns=settings.MAX_CONVERSATION_TURNS,
        max_tokens=settings.MAX_CONVERSATION_TOKENS,
        enable_summary=settings.ENABLE_CONVERSATION_SUMMARY
    )
    agent = IterativeRAGAgent(ctx_mgr, llm, conversation_memory=conv_memory)

    # Startup info
    print("\n" + "=" * 70)
    print("RAG Research System Started")
    print(f"   Mode: {settings.RETRIEVAL_MODE.upper()}")
    print(f"   Generation Model: {settings.LLM_MODE.upper()} ({settings.ONLINE_LLM_MODEL if settings.LLM_MODE == 'online' else settings.LOCAL_LLM_MODEL})")
    print(f"   Embedding Model: {settings.EMBEDDING_MODEL} (local)")
    print(f"   Memory Retrieval: {'RF-Mem Dual-Path' if settings.ENABLE_RF_MEM else 'Standard'}")
    print(f"   Conversation Memory: {'Enabled' if settings.ENABLE_CONVERSATION_MEMORY else 'Disabled'}")
    print("=" * 70)
    print("Quick Commands:")
    print("   - mode fast|balanced|accurate  : Switch retrieval strategy")
    print("   - model local|online           : Switch generation model")
    print("   - memory status/clear          : View/clear conversation memory")
    print("   - exit / quit / q              : Exit system")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nYour query: ").strip()
            if not user_input:
                continue

            # Command: Switch retrieval mode
            if user_input.lower().startswith("mode "):
                parts = user_input.split()
                if len(parts) == 2 and parts[1] in ["fast", "balanced", "accurate"]:
                    apply_mode_preset(parts[1])
                    settings.RETRIEVAL_MODE = parts[1]
                    agent = IterativeRAGAgent(ctx_mgr, llm, conversation_memory=conv_memory)
                    print(f"Retrieval strategy switched: {parts[1].upper()}")
                else:
                    print("Usage: mode fast | balanced | accurate")
                continue

            # Command: Switch generation model
            if user_input.lower().startswith("model "):
                parts = user_input.split()
                if len(parts) == 2 and parts[1] in ["local", "online"]:
                    if LLMRouter.switch_mode(parts[1]):
                        llm = LLMRouter.get_llm()
                        ctx_mgr.llm = llm
                        agent = IterativeRAGAgent(ctx_mgr, llm, conversation_memory=conv_memory)
                        model_name = settings.ONLINE_LLM_MODEL if parts[1] == "online" else settings.LOCAL_LLM_MODEL
                        print(f"Generation model switched: {parts[1].upper()} ({model_name})")
                    else:
                        print("Switch failed, please check configuration")
                else:
                    print("Usage: model local | online")
                continue

            # Command: Conversation memory management
            if user_input.lower().startswith("memory "):
                parts = user_input.split()
                if len(parts) == 2:
                    if parts[1] == "status":
                        if conv_memory:
                            info = conv_memory.get_session_info()
                            print(f"\nConversation Memory Status:")
                            print(f"   Turns: {info['turn_count']}")
                            print(f"   Messages: {info['message_count']}")
                            print(f"   Session Duration: {info['session_duration']:.0f}s")
                            print(f"   Summary: {'Yes' if info['has_summary'] else 'No'}")
                        else:
                            print("Conversation memory not enabled")
                    elif parts[1] == "clear":
                        if conv_memory:
                            conv_memory.clear()
                            print("Conversation memory cleared")
                        else:
                            print("Conversation memory not enabled")
                    else:
                        print("Usage: memory status | clear")
                continue

            # Exit command
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nThank you for using, exiting safely.")
                break

            # Execute retrieval and generation
            start_time = time.time()
            print(f"Executing intelligent retrieval (mode: {settings.RETRIEVAL_MODE})...")

            raw_response = agent.run(user_input)
            elapsed = time.time() - start_time

            # Post-processing: Citation validation
            validated_response = validate_citations(raw_response, getattr(ctx_mgr, 'last_docs', []))

            # Output results
            print(f"\nTime: {elapsed:.2f}s")
            print("=" * 70)
            print("Generated Result:")
            print("=" * 70)
            print(validated_response)

            # Persistent save
            timestamp = int(time.time())
            output_file = f"answer_{timestamp}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Query: {user_input}\n")
                f.write(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Mode: {settings.RETRIEVAL_MODE} | Model: {settings.LLM_MODE}\n\n")
                f.write(validated_response)
            print(f"\nAutomatically saved to: {output_file}")

        except KeyboardInterrupt:
            print("\n\nInterrupt detected, exiting safely...")
            break
        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
            print(f"Error occurred: {type(e).__name__}")
            print("Suggestion: Check logs or retry, try mode accurate for complex queries")


def run_web_server(host: str = "0.0.0.0", port: int = 8000):
    """Start web server with browser auto-open after server is ready"""
    import webbrowser
    import threading
    import uvicorn
    import time
    import socket
    
    url = f"http://localhost:{port}"
    
    def wait_for_server_ready():
        """Poll the port until server is ready, then open browser"""
        max_wait = 120  # Maximum wait time in seconds
        check_interval = 0.5  # Check every 0.5 seconds
        elapsed = 0
        
        print("\nWaiting for server to be ready...")
        
        while elapsed < max_wait:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                
                if result == 0:
                    print(f"Server is ready! Opening browser...")
                    time.sleep(0.5)  # Small delay to ensure server is fully initialized
                    webbrowser.open(url)
                    return
            except Exception:
                pass
            
            time.sleep(check_interval)
            elapsed += check_interval
        
        print(f"Warning: Server did not become ready within {max_wait} seconds")
    
    # Start browser opener thread
    browser_thread = threading.Thread(target=wait_for_server_ready, daemon=True)
    browser_thread.start()
    
    print("\n" + "=" * 70)
    print("Research RAG Web Server Starting...")
    print(f"   Access URL: {url}")
    print(f"   Mode: {settings.RETRIEVAL_MODE.upper()}")
    print(f"   Generation Model: {settings.LLM_MODE.upper()}")
    print(f"   Memory Retrieval: {'RF-Mem Dual-Path' if settings.ENABLE_RF_MEM else 'Standard'}")
    print("=" * 70)
    print("Press Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from web.server import app
    
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    """Main entry"""
    parser = argparse.ArgumentParser(
        description="Research RAG System | Supports local/online model hybrid deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start web interface (recommended)
  python main.py --web

  # Fast mode + online model (recommended for 8GB VRAM users)
  python main.py --mode fast --model online

  # High accuracy mode + local model (requires large VRAM)
  python main.py --mode accurate --model local

  # Rebuild index
  python main.py --rebuild
        """
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild vector database and BM25 index"
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "accurate"],
        default="fast",
        help="Retrieval strategy mode (default: fast)"
    )
    parser.add_argument(
        "--model",
        choices=["local", "online"],
        default=None,
        help="Generation model source (default: read from .env)"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start web interface with browser auto-open"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Web server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web server port (default: 8000)"
    )

    args = parser.parse_args()

    # Apply command line configuration
    if args.mode:
        settings.RETRIEVAL_MODE = args.mode
        apply_mode_preset(args.mode)

    if args.model:
        settings.LLM_MODE = args.model
        logger.info(f"Command line specified generation model: {args.model.upper()}")

    # Execute main logic
    if args.rebuild:
        rebuild_index()
    elif args.web:
        run_web_server(host=args.host, port=args.port)
    else:
        run_interactive()


if __name__ == "__main__":
    # Configure log format
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("research_rag.log", encoding="utf-8", mode="a")
        ]
    )

    # Suppress unnecessary warnings
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    main()
