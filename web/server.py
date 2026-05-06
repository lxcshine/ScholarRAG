# -*- coding: utf-8 -*-
"""
FastAPI Web Server for Research RAG System
Provides REST API and SSE streaming for web interface
Optimized for fast response with lazy loading and session caching
"""

import os
import sys
import time
import json
import logging
import asyncio
import queue
import threading
from pathlib import Path
from typing import Optional, List, Dict, AsyncGenerator
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from core.vector_store import load_vector_store
from core.reranker import Reranker
from core.llm_router import LLMRouter
from core.table_processor import TableProcessor
from pipeline.context_manager import ContextManager
from pipeline.iterative_agent import IterativeRAGAgent
from pipeline.conversation_memory import ConversationMemory
from pipeline.persistent_history import persistent_history
from pipeline.mysql_history import mysql_history

logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=4)
table_processor = TableProcessor()


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class SessionInfo(BaseModel):
    session_id: Optional[str] = None


class SessionRename(BaseModel):
    session_id: str
    title: str


class SystemConfig(BaseModel):
    mode: Optional[str] = None
    model: Optional[str] = None


class SessionData:
    def __init__(self):
        self.agent = None
        self.ctx_mgr = None
        self.llm = None
        self.conv_memory = None
        self.created_at = time.time()
        self._initialized = False
    
    def initialize(self):
        if self._initialized:
            return
        
        vstore = load_vector_store()
        if not vstore:
            raise RuntimeError("Vector store not available")
        
        self.llm = LLMRouter.get_llm()
        reranker = Reranker()
        self.ctx_mgr = ContextManager(vstore, reranker, self.llm)
        self.conv_memory = ConversationMemory(
            max_turns=settings.MAX_CONVERSATION_TURNS,
            max_tokens=settings.MAX_CONVERSATION_TOKENS,
            enable_summary=settings.ENABLE_CONVERSATION_SUMMARY
        )
        self.agent = IterativeRAGAgent(self.ctx_mgr, self.llm, conversation_memory=self.conv_memory)
        self._initialized = True


active_sessions: Dict[str, SessionData] = {}
default_session = SessionData()


def get_session(session_id: str = None) -> SessionData:
    if not session_id or session_id == "default":
        return default_session
    
    if session_id not in active_sessions:
        active_sessions[session_id] = SessionData()
    
    return active_sessions[session_id]


def format_sse(data: str, event: str = None) -> str:
    msg = ""
    if event:
        msg += f"event: {event}\n"
    msg += f"data: {data}\n\n"
    return msg


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Research RAG Web Server starting...")
    try:
        default_session.initialize()
        logger.info("Default session initialized")
        
        # Load table schemas from vector store documents
        try:
            vstore = load_vector_store()
            if vstore:
                all_docs = vstore.get()
                if all_docs and all_docs.get("documents"):
                    for i, content in enumerate(all_docs["documents"]):
                        meta = all_docs["metadatas"][i] if all_docs.get("metadatas") else {}
                        doc_id = meta.get("doc_id", "unknown")
                        section = meta.get("section", "")
                        table_processor.extract_tables_from_markdown(
                            content,
                            doc_id=doc_id,
                            section=section
                        )
                    table_count = len(table_processor.table_schemas)
                    if table_count > 0:
                        logger.info(f"Loaded {table_count} table schemas from vector store")
                    else:
                        logger.info("No table schemas found in vector store")
        except Exception as e:
            logger.warning(f"Failed to load table schemas: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize default session: {e}")
    yield
    logger.info("Research RAG Web Server shutting down...")
    active_sessions.clear()


app = FastAPI(
    title="Research RAG System",
    description="Academic Paper RAG System with RF-Mem Memory Retrieval",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = BASE_DIR / "index.html"
    if frontend_path.exists():
        with open(frontend_path, "r", encoding="utf-8") as f:
            content = f.read()
        return Response(content=content, media_type="text/html; charset=utf-8")
    return HTMLResponse(content="<h1>Frontend not found. Please run from project root.</h1>", media_type="text/html; charset=utf-8")


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy" if default_session._initialized else "initializing",
        "llm_mode": settings.LLM_MODE,
        "retrieval_mode": settings.RETRIEVAL_MODE,
        "rf_mem_enabled": settings.ENABLE_RF_MEM,
        "conversation_memory": settings.ENABLE_CONVERSATION_MEMORY
    }


@app.post("/api/chat")
async def chat(request: ChatMessage):
    try:
        session = get_session(request.session_id)
        session.initialize()
        
        start_time = time.time()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, session.agent.run, request.message)
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "response": response,
            "elapsed": round(elapsed, 2),
            "session_id": request.session_id or "default"
        }
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatMessage):
    session = get_session(request.session_id)
    session.initialize()
    
    agent = session.agent
    loop = asyncio.get_event_loop()
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield format_sse(
                json.dumps({"status": "retrieving", "message": "Retrieving relevant documents..."}),
                "status"
            )
            
            start_time = time.time()
            
            if agent.conversation_memory and settings.ENABLE_CONVERSATION_MEMORY:
                agent.conversation_memory.add_user_message(request.message)
            
            session_id = request.session_id or "default"
            if settings.ENABLE_PERSISTENT_HISTORY:
                persistent_history.save_message(session_id, "user", request.message)
            if settings.ENABLE_MYSQL_HISTORY and mysql_history._initialized:
                mysql_history.save_message(session_id, "user", request.message)
            
            contextual_query = agent._build_contextual_query(request.message)
            query_emb = agent._get_query_embedding(contextual_query)
            
            docs = await loop.run_in_executor(executor, agent.context_mgr.retrieve, contextual_query, query_emb)
            
            if not docs:
                yield format_sse(
                    json.dumps({"status": "no_results", "message": "No relevant documents found."}),
                    "status"
                )
                yield format_sse(
                    json.dumps({
                        "response": "No relevant documents found in the knowledge base.",
                        "elapsed": round(time.time() - start_time, 2),
                        "doc_count": 0
                    }),
                    "complete"
                )
                return
            
            yield format_sse(
                json.dumps({
                    "status": "generating",
                    "message": f"Found {len(docs)} document(s), generating response...",
                    "doc_count": len(docs)
                }),
                "status"
            )
            
            from pipeline.context_compressor import compress_context
            context = await loop.run_in_executor(executor, compress_context, docs, request.message)
            prompt_text = agent._build_generation_prompt(request.message, context)
            
            token_queue = queue.Queue()
            full_response = []
            
            def generate_tokens():
                try:
                    for chunk in agent.llm.stream(prompt_text):
                        if hasattr(chunk, "content") and chunk.content:
                            token_queue.put(chunk.content)
                            full_response.append(chunk.content)
                    token_queue.put(None)
                except Exception as e:
                    token_queue.put(e)
            
            thread = threading.Thread(target=generate_tokens, daemon=True)
            thread.start()
            
            while True:
                try:
                    token = token_queue.get(timeout=30)
                    if token is None:
                        break
                    if isinstance(token, Exception):
                        raise token
                    
                    yield format_sse(
                        json.dumps({"token": token}),
                        "token"
                    )
                    await asyncio.sleep(0)
                except queue.Empty:
                    break
            
            thread.join(timeout=5)
            
            final_response = "".join(full_response)
            
            if agent.conversation_memory:
                agent.conversation_memory.add_ai_message(final_response)
                agent.conversation_memory.compress_history(agent.llm)
            
            if settings.ENABLE_PERSISTENT_HISTORY:
                session_id = request.session_id or "default"
                persistent_history.save_message(session_id, "assistant", final_response)
                if agent.conversation_memory:
                    messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in agent.conversation_memory.conversation_history
                    ]
                    persistent_history.save_conversation(
                        session_id,
                        messages,
                        agent.conversation_memory.session_summary
                    )
            
            if settings.ENABLE_MYSQL_HISTORY and mysql_history._initialized:
                session_id = request.session_id or "default"
                mysql_history.save_message(session_id, "assistant", final_response)
                if agent.conversation_memory and agent.conversation_memory.turn_count <= 2:
                    title = request.message[:80] + ("..." if len(request.message) > 80 else "")
                    mysql_history.save_session_info(
                        session_id,
                        title=title,
                        summary=agent.conversation_memory.session_summary
                    )
            
            elapsed = round(time.time() - start_time, 2)
            
            doc_sources = []
            for doc in docs[:5]:
                doc_sources.append({
                    "title": doc.metadata.get("title", "\u672a\u77e5"),
                    "year": doc.metadata.get("year", "?"),
                    "section": doc.metadata.get("section", ""),
                    "doc_id": doc.metadata.get("doc_id", "")
                })
            
            yield format_sse(
                json.dumps({
                    "response": final_response,
                    "elapsed": elapsed,
                    "doc_count": len(docs),
                    "sources": doc_sources
                }),
                "complete"
            )
            
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield format_sse(
                json.dumps({"error": str(e)}),
                "error"
            )
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/sessions")
async def list_sessions():
    sessions = []
    for sid, data in active_sessions.items():
        sessions.append({
            "session_id": sid,
            "turn_count": data.conv_memory.turn_count if data.conv_memory else 0,
            "created_at": data.created_at
        })
    return {"sessions": sessions}


@app.post("/api/session/clear")
async def clear_session(request: SessionInfo):
    session_id = request.session_id or "default"
    session = get_session(session_id)
    if session.conv_memory:
        session.conv_memory.clear()
    return {"success": True, "message": "Session cleared"}


@app.get("/api/history")
async def get_history(session_id: str = "default"):
    if settings.ENABLE_MYSQL_HISTORY and mysql_history._initialized:
        session = mysql_history.load_session(session_id)
        if session:
            return {
                "success": True,
                "session_id": session["session_id"],
                "messages": session.get("messages", []),
                "summary": session.get("summary", ""),
                "message_count": session.get("message_count", 0)
            }
    
    history = persistent_history.load_history(session_id)
    if history:
        return {
            "success": True,
            "session_id": history.get("session_id"),
            "messages": history.get("messages", []),
            "summary": history.get("summary", ""),
            "message_count": history.get("message_count", 0)
        }
    return {"success": True, "messages": [], "summary": "", "message_count": 0}


@app.get("/api/history/sessions")
async def list_history_sessions():
    if settings.ENABLE_MYSQL_HISTORY and mysql_history._initialized:
        sessions = mysql_history.list_sessions()
        return {"sessions": sessions}
    
    sessions = persistent_history.list_sessions()
    return {"sessions": sessions}


@app.delete("/api/history/{session_id}")
async def delete_history_session(session_id: str):
    mysql_deleted = False
    file_deleted = False
    
    if settings.ENABLE_MYSQL_HISTORY and mysql_history._initialized:
        mysql_deleted = mysql_history.delete_session(session_id)
    
    file_deleted = persistent_history.delete_session(session_id)
    
    if session_id in active_sessions:
        session = active_sessions.pop(session_id)
        if session.conv_memory:
            session.conv_memory.clear()
    
    return {
        "success": mysql_deleted or file_deleted,
        "message": f"Session {session_id} deleted"
    }


@app.put("/api/history/{session_id}/rename")
async def rename_history_session(session_id: str, request: SessionRename):
    mysql_renamed = False
    file_renamed = False
    
    if settings.ENABLE_MYSQL_HISTORY and mysql_history._initialized:
        mysql_renamed = mysql_history.rename_session(session_id, request.title)
    
    file_renamed = persistent_history.rename_session(session_id, request.title)
    
    return {
        "success": mysql_renamed or file_renamed,
        "message": f"Session {session_id} renamed to {request.title}"
    }


@app.post("/api/config")
async def update_config(config: SystemConfig):
    if config.mode and config.mode in ["fast", "balanced", "accurate"]:
        settings.RETRIEVAL_MODE = config.mode
        presets = {
            "fast": {"MAX_ITERATIONS": 1, "TOP_K_RETRIEVAL": 10, "TOP_N_RERANK": 3},
            "balanced": {"MAX_ITERATIONS": 2, "TOP_K_RETRIEVAL": 15, "TOP_N_RERANK": 5},
            "accurate": {"MAX_ITERATIONS": 3, "TOP_K_RETRIEVAL": 25, "TOP_N_RERANK": 8}
        }
        for k, v in presets.get(config.mode, {}).items():
            setattr(settings, k, v)
    
    if config.model and config.model in ["local", "online"]:
        LLMRouter.switch_mode(config.model)
    
    return {
        "success": True,
        "mode": settings.RETRIEVAL_MODE,
        "model": settings.LLM_MODE
    }


@app.get("/api/config")
async def get_config():
    return {
        "retrieval_mode": settings.RETRIEVAL_MODE,
        "llm_mode": settings.LLM_MODE,
        "local_model": settings.LOCAL_LLM_MODEL,
        "online_model": settings.ONLINE_LLM_MODEL,
        "rf_mem_enabled": settings.ENABLE_RF_MEM,
        "conversation_memory": settings.ENABLE_CONVERSATION_MEMORY
    }


def create_app():
    return app
