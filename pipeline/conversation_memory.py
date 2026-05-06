# -*- coding: utf-8 -*-
"""
Conversation Memory Manager

Features:
  - Maintain multi-turn conversation history
  - Generate context-aware queries
  - Support conversation summarization and memory compression
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Conversation History Manager
    
    Maintains conversation context, supports:
      - History message storage and retrieval
      - Context window management (prevent exceeding token limits)
      - Conversation summary compression
      - Query rewriting (combining historical context)
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 8000,
        enable_summary: bool = True
    ):
        """
        Initialize conversation memory
        
        Args:
            max_turns: Maximum number of conversation turns to keep
            max_tokens: Maximum token limit
            enable_summary: Whether to enable conversation summarization
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.enable_summary = enable_summary
        
        self.messages: List[BaseMessage] = []
        self.conversation_history: List[Dict] = []
        self.session_summary: str = ""
        self.session_start_time = time.time()
        self.turn_count = 0
        
        logger.info(
            f"Conversation Memory initialized | "
            f"Max turns: {max_turns} | Max tokens: {max_tokens}"
        )

    def add_user_message(self, content: str):
        """Add user message"""
        self.messages.append(HumanMessage(content=content))
        self.conversation_history.append({
            "role": "user",
            "content": content,
            "timestamp": time.time()
        })
        self.turn_count += 1
        logger.debug(f"User message recorded | Current turns: {self.turn_count}")

    def add_ai_message(self, content: str):
        """Add AI response"""
        self.messages.append(AIMessage(content=content))
        self.conversation_history.append({
            "role": "assistant",
            "content": content,
            "timestamp": time.time()
        })
        logger.debug("AI response recorded")

    def get_recent_messages(self, n: Optional[int] = None) -> List[BaseMessage]:
        """
        Get recent n conversation messages
        
        Args:
            n: Return last n messages, None returns all
            
        Returns:
            Message list
        """
        if n is None:
            return self.messages
        
        return self.messages[-n:]

    def get_conversation_context(self) -> str:
        """
        Get formatted conversation context string
        
        Returns:
            Formatted conversation history text
        """
        parts = []
        
        if self.session_summary:
            parts.append(f"[Conversation Summary]\n{self.session_summary}\n")
        
        recent = self.conversation_history[-self.max_turns * 2:]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content']}")
        
        return "\n\n".join(parts)

    def get_query_with_context(self, current_query: str) -> str:
        """
        Generate query with context (for retrieval enhancement)
        
        Args:
            current_query: Current user input
            
        Returns:
            Complete query combined with historical context
        """
        if not self.conversation_history:
            return current_query
        
        context_parts = []
        
        if self.session_summary:
            context_parts.append(f"[Conversation Background] {self.session_summary}")
        
        recent_turns = self.conversation_history[-self.max_turns * 2:]
        if recent_turns:
            context_parts.append("[History]")
            for msg in recent_turns:
                role = "User" if msg["role"] == "user" else "Assistant"
                context_parts.append(f"{role}: {msg['content']}")
        
        context_parts.append(f"[Current Question] {current_query}")
        
        return "\n".join(context_parts)

    def compress_history(self, llm=None):
        """
        Compress conversation history into summary
        
        Args:
            llm: Optional LLM instance for generating summary
        """
        if not self.enable_summary or not self.conversation_history:
            return
        
        if llm and len(self.conversation_history) > self.max_turns:
            try:
                from langchain_core.prompts import ChatPromptTemplate
                
                summary_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a conversation summarization assistant. Compress the following conversation history into a concise summary, preserving key information and context."),
                    ("human", "Please summarize the following conversation:\n\n{history}")
                ])
                
                history_text = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in self.conversation_history[-self.max_turns * 2:]
                ])
                
                response = (summary_prompt | llm).invoke({"history": history_text})
                self.session_summary = response.content
                
                self.conversation_history = self.conversation_history[-self.max_turns:]
                self.messages = self.messages[-self.max_turns * 2:]
                
                logger.info(f"Conversation history compressed | Summary length: {len(self.session_summary)}")
                
            except Exception as e:
                logger.warning(f"Failed to generate conversation summary: {e}")
        else:
            self.conversation_history = self.conversation_history[-self.max_turns:]
            self.messages = self.messages[-self.max_turns * 2:]

    def clear(self):
        """Clear conversation history"""
        self.messages.clear()
        self.conversation_history.clear()
        self.session_summary = ""
        self.turn_count = 0
        self.session_start_time = time.time()
        logger.info("Conversation history cleared")

    def get_session_info(self) -> Dict:
        """Get session information"""
        return {
            "turn_count": self.turn_count,
            "message_count": len(self.messages),
            "session_duration": time.time() - self.session_start_time,
            "has_summary": bool(self.session_summary)
        }

    def is_empty(self) -> bool:
        """Check if conversation is empty"""
        return len(self.messages) == 0

    def get_last_user_query(self) -> Optional[str]:
        """Get last user input"""
        for msg in reversed(self.conversation_history):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def get_last_ai_response(self) -> Optional[str]:
        """Get last AI response"""
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant":
                return msg["content"]
        return None
