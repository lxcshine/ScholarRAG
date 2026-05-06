# -*- coding: utf-8 -*-
"""
Persistent Conversation History Storage

Saves and loads conversation history to/from JSON files.
Survives server restarts and browser refreshes.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class PersistentHistory:
    """
    Manages persistent storage of conversation history.
    Each conversation is stored as a separate JSON file.
    """

    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = str(Path(__file__).resolve().parent.parent / "history")
        
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        logger.info(f"Persistent history storage: {self.storage_dir}")

    def _get_file_path(self, session_id: str) -> str:
        safe_id = session_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return os.path.join(self.storage_dir, f"{safe_id}.json")

    def save_message(self, session_id: str, role: str, content: str):
        """Append a message to the conversation history file."""
        file_path = self._get_file_path(session_id)
        
        history = self._load_history(file_path)
        history["messages"].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        history["updated_at"] = time.time()
        history["message_count"] = len(history["messages"])
        
        self._save_history(file_path, history)

    def save_conversation(self, session_id: str, messages: List[Dict], summary: str = ""):
        """Save entire conversation at once."""
        file_path = self._get_file_path(session_id)
        
        existing = self._load_history(file_path)
        
        title = existing.get("title", "")
        if not title and messages:
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    title = content[:80] + ("..." if len(content) > 80 else "")
                    break
        
        history = {
            "session_id": session_id,
            "title": title,
            "messages": messages,
            "summary": summary,
            "created_at": existing.get("created_at", time.time()),
            "updated_at": time.time(),
            "message_count": len(messages)
        }
        
        self._save_history(file_path, history)

    def load_history(self, session_id: str) -> Optional[Dict]:
        """Load conversation history for a session."""
        file_path = self._get_file_path(session_id)
        return self._load_history(file_path)

    def list_sessions(self) -> List[Dict]:
        """List all saved conversation sessions."""
        sessions = []
        if not os.path.exists(self.storage_dir):
            return sessions
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.storage_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sessions.append({
                        "session_id": data.get("session_id", filename.replace(".json", "")),
                        "title": data.get("title", ""),
                        "message_count": data.get("message_count", 0),
                        "created_at": data.get("created_at", 0),
                        "updated_at": data.get("updated_at", 0),
                        "summary": data.get("summary", "")[:100]
                    })
                except Exception as e:
                    logger.warning(f"Failed to load session {filename}: {e}")
        
        sessions.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        file_path = self._get_file_path(session_id)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Session deleted: {session_id}")
            return True
        return False

    def rename_session(self, session_id: str, new_title: str) -> bool:
        """Rename a conversation session."""
        file_path = self._get_file_path(session_id)
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["title"] = new_title
            data["updated_at"] = time.time()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Session renamed: {session_id} -> {new_title}")
            return True
        except Exception as e:
            logger.error(f"Failed to rename session {session_id}: {e}")
            return False

    def clear_all(self):
        """Clear all conversation history."""
        if os.path.exists(self.storage_dir):
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    os.remove(os.path.join(self.storage_dir, filename))
            logger.info("All conversation history cleared")

    def _load_history(self, file_path: str) -> Dict:
        """Load history from file."""
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load history from {file_path}: {e}")
        
        return {
            "session_id": Path(file_path).stem,
            "messages": [],
            "summary": "",
            "created_at": time.time(),
            "updated_at": time.time(),
            "message_count": 0
        }

    def _save_history(self, file_path: str, history: Dict):
        """Save history to file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history to {file_path}: {e}")


persistent_history = PersistentHistory()
