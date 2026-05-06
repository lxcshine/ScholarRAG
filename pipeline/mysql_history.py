# -*- coding: utf-8 -*-
"""
MySQL-based Conversation History Storage

Stores conversation sessions and messages in MySQL database.
Provides CRUD operations for session management.
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logger.warning("pymysql not installed, MySQL history storage disabled")


class MySQLHistory:
    """
    Manages conversation history in MySQL database.
    Tables: conversation_sessions, conversation_messages
    """

    def __init__(self, host: str = "localhost", port: int = 3306,
                 user: str = "root", password: str = "123456",
                 database: str = "research_rag"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._initialized = False

        if not MYSQL_AVAILABLE:
            logger.error("pymysql not available, MySQL history disabled")
            return

        try:
            self._init_database()
            self._initialized = True
            logger.info(f"MySQL history storage initialized: {host}:{port}/{database}")
        except Exception as e:
            logger.error(f"Failed to initialize MySQL history: {e}")
            self._initialized = False

    @contextmanager
    def _get_connection(self):
        conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_database(self):
        """Create database and tables if not exists."""
        conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            charset="utf8mb4"
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                cursor.execute(f"USE `{self.database}`")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_sessions (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(100) UNIQUE NOT NULL,
                        title VARCHAR(500) DEFAULT '',
                        summary TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        message_count INT DEFAULT 0,
                        INDEX idx_updated (updated_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(100) NOT NULL,
                        role VARCHAR(20) NOT NULL,
                        content LONGTEXT NOT NULL,
                        timestamp DOUBLE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_session (session_id),
                        INDEX idx_timestamp (timestamp)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)

                conn.commit()
                logger.info("Database tables created successfully")
        finally:
            conn.close()

    def save_message(self, session_id: str, role: str, content: str):
        """Append a message to the conversation."""
        if not self._initialized:
            return

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO conversation_messages (session_id, role, content, timestamp) "
                        "VALUES (%s, %s, %s, %s)",
                        (session_id, role, content, time.time())
                    )

                    cursor.execute(
                        "INSERT INTO conversation_sessions (session_id) VALUES (%s) "
                        "ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP",
                        (session_id,)
                    )

                    cursor.execute(
                        "UPDATE conversation_sessions SET message_count = "
                        "(SELECT COUNT(*) FROM conversation_messages WHERE session_id = %s) "
                        "WHERE session_id = %s",
                        (session_id, session_id)
                    )
        except Exception as e:
            logger.error(f"Failed to save message: {e}")

    def save_session_info(self, session_id: str, title: str = "", summary: str = ""):
        """Update session metadata."""
        if not self._initialized:
            return

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO conversation_sessions (session_id, title, summary) "
                        "VALUES (%s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE title = VALUES(title), summary = VALUES(summary), "
                        "updated_at = CURRENT_TIMESTAMP",
                        (session_id, title, summary)
                    )
        except Exception as e:
            logger.error(f"Failed to save session info: {e}")

    def load_messages(self, session_id: str) -> List[Dict]:
        """Load all messages for a session, ordered by timestamp."""
        if not self._initialized:
            return []

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT role, content, timestamp FROM conversation_messages "
                        "WHERE session_id = %s ORDER BY timestamp ASC",
                        (session_id,)
                    )
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to load messages: {e}")
            return []

    def load_session(self, session_id: str) -> Optional[Dict]:
        """Load session info with messages."""
        if not self._initialized:
            return None

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT session_id, title, summary, message_count, "
                        "DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s') as created_at, "
                        "DATE_FORMAT(updated_at, '%%Y-%%m-%%d %%H:%%i:%%s') as updated_at "
                        "FROM conversation_sessions WHERE session_id = %s",
                        (session_id,)
                    )
                    session = cursor.fetchone()
                    if session:
                        session["messages"] = self.load_messages(session_id)
                    return session
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None

    def list_sessions(self) -> List[Dict]:
        """List all sessions ordered by updated_at descending."""
        if not self._initialized:
            return []

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT session_id, title, summary, message_count, "
                        "DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s') as created_at, "
                        "DATE_FORMAT(updated_at, '%%Y-%%m-%%d %%H:%%i:%%s') as updated_at "
                        "FROM conversation_sessions ORDER BY updated_at DESC"
                    )
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        if not self._initialized:
            return False

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM conversation_messages WHERE session_id = %s", (session_id,))
                    cursor.execute("DELETE FROM conversation_sessions WHERE session_id = %s", (session_id,))
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False

    def rename_session(self, session_id: str, new_title: str) -> bool:
        """Rename a session."""
        if not self._initialized:
            return False

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE conversation_sessions SET title = %s, updated_at = CURRENT_TIMESTAMP "
                        "WHERE session_id = %s",
                        (new_title, session_id)
                    )
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to rename session: {e}")
            return False

    def get_session_count(self) -> int:
        """Get total number of sessions."""
        if not self._initialized:
            return 0

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) as count FROM conversation_sessions")
                    return cursor.fetchone()["count"]
        except Exception as e:
            logger.error(f"Failed to get session count: {e}")
            return 0


mysql_history = MySQLHistory()
