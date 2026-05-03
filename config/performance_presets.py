# 性能预设配置
PRESETS = {
    "fast": {  # 快速模式：响应<30秒，适合简单查询
        "MAX_ITERATIONS": 1,
        "TOP_K_RETRIEVAL": 8,
        "TOP_N_RERANK": 3,
        "EMBEDDING_MODEL": "all-minilm",
        "ENABLE_HYDE": False,
        "ENABLE_QUERY_REWRITE": False,
    },
    "balanced": {  # 平衡模式：响应<2分钟，默认推荐
        "MAX_ITERATIONS": 2,
        "TOP_K_RETRIEVAL": 12,
        "TOP_N_RERANK": 5,
        "EMBEDDING_MODEL": "nomic-embed-text",
        "ENABLE_HYDE": True,
        "ENABLE_QUERY_REWRITE": True,
    },
    "accurate": {  # 精准模式：响应<5分钟，复杂查询用
        "MAX_ITERATIONS": 3,
        "TOP_K_RETRIEVAL": 20,
        "TOP_N_RERANK": 8,
        "EMBEDDING_MODEL": "bge-m3",
        "ENABLE_HYDE": True,
        "ENABLE_QUERY_REWRITE": True,
    }
}
