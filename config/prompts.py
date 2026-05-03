QUERY_EXPANSION_PROMPT = """你是一个学术检索专家。将以下用户问题重写为3个不同视角的检索Query，聚焦方法论、数据集、评估指标。
Original Query: {query}
Output JSON list of strings only."""

HYDE_PROMPT = """请基于学术常识，为以下问题生成一段“假设性答案/核心结论摘要”。该文本仅用于提升向量检索的语义匹配度，无需完全准确。
Query: {query}
Hypothetical Answer:"""

SUFFICIENCY_CHECK_PROMPT = """根据当前检索到的上下文，判断是否已足够回答用户问题。若否，明确指出缺失的关键信息（如具体模型名称、数据集指标、对比实验等）。
Query: {query}
Context Snippet: {context_preview}
Output JSON only: {{"is_sufficient": true/false, "missing_info": "string"}}"""

FINAL_GENERATION_PROMPT = """你是一个严谨的科研助手。请严格基于提供的上下文回答问题。
规则：
1. 每个事实/数据/结论后必须标注来源，格式为 [DocID: xxx]。
2. 涉及指标对比时，必须使用Markdown表格。
3. 若上下文中未提及，明确回答“未在文献库中找到相关数据”，严禁编造。
4. 保持学术客观语气，结构清晰。
Context: {context}
Query: {query}
Answer:"""
