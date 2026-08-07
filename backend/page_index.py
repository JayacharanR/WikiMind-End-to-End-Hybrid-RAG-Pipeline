"""PageIndex Vectorless Tree Navigation.

Implements the L3 retrieval fallback: structural navigation of Wikipedia articles.
Given an article's raw markdown text, it parses the headers into a hierarchical
Table of Contents (ToC) JSON tree. An LLM reasons over this ToC to select the
most relevant section(s) for a query, allowing exact extraction of section content
while preserving full structural context, completely bypassing vector similarity
limitations.

Parsed trees are cached in Redis to avoid re-parsing on subsequent queries.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.cache import get_redis_client
from backend.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

NAVIGATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Wikipedia navigator. Given a user query and the Table of Contents (ToC) of a Wikipedia article, identify the section(s) most likely to contain the answer. Return ONLY a JSON list of the exact section titles you want to read. Do not include markdown formatting like ```json in your response. If no section seems relevant, return an empty list []."),
    ("user", "Query: {query}\n\nArticle ToC:\n{toc}\n\nSelected Sections (JSON list):")
])


# ---------------------------------------------------------------------------
# Tree Parsing
# ---------------------------------------------------------------------------

def parse_article_tree(markdown_text: str) -> Dict[str, Any]:
    """Parse Wikipedia markdown text into a hierarchical JSON tree.
    
    Extracts standard markdown headers (# Header, == Header ==) and builds
    a nested dictionary representing the document structure, with the actual
    text content stored at the leaves.
    
    Args:
        markdown_text: Raw Wikipedia article text.
        
    Returns:
        Dict representing the structural hierarchy.
    """
    # Simplified parser for MVP. In a production setting, this would handle
    # MediaWiki syntax natively or rely on a robust markdown AST parser.
    
    tree = {"title": "Root", "level": 0, "content": "", "children": {}}
    current_path = [tree]
    
    lines = markdown_text.split("\n")
    buffer = []
    
    def flush_buffer():
        if buffer and current_path:
            current_path[-1]["content"] += "\n".join(buffer) + "\n"
        buffer.clear()
        
    # Regex to match Wikipedia-style headers like == Section == or markdown ## Section
    header_regex = re.compile(r'^(={2,6})\s*(.*?)\s*\1$|^(#{2,6})\s*(.*)$')
    
    for line in lines:
        match = header_regex.match(line)
        if match:
            flush_buffer()
            # Determine level and title
            if match.group(1):
                level = len(match.group(1))
                title = match.group(2).strip()
            else:
                level = len(match.group(3))
                title = match.group(4).strip()
                
            new_node = {"title": title, "level": level, "content": "", "children": {}}
            
            # Pop up the tree until we find the right parent
            while current_path and current_path[-1]["level"] >= level:
                current_path.pop()
                
            if not current_path:
                current_path = [tree]
                
            current_path[-1]["children"][title] = new_node
            current_path.append(new_node)
        else:
            buffer.append(line)
            
    flush_buffer()
    return tree


def _extract_toc(tree: Dict[str, Any], indent: int = 0) -> str:
    """Recursively format the tree structure into a readable string ToC."""
    toc = ""
    if tree["level"] > 0:
        toc += "  " * (indent - 1) + f"- {tree['title']}\n"
    
    for child in tree.get("children", {}).values():
        toc += _extract_toc(child, indent + 1)
        
    return toc


def _find_section(tree: Dict[str, Any], section_title: str) -> Optional[Dict[str, Any]]:
    """Recursively search for a section by exact title."""
    if tree.get("title") == section_title:
        return tree
        
    for child in tree.get("children", {}).values():
        result = _find_section(child, section_title)
        if result:
            return result
            
    return None


def _get_full_section_content(node: Dict[str, Any]) -> str:
    """Get the text of a section and all its nested subsections."""
    content = f"## {node.get('title', '')}\n{node.get('content', '')}"
    for child in node.get("children", {}).values():
        content += _get_full_section_content(child)
    return content


# ---------------------------------------------------------------------------
# Redis Caching
# ---------------------------------------------------------------------------

async def _get_cached_tree(article_title: str) -> Optional[Dict[str, Any]]:
    """Retrieve a parsed article tree from Redis L1 cache.

    Returns None gracefully if Redis is unavailable.
    """
    try:
        client = await get_redis_client()
        if client is None:
            return None
        key = f"wikimind:pageindex:{article_title}"
        data = await client.get(key)
        if data:
            return json.loads(data)
    except Exception as exc:
        logger.debug("Redis PageIndex cache miss for %s: %s", article_title, exc)
    return None


async def _set_cached_tree(article_title: str, tree: Dict[str, Any]) -> None:
    """Store a parsed article tree in Redis L1 cache (24h TTL).

    Silently skips if Redis is unavailable.
    """
    try:
        client = await get_redis_client()
        if client is None:
            return
        key = f"wikimind:pageindex:{article_title}"
        await client.setex(key, 86400, json.dumps(tree))
    except Exception as exc:
        logger.debug("Failed to cache PageIndex tree for %s: %s", article_title, exc)


# ---------------------------------------------------------------------------
# Navigation Logic
# ---------------------------------------------------------------------------

async def navigate_article(query: str, article_title: str, markdown_text: str) -> str:
    """Navigate an article using LLM-driven ToC reasoning to extract relevant sections.
    
    Args:
        query: The user's search query.
        article_title: The title of the article (used for caching).
        markdown_text: The full raw markdown text of the article.
        
    Returns:
        The extracted section content preserving structure, or empty string if
        no relevant sections were found.
    """
    # 1. Parse or Retrieve Tree
    tree = await _get_cached_tree(article_title)
    if not tree:
        logger.debug("Parsing PageIndex tree for %s...", article_title)
        tree = parse_article_tree(markdown_text)
        await _set_cached_tree(article_title, tree)
    else:
        logger.debug("Loaded PageIndex tree for %s from cache.", article_title)
        
    # 2. Extract ToC
    toc = _extract_toc(tree)
    if not toc.strip():
        # If no ToC could be parsed, just return the root content (it's a short article)
        return tree.get("content", "")
        
    # 3. LLM Reasoning
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.llm_base_url,
        temperature=0.0,
    )
    
    chain = NAVIGATE_PROMPT | llm
    
    try:
        logger.debug("Asking LLM to navigate ToC for query: %s", query)
        res = await chain.ainvoke({"query": query, "toc": toc})
        content = res.content if hasattr(res, "content") else str(res)
        
        # Parse the JSON list
        try:
            selected_sections = json.loads(content)
            if not isinstance(selected_sections, list):
                selected_sections = []
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM ToC selection as JSON: %s", content)
            selected_sections = []
            
        # 4. Extract Content
        extracted_content = ""
        for section_title in selected_sections:
            node = _find_section(tree, section_title)
            if node:
                extracted_content += _get_full_section_content(node) + "\n\n"
                
        if extracted_content:
            logger.info("PageIndex extracted %d sections from %s.", len(selected_sections), article_title)
            return extracted_content.strip()
            
        logger.info("PageIndex found no relevant sections in %s.", article_title)
        return ""
        
    except Exception as exc:
        logger.error("PageIndex navigation failed: %s", exc)
        return ""
