#!/usr/bin/env python3
"""
Shared LLM utility functions.

Centralizes JSON extraction logic used across CommandParser, RobotLLMAgent,
and RobotConstitution.  The canonical implementation follows CommandParser's
3-stage approach (direct → markdown block → bare regex), with JS-style comment
stripping on the code-block and bare-regex paths to handle LLMs that emit
// line comments inside JSON.
"""

import json
import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def extract_json(content: str) -> Optional[Dict]:
    """
    Extract a JSON object from an LLM response string.

    Stages (in order):
    1. Direct ``json.loads`` — succeeds when the model emits clean JSON.
    2. Markdown code-block extraction (```json ... ``` or ``` ... ```) with
       JS ``//`` comment stripping on the extracted block.
    3. Bare ``{...}`` regex match with JS comment stripping.

    Returns ``None`` and logs an error if all stages fail.

    Args:
        content: Raw LLM response text.

    Returns:
        Parsed dict, or None if the response could not be decoded.
    """
    # Stage 1: direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed: {e}")

    # Stage 2: markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Markdown JSON parse failed: {e}. "
                f"Content length: {len(json_str)}, preview: {json_str[:200]}"
            )
        # Retry after stripping JS-style // comments (LLMs often emit these)
        json_str_clean = re.sub(r"//[^\n]*", "", json_str)
        try:
            return json.loads(json_str_clean)
        except json.JSONDecodeError:
            pass

    # Stage 3: bare {…} regex
    json_match = re.search(r"\{.*?\}", content, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Regex JSON parse failed: {e}. Content length: {len(json_str)}"
            )
        json_str_clean = re.sub(r"//[^\n]*", "", json_str)
        try:
            return json.loads(json_str_clean)
        except json.JSONDecodeError:
            pass

    logger.error(
        f"All JSON extraction methods failed. "
        f"Response length: {len(content)}, preview: {content[:500]}"
    )
    return None
