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

# Matches 3-element Python tuple coordinate syntax: (x, y, z)
# LLMs sometimes emit these instead of JSON arrays when generating coordinate params.
_TUPLE_RE = re.compile(
    r"\((-?[\d.]+(?:e[+-]?\d+)?)"
    r",\s*(-?[\d.]+(?:e[+-]?\d+)?)"
    r",\s*(-?[\d.]+(?:e[+-]?\d+)?)\)"
)

# Matches simple arithmetic expressions in JSON numeric positions, e.g. -0.09 + 0.05
# LLMs sometimes emit these instead of pre-computed values.
_ARITHMETIC_RE = re.compile(
    r"(-?[\d.]+(?:e[+-]?\d+)?)\s*([+\-])\s*([\d.]+(?:e[+-]?\d+)?)"
)


def _sanitize_tuples(s: str) -> str:
    """Replace Python tuple coordinate syntax ``(x, y, z)`` with JSON arrays ``[x, y, z]``."""
    return _TUPLE_RE.sub(r"[\1, \2, \3]", s)


def _sanitize_arithmetic(s: str) -> str:
    """Evaluate simple ``a + b`` / ``a - b`` expressions so JSON can parse numeric fields."""

    def _eval(m: re.Match) -> str:
        a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
        result = a + b if op == "+" else a - b
        return repr(result)

    return _ARITHMETIC_RE.sub(_eval, s)


def extract_json(content: str) -> Optional[Dict]:
    """
    Extract a JSON object from an LLM response string.

    Stages (in order):
    1. Direct ``json.loads`` — succeeds when the model emits clean JSON.
    2. Markdown code-block extraction (```json ... ``` or ``` ... ```) with
       JS ``//`` comment stripping on the extracted block. Also includes
       Stage 2b: JSONL fallback (newline-delimited JSON objects within the block).
    3. Bare ``{...}`` regex match with JS comment stripping for single objects
       embedded in prose.
    3b. Bare JSONL — multiple ``{...}`` objects on separate lines without
       markdown fences, wrapped as ``{"commands": [...]}``.

    Returns ``None`` and logs an error if all stages fail.
    """
    # Pre-process: replace Python tuple syntax and inline arithmetic before any JSON parse.
    content = _sanitize_tuples(content)
    content = _sanitize_arithmetic(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed: {e}")

    # Also try unclosed markdown block (truncated LLM response has no closing ```)
    _open_fence = re.search(r"```(?:json)?\s*\n?(.*)", content, re.DOTALL)
    _closed_fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    json_match = _closed_fence or _open_fence
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug(
                f"Markdown JSON parse failed: {e}. "
                f"Content length: {len(json_str)}, preview: {json_str[:200]}"
            )
        # Retry after stripping JS-style // comments (LLMs often emit these)
        json_str_clean = re.sub(r"//[^\n]*", "", json_str)
        try:
            return json.loads(json_str_clean)
        except json.JSONDecodeError:
            pass
        # Stage 2b-pre: wrap entire block in array brackets — handles multi-object
        # responses with or without commas between objects.
        # LLMs often emit JSONL-style (no commas): {...}\n{...}\n{...}
        for _candidate in (json_str, re.sub(r"\}\s*\{", "},{", json_str)):
            try:
                _result = json.loads("[" + _candidate + "]")
                if isinstance(_result, list) and all(
                    isinstance(x, dict) for x in _result
                ):
                    return {"commands": _result}
            except json.JSONDecodeError:
                pass
        # Stage 2b: JSONL fallback — model emitted newline-delimited JSON objects
        # Wrap into {"commands": [...]} so downstream parser can handle it
        lines = [l.strip() for l in json_str.splitlines() if l.strip().startswith("{")]
        if len(lines) > 1:
            try:
                commands = [json.loads(l.rstrip(",")) for l in lines]
                return {"commands": commands}
            except json.JSONDecodeError:
                pass

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

    # Stage 3b: bare JSONL — model emitted multiple {…} objects separated by
    # commas/newlines without markdown fences.  Wrap into {"commands": [...]}.
    # NOTE: Stage 3b only fires if Stage 3's non-greedy regex failed to parse
    # a complete object (e.g., nested params dict causes {.*?} to stop early).
    # If the first bare object is flat (no nested dicts), Stage 3 parses it
    # and returns it alone — Stage 3b will not run.  This is accepted behaviour
    # for the project's use case where operations always have a "params" dict.

    # Stage 3b-pre: array-wrap attempt for bare multi-line multi-object content.
    # Finds the first { and last } in content and wraps in [...].
    # Also tries comma-insertion to handle JSONL-style output without commas.
    _first_brace = content.find("{")
    _last_brace = content.rfind("}")
    if _first_brace != -1 and _last_brace > _first_brace:
        _candidate = content[_first_brace : _last_brace + 1]
        for _c in (_candidate, re.sub(r"\}\s*\{", "},{", _candidate)):
            try:
                _result = json.loads("[" + _c + "]")
                if isinstance(_result, list) and all(
                    isinstance(x, dict) for x in _result
                ):
                    return {"commands": _result}
            except json.JSONDecodeError:
                pass

    bare_lines = [l.strip() for l in content.splitlines() if l.strip().startswith("{")]
    if len(bare_lines) > 1:
        try:
            commands = [json.loads(l.rstrip(",")) for l in bare_lines]
            return {"commands": commands}
        except json.JSONDecodeError:
            pass

    logger.error(
        f"All JSON extraction methods failed. "
        f"Response length: {len(content)}, preview: {content[:500]}"
    )
    return None
