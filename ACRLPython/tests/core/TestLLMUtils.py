#!/usr/bin/env python3
"""
Tests for core.LLMUtils.extract_json

Covers all extraction stages:
  - Clean JSON (stage 1: direct parse)
  - Markdown-fenced JSON (stage 2)
  - Markdown + JS // comments (stage 2 with comment stripping)
  - Bare JSON embedded in prose (stage 3)
  - Bare JSON + JS // comments (stage 3 with comment stripping)
  - Unparseable response → None
  - Empty string → None
"""

import pytest

from core.LLMUtils import extract_json


class TestExtractJsonDirect:
    """Stage 1: direct json.loads succeeds."""

    def test_clean_json_object(self):
        result = extract_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_nested_json(self):
        payload = '{"commands": [{"op": "move", "robot_id": "Robot1"}]}'
        result = extract_json(payload)
        assert result["commands"][0]["op"] == "move"

    def test_json_with_whitespace(self):
        result = extract_json('  \n{ "a": 1 }\n  ')
        assert result == {"a": 1}


class TestExtractJsonMarkdown:
    """Stage 2: markdown code block extraction."""

    def test_json_fenced_block(self):
        content = '```json\n{"violates": false, "reason": "safe"}\n```'
        result = extract_json(content)
        assert result == {"violates": False, "reason": "safe"}

    def test_generic_fenced_block(self):
        content = '```\n{"a": 1}\n```'
        result = extract_json(content)
        assert result == {"a": 1}

    def test_markdown_with_js_comments(self):
        content = (
            "```json\n"
            "{\n"
            '  "accept": true, // the robot accepts\n'
            '  "concerns": []\n'
            "}\n"
            "```"
        )
        result = extract_json(content)
        assert result is not None
        assert result["accept"] is True
        assert result["concerns"] == []

    def test_markdown_preamble_then_block(self):
        content = "Here is the result:\n```json\n{\"ok\": true}\n```\nDone."
        result = extract_json(content)
        assert result == {"ok": True}


class TestExtractJsonBareRegex:
    """Stage 3: bare {…} regex in surrounding prose."""

    def test_json_embedded_in_prose(self):
        content = 'The robot says {"status": "ready"} and waits.'
        result = extract_json(content)
        assert result == {"status": "ready"}

    def test_bare_json_with_js_comments(self):
        content = (
            'Output: {"violates": false // no rule violated\n} done.'
        )
        result = extract_json(content)
        assert result is not None
        assert result["violates"] is False


class TestExtractJsonFailures:
    """Cases that should return None."""

    def test_empty_string(self):
        result = extract_json("")
        assert result is None

    def test_plain_text_no_json(self):
        result = extract_json("The robot moved to position A.")
        assert result is None

    def test_malformed_json_everywhere(self):
        result = extract_json("{invalid json content here}")
        assert result is None

    def test_think_tags_without_json(self):
        result = extract_json("[THINK]Reasoning goes here[/THINK]")
        assert result is None
