# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for the shared frontmatter helpers (recall + conflicts read through these)."""

from __future__ import annotations

from wikimoth.frontmatter import split_frontmatter, unquote_scalar


def test_split_frontmatter_basic():
    block, body = split_frontmatter("---\nname: x\n---\nhello\nworld\n")
    assert "name: x" in block
    assert body == "hello\nworld\n"


def test_split_frontmatter_none():
    assert split_frontmatter("no fm here") == ("", "no fm here")


def test_split_frontmatter_unterminated():
    block, body = split_frontmatter("---\nname: x\nno close")
    assert "name: x" in block
    assert body == ""


def test_split_frontmatter_crlf():
    block, body = split_frontmatter("---\r\nname: x\r\n---\r\nbody\r\n")
    assert "name: x" in block
    assert "body" in body


def test_unquote_plain():
    assert unquote_scalar("hello") == "hello"


def test_unquote_quoted():
    assert unquote_scalar('"hello world"') == "hello world"


def test_unquote_quoted_with_escape():
    assert unquote_scalar('"a \\"b\\" c"') == 'a "b" c'


def test_unquote_quoted_with_trailing_comment():
    assert unquote_scalar('"Brain Forge" # legacy') == "Brain Forge"


def test_unquote_unquoted_trailing_comment():
    assert unquote_scalar("active # current") == "active"


def test_unquote_keeps_hash_token():
    assert unquote_scalar("C#") == "C#"


def test_roundtrip_yaml_str_unquote_is_identity():
    # _yaml_str (the writer) -> unquote_scalar (the reader) must be identity for
    # arbitrary values, incl. Windows paths ending in a backslash (regression).
    from wikimoth.capture.note import _yaml_str

    for s in [
        "plain",
        "with spaces",
        'has "quotes"',
        "C:\\path\\",
        "D:\\dev\\wikimoth\\",
        "ends with backslash\\",
        "a # b",
        "C#",
        'mix "q" and \\ back\\',
        "",
    ]:
        assert unquote_scalar(_yaml_str(s)) == s, s
