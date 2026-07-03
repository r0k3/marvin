"""Unit tests for the TOON encoder behind the AXI CLI."""

from __future__ import annotations

from marvin.toon import encode_error, encode_help, encode_kv, encode_table, encode_value


class TestEncodeValue:
    def test_plain_string_unquoted(self):
        assert encode_value("hello world") == "hello world"

    def test_comma_forces_quotes(self):
        assert encode_value("a, b") == '"a, b"'

    def test_embedded_quote_doubled(self):
        assert encode_value('say "hi"') == '"say ""hi"""'

    def test_newline_flattened(self):
        assert encode_value("line1\nline2") == "line1 line2"

    def test_none_is_empty(self):
        assert encode_value(None) == ""

    def test_bool_lowercase(self):
        assert encode_value(True) == "true"
        assert encode_value(False) == "false"

    def test_float_compact(self):
        assert encode_value(0.752341) == "0.7523"

    def test_list_semicolon_joined(self):
        assert encode_value(["a", "b"]) == "a;b"


class TestEncodeTable:
    def test_header_declares_count_and_schema(self):
        out = encode_table(
            "hits",
            [{"title": "T1", "kind": "semantic"}, {"title": "T2", "kind": "episodic"}],
            ["title", "kind"],
        )
        lines = out.splitlines()
        assert lines[0] == "hits[2]{title,kind}:"
        assert lines[1] == "  T1,semantic"
        assert lines[2] == "  T2,episodic"

    def test_empty_is_definitive(self):
        assert encode_table("hits", [], ["title"]) == "hits[0]:"

    def test_empty_with_reason(self):
        out = encode_table("hits", [], ["title"], empty='no matches for "x"')
        assert out == 'hits[0]: (no matches for "x")'

    def test_missing_field_renders_empty(self):
        out = encode_table("rows", [{"a": 1}], ["a", "b"])
        assert out.splitlines()[1] == "  1,"


class TestBlocks:
    def test_kv_block(self):
        out = encode_kv("vault", {"path": "/tmp/v", "notes": 4})
        assert out.splitlines() == ["vault:", "  path: /tmp/v", "  notes: 4"]

    def test_help_block_aligned_with_placeholders(self):
        out = encode_help([("marvin read <path>", "open a note"), ("marvin sync", "reindex")])
        lines = out.splitlines()
        assert lines[0] == "help[2]:"
        assert "<path>" in lines[1] and "# open a note" in lines[1]

    def test_help_empty(self):
        assert encode_help([]) == ""

    def test_error_block(self):
        out = encode_error("runtime", "vault not found: /x")
        assert out.splitlines()[0] == "error[1]{code,message}:"
        assert "vault not found: /x" in out
