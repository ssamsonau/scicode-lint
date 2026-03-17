"""Tests for the preprocessing module."""

from scicode_lint.preprocessing.comments import strip_comments


class TestStripComments:
    """Tests for strip_comments function."""

    def test_simple_comment(self) -> None:
        """Strip a simple end-of-line comment."""
        code = "x = 1  # this is a comment"
        result = strip_comments(code)
        assert "# this is a comment" not in result
        assert "x = 1" in result

    def test_preserves_line_numbers(self) -> None:
        """Comments are replaced with spaces, preserving line structure."""
        code = "x = 1  # comment\ny = 2"
        result = strip_comments(code)
        lines = result.splitlines()
        assert len(lines) == 2
        assert lines[1].strip() == "y = 2"

    def test_full_line_comment(self) -> None:
        """Strip a full-line comment."""
        code = "# full line comment\nx = 1"
        result = strip_comments(code)
        lines = result.splitlines()
        assert len(lines) == 2
        assert lines[0].strip() == ""
        assert lines[1].strip() == "x = 1"

    def test_hash_in_string(self) -> None:
        """Hash inside string literal is preserved."""
        code = 'x = "# not a comment"'
        result = strip_comments(code)
        assert '"# not a comment"' in result

    def test_hash_in_fstring(self) -> None:
        """Hash inside f-string is preserved."""
        code = 'x = f"value: {y}  # not a comment"'
        result = strip_comments(code)
        assert '"value: {y}  # not a comment"' in result

    def test_docstring_preserved(self) -> None:
        """Docstrings are not stripped."""
        code = '"""This is a docstring."""\nx = 1'
        result = strip_comments(code)
        assert '"""This is a docstring."""' in result

    def test_multiline_docstring_preserved(self) -> None:
        """Multi-line docstrings are preserved."""
        code = '''def foo():
    """
    Multi-line docstring.
    Second line.
    """
    pass'''
        result = strip_comments(code)
        assert "Multi-line docstring." in result
        assert "Second line." in result

    def test_multiple_comments(self) -> None:
        """Strip multiple comments from code."""
        code = """# header comment
x = 1  # inline comment
# another comment
y = 2"""
        result = strip_comments(code)
        assert "# header" not in result
        assert "# inline" not in result
        assert "# another" not in result
        assert "x = 1" in result
        assert "y = 2" in result

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        assert strip_comments("") == ""

    def test_syntax_error_returns_original(self) -> None:
        """Code with syntax error returns original unchanged."""
        code = "def foo(\n  # unterminated"
        result = strip_comments(code)
        assert result == code

    def test_shebang_stripped(self) -> None:
        """Shebang line (starting with #!) is stripped like any comment."""
        code = "#!/usr/bin/env python\nx = 1"
        result = strip_comments(code)
        assert "#!/usr/bin/env python" not in result
        assert "x = 1" in result

    def test_mixed_quotes_in_strings(self) -> None:
        """Handles mixed quote styles correctly."""
        code = """x = "# not comment"
y = '# also not comment'  # real comment"""
        result = strip_comments(code)
        assert '"# not comment"' in result
        assert "'# also not comment'" in result
        assert "# real comment" not in result

    def test_comment_after_multiline_string(self) -> None:
        """Comment after a multi-line string is stripped."""
        code = '''x = """
multi
line
"""  # comment here'''
        result = strip_comments(code)
        assert "# comment here" not in result
        assert "multi" in result

    def test_type_comments_stripped(self) -> None:
        """Type comments are stripped like regular comments."""
        code = "x = []  # type: List[int]"
        result = strip_comments(code)
        assert "# type:" not in result
        assert "x = []" in result

    def test_preserves_indentation(self) -> None:
        """Indentation before and after comments is preserved."""
        code = """def foo():
    x = 1  # comment
    y = 2"""
        result = strip_comments(code)
        lines = result.splitlines()
        assert lines[1].startswith("    x = 1")
        assert lines[2].startswith("    y = 2")

    def test_no_comments(self) -> None:
        """Code without comments is unchanged."""
        code = '''def foo():
    """Docstring."""
    return 42'''
        result = strip_comments(code)
        assert result == code
