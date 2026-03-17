"""Comment stripping for Python code.

Strips # comments from Python source code while preserving:
- Line numbers (comments replaced with whitespace)
- String literals containing # characters
- Docstrings (triple-quoted strings)

This preprocessing step ensures:
1. Detection based on code structure, not stated intent
2. No intention leakage in test file evaluation
3. Misleading real-world comments don't confuse the model
4. Reduced token usage (comments stripped from context)
"""

import io
import tokenize

from loguru import logger


def strip_comments(code: str) -> str:
    """Strip # comments from Python code, preserving line numbers and docstrings.

    Uses Python's tokenize module to correctly identify comments
    (handles # inside strings, f-strings, etc.)

    Args:
        code: Python source code as a string.

    Returns:
        Code with # comments replaced by whitespace (preserves line numbers).
        Returns original code unchanged if tokenization fails (syntax error).

    Example:
        >>> code = 'x = 1  # comment\\ny = "# not a comment"'
        >>> result = strip_comments(code)
        >>> '# comment' not in result
        True
        >>> '"# not a comment"' in result
        True
    """
    if not code:
        return code

    try:
        lines = code.splitlines(keepends=True)
        comments: list[tuple[int, int, int]] = []

        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                comments.append((tok.start[0], tok.start[1], tok.end[1]))

        for line_no, start_col, end_col in reversed(comments):
            line_idx = line_no - 1
            if line_idx < len(lines):
                line = lines[line_idx]
                comment_len = end_col - start_col
                lines[line_idx] = line[:start_col] + " " * comment_len + line[end_col:]

        return "".join(lines)
    except tokenize.TokenError as e:
        logger.warning(f"Comment stripping skipped (syntax error in analyzed file): {e}")
        return code
