"""Tests for context length validation."""

import pytest

from scicode_lint.llm.exceptions import ContextLengthError
from scicode_lint.llm.tokens import (
    check_context_length,
    estimate_prompt_tokens,
    estimate_tokens,
)


def test_estimate_tokens_basic() -> None:
    """Test basic token estimation."""
    # Empty string
    assert estimate_tokens("") == 0

    # Simple text (4 chars = 1 token)
    assert estimate_tokens("test") == 1
    assert estimate_tokens("hello world") == 2

    # Code snippet (~100 chars = ~25 tokens)
    code = "def hello():\n    print('world')\n    return True"
    tokens = estimate_tokens(code)
    assert 10 < tokens < 20  # Should be ~12 tokens


def test_estimate_prompt_tokens() -> None:
    """Test full prompt token estimation with overhead."""
    code = "x = 1"
    system = "You are a code analyzer"
    user = f"Code:\n{code}\n\nQuestion: Is this good?"

    tokens = estimate_prompt_tokens(code, system, user)

    # Should include:
    # - System prompt (~6 tokens)
    # - User prompt (~8 tokens)
    # - Structured output overhead (200 tokens)
    # - Safety margin (10%)
    # Total: ~236 tokens
    assert 200 < tokens < 300


def test_check_context_length_fits() -> None:
    """Test context length check when input fits."""
    code = "def hello(): pass"
    system = "You are a code analyzer"
    user = f"Code:\n{code}"

    # Should fit in 8000 tokens
    fits, estimated = check_context_length(
        code=code,
        system_prompt=system,
        user_prompt=user,
        max_tokens=8000,
        file_path="test.py",
    )

    assert fits is True
    assert estimated < 8000


def test_check_context_length_exceeds() -> None:
    """Test context length check when input exceeds limit."""
    # Create large code (10000 chars = ~2500 tokens)
    # Plus overhead = ~2700 tokens total
    code = "x = 1\n" * 2000
    system = "You are a code analyzer"
    user = f"Code:\n{code}"

    # Should exceed 1000 token limit
    with pytest.raises(ContextLengthError) as exc_info:
        check_context_length(
            code=code,
            system_prompt=system,
            user_prompt=user,
            max_tokens=1000,
            file_path="large_file.py",
        )

    error = exc_info.value
    assert error.file_path == "large_file.py"
    assert error.estimated_tokens > 1000
    assert error.max_tokens == 1000
    assert "large_file.py" in error.message
    assert "Suggestions:" in error.message


def test_context_length_error_message() -> None:
    """Test error message formatting."""
    error = ContextLengthError(
        file_path="test.py",
        estimated_tokens=10000,
        max_tokens=8000,
    )

    message = str(error)
    assert "test.py" in message
    assert "10,000" in message
    assert "8,000" in message
    assert "2,000" in message  # overflow
    assert "Suggestions:" in message
    assert "Split into smaller files" in message
