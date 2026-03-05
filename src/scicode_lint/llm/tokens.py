"""Token estimation utilities for context length validation."""


def estimate_tokens(text: str) -> int:
    """Estimate token count for text using heuristic.

    Uses rough approximation: ~4 characters per token.
    This is conservative for code (actual ratio is often 3-3.5 for Python).

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count

    Example:
        >>> code = "def hello():\\n    print('world')"
        >>> tokens = estimate_tokens(code)
        >>> tokens
        9
    """
    # Conservative estimate: 4 chars per token
    # Real tokenizers vary but this is safe for pre-flight checks
    return len(text) // 4


def estimate_prompt_tokens(
    code: str,
    system_prompt: str,
    user_prompt_template: str,
) -> int:
    """Estimate total token count for a complete prompt.

    Accounts for:
    - System prompt
    - User prompt (including code)
    - Structured output overhead (~200 tokens for JSON schema)
    - Output token buffer (~200 tokens for response)
    - Safety margin (10% buffer)

    Args:
        code: Source code being analyzed
        system_prompt: System message text
        user_prompt_template: User prompt template (includes code)

    Returns:
        Estimated total token count including overhead

    Example:
        >>> code = "def foo(): pass"
        >>> system = "You are a code analyzer"
        >>> user = "Code:\\n" + code + "\\nQuestion: Is this good?"
        >>> tokens = estimate_prompt_tokens(code, system, user)
    """
    # Estimate base tokens
    system_tokens = estimate_tokens(system_prompt)
    user_tokens = estimate_tokens(user_prompt_template)

    # Add structured output overhead
    # JSON schema adds ~100-300 tokens depending on complexity
    # DetectionResult schema is relatively simple
    structured_output_overhead = 200

    # Add safety margin (10%)
    # Note: Output token buffer is handled separately in check_context_length()
    total = system_tokens + user_tokens + structured_output_overhead
    safety_margin = int(total * 0.1)

    return total + safety_margin


def check_context_length(
    code: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    file_path: str = "unknown",
    output_buffer: int = 400,
) -> tuple[bool, int]:
    """Check if prompt fits within context length.

    Reserves space for output tokens by validating against max_tokens - output_buffer.
    Uses conservative buffer to account for tokenizer estimation errors.

    Args:
        code: Source code being analyzed
        system_prompt: System message
        user_prompt: User prompt (includes code)
        max_tokens: Maximum context length (total input + output)
        file_path: File path for error messages
        output_buffer: Tokens to reserve for model output and safety margin (default: 400)

    Returns:
        Tuple of (fits_in_context, estimated_tokens)

    Raises:
        ContextLengthError: If input exceeds max_tokens - output_buffer

    Example:
        >>> code = "def hello(): pass"
        >>> fits, tokens = check_context_length(
        ...     code, "system", "user: " + code, 8000, "test.py"
        ... )
        >>> fits
        True
    """
    from scicode_lint.llm.exceptions import ContextLengthError

    estimated = estimate_prompt_tokens(code, system_prompt, user_prompt)

    # Reserve space for output by checking against reduced limit
    max_input_tokens = max_tokens - output_buffer

    if estimated > max_input_tokens:
        raise ContextLengthError(
            file_path=file_path,
            estimated_tokens=estimated,
            max_tokens=max_input_tokens,
        )

    return True, estimated
