#!/usr/bin/env python3
"""Comprehensive pattern validation - deterministic quality checks.

Usage:
    python scripts/validate_pattern_tests.py           # Check all patterns
    python scripts/validate_pattern_tests.py --fix     # Auto-fix what's possible
    python scripts/validate_pattern_tests.py ml-002    # Check specific pattern
    python scripts/validate_pattern_tests.py --strict  # Fail on warnings too
    python scripts/validate_pattern_tests.py --fetch-refs   # Fetch and cache reference docs
    python scripts/validate_pattern_tests.py --clean-cache  # Remove orphaned cache files

Checks performed:
1. TOML/file sync - every test file has TOML entry and vice versa
2. Schema validation - pattern.toml matches Pydantic model
3. Data leakage - no BUG/CORRECT/WRONG hints in test files
4. Test file count - minimum 3 positive, 3 negative (warning)
5. TODO markers - no unfinished placeholders in TOML
6. Detection question format - ends with YES/NO conditions
7. Test file syntax - all .py files are valid Python
8. Empty fields - required fields have content
9. Category mismatch - meta.category matches directory location
10. Snippet verification - expected_location snippets exist in files (warning)
11. Related patterns - related_patterns references exist
12. Test file diversity - detect copy-paste (AST similarity)
13. Reference URLs - fetch and cache reference documentation (--fetch-refs)
"""

import argparse
import ast
import asyncio
import hashlib
import re
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import tomllib
from pydantic import BaseModel

# Cache settings for reference docs
DOC_CACHE_DIR = Path(__file__).parent / "doc_cache"
DOC_CACHE_RAW_DIR = DOC_CACHE_DIR / "raw"
DOC_CACHE_CLEAN_DIR = DOC_CACHE_DIR / "clean"
DOC_CACHE_MAX_AGE_DAYS = 7

# vLLM context: 16K tokens input ~ 64KB text. With ~100 chars/line, use 500 lines/chunk
VLLM_CHUNK_SIZE = 500
VLLM_CHUNK_OVERLAP = 100
MAX_REFERENCE_URLS = 5
MAX_DOC_LINES = 1000  # Warn if cached doc exceeds this (find more specific page)


# Pydantic model for vLLM doc extraction response
class CutRange(BaseModel):
    """Line range to cut from documentation."""

    start: int
    end: int


class DocCutResponse(BaseModel):
    """Response indicating which line ranges to cut."""

    cut: list[list[int]]  # List of [start, end] pairs


# Data leakage patterns to detect in test files
DATA_LEAKAGE_PATTERNS = [
    (r"#\s*BUG:", "BUG marker comment"),
    (r"#\s*CORRECT:", "CORRECT marker comment"),
    (r"#\s*WRONG", "WRONG marker comment"),
    (r"#\s*FIXME:", "FIXME marker comment"),
    (r"#\s*This is (wrong|incorrect|buggy|the bug)", "Hint comment"),
    (r"#\s*This (leaks|causes|creates)", "Hint comment explaining issue"),
    (r'""".*?(bug|issue|problem|incorrect|wrong).*?"""', "Docstring with hint"),
]

# Required YES/NO pattern at end of detection question
YES_NO_PATTERN = re.compile(r"YES\s*=.*\n\s*NO\s*=", re.IGNORECASE | re.MULTILINE)


@dataclass
class ValidationIssue:
    """A single validation issue."""

    level: str  # "error" or "warning"
    check: str  # which check found this
    message: str
    file: str = ""


@dataclass
class ValidationResult:
    """Result of validating a pattern."""

    pattern_id: str
    category: str
    issues: list[ValidationIssue] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.level == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.level == "warning" for i in self.issues)


def get_test_files_on_disk(pattern_dir: Path) -> dict[str, set[str]]:
    """Get all test files on disk for a pattern."""
    result: dict[str, set[str]] = {"positive": set(), "negative": set(), "context_dependent": set()}

    for test_type in result.keys():
        test_dir = pattern_dir / f"test_{test_type}"
        if test_dir.exists():
            result[test_type] = {
                f.name for f in test_dir.glob("*.py") if not f.name.startswith("_")
            }

    return result


def get_test_files_in_toml(toml_data: dict[str, Any]) -> dict[str, set[str]]:
    """Get all test files referenced in pattern.toml."""
    result: dict[str, set[str]] = {"positive": set(), "negative": set(), "context_dependent": set()}

    tests = toml_data.get("tests", {})
    for test_type in result.keys():
        for entry in tests.get(test_type, []):
            file_path = entry.get("file", "")
            result[test_type].add(Path(file_path).name)

    return result


# =============================================================================
# CHECK 1: TOML/File Sync
# =============================================================================


def check_toml_file_sync(
    pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult
) -> dict[str, dict[str, list[str]]]:
    """Check that test files on disk match TOML entries."""
    on_disk = get_test_files_on_disk(pattern_dir)
    in_toml = get_test_files_in_toml(toml_data)

    issues: dict[str, dict[str, list[str]]] = {"missing_in_toml": {}, "missing_on_disk": {}}

    for test_type in ["positive", "negative", "context_dependent"]:
        missing_in_toml = on_disk[test_type] - in_toml[test_type]
        missing_on_disk = in_toml[test_type] - on_disk[test_type]

        if missing_in_toml:
            issues["missing_in_toml"][test_type] = sorted(missing_in_toml)
            for f in missing_in_toml:
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "toml_sync",
                        f"File on disk not in TOML: test_{test_type}/{f}",
                    )
                )

        if missing_on_disk:
            issues["missing_on_disk"][test_type] = sorted(missing_on_disk)
            for f in missing_on_disk:
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "toml_sync",
                        f"TOML entry but file missing: test_{test_type}/{f}",
                    )
                )

    return issues


# =============================================================================
# CHECK 2: Schema Validation
# =============================================================================


def check_schema(pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult) -> bool:
    """Validate pattern.toml against Pydantic schema."""
    try:
        from scicode_lint.detectors.pattern_models import PatternTOML

        PatternTOML(**toml_data)
        return True
    except Exception as e:
        result.issues.append(ValidationIssue("error", "schema", f"Schema validation failed: {e}"))
        return False


# =============================================================================
# CHECK 3: Data Leakage Detection
# =============================================================================


def check_data_leakage(pattern_dir: Path, result: ValidationResult) -> None:
    """Check test files for hint comments that leak answers."""
    for test_type in ["positive", "negative", "context_dependent"]:
        test_dir = pattern_dir / f"test_{test_type}"
        if not test_dir.exists():
            continue

        for py_file in test_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                content = py_file.read_text()
            except Exception:
                continue

            for pattern, desc in DATA_LEAKAGE_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    result.issues.append(
                        ValidationIssue(
                            "error",
                            "data_leakage",
                            f"{desc} found",
                            file=f"test_{test_type}/{py_file.name}",
                        )
                    )


# =============================================================================
# CHECK 4: Test File Count
# =============================================================================


def check_test_count(pattern_dir: Path, result: ValidationResult) -> None:
    """Check minimum test file counts (3 positive, 3 negative recommended)."""
    on_disk = get_test_files_on_disk(pattern_dir)

    pos_count = len(on_disk["positive"])
    neg_count = len(on_disk["negative"])

    if pos_count < 3:
        result.issues.append(
            ValidationIssue(
                "warning",
                "test_count",
                f"Only {pos_count} positive tests (recommend 3+)",
            )
        )

    if neg_count < 3:
        result.issues.append(
            ValidationIssue(
                "warning",
                "test_count",
                f"Only {neg_count} negative tests (recommend 3+)",
            )
        )


# =============================================================================
# CHECK 5: TODO Markers
# =============================================================================


def check_todo_markers(toml_path: Path, result: ValidationResult) -> None:
    """Check for unfinished TODO placeholders in TOML."""
    content = toml_path.read_text()

    # Find TODO patterns
    todo_matches = re.findall(r"TODO[:\s].*", content, re.IGNORECASE)
    for match in todo_matches:
        result.issues.append(
            ValidationIssue("error", "todo_marker", f"Unfinished: {match.strip()}")
        )


# =============================================================================
# CHECK 6: Detection Question Format
# =============================================================================


def check_detection_question(toml_data: dict[str, Any], result: ValidationResult) -> None:
    """Check detection question ends with YES/NO conditions."""
    detection = toml_data.get("detection", {})
    question = detection.get("question", "")

    if not question:
        result.issues.append(
            ValidationIssue("error", "detection_format", "Missing detection question")
        )
        return

    if not YES_NO_PATTERN.search(question):
        result.issues.append(
            ValidationIssue(
                "warning",
                "detection_format",
                "Detection question should end with 'YES = ...' and 'NO = ...' conditions",
            )
        )


# =============================================================================
# CHECK 7: Test File Syntax
# =============================================================================


def check_test_syntax(pattern_dir: Path, result: ValidationResult) -> None:
    """Check all test files are valid Python."""
    for test_type in ["positive", "negative", "context_dependent"]:
        test_dir = pattern_dir / f"test_{test_type}"
        if not test_dir.exists():
            continue

        for py_file in test_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                source = py_file.read_text()
                ast.parse(source)
            except SyntaxError as e:
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "syntax",
                        f"Invalid Python syntax: {e}",
                        file=f"test_{test_type}/{py_file.name}",
                    )
                )


# =============================================================================
# CHECK 8: Empty Fields
# =============================================================================


def check_empty_fields(toml_data: dict[str, Any], result: ValidationResult) -> None:
    """Check required fields have content."""
    # Check meta fields
    meta = toml_data.get("meta", {})
    for field_name in ["description", "explanation"]:
        value = meta.get(field_name, "")
        if not value or not value.strip():
            result.issues.append(
                ValidationIssue("error", "empty_field", f"meta.{field_name} is empty")
            )

    # Check detection fields
    detection = toml_data.get("detection", {})
    for field_name in ["question", "warning_message"]:
        value = detection.get(field_name, "")
        if not value or not value.strip():
            result.issues.append(
                ValidationIssue("error", "empty_field", f"detection.{field_name} is empty")
            )

    # Check test entries
    tests = toml_data.get("tests", {})
    for test_type in ["positive", "negative", "context_dependent"]:
        for i, entry in enumerate(tests.get(test_type, [])):
            if not entry.get("description", "").strip():
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "empty_field",
                        f"tests.{test_type}[{i}].description is empty",
                    )
                )
            if test_type == "positive" and not entry.get("expected_issue", "").strip():
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "empty_field",
                        f"tests.{test_type}[{i}].expected_issue is empty",
                    )
                )


# =============================================================================
# CHECK 9: Category Mismatch
# =============================================================================


def check_category_mismatch(
    pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult
) -> None:
    """Check that meta.category matches the pattern's directory location."""
    meta = toml_data.get("meta", {})
    toml_category = meta.get("category", "")
    dir_category = pattern_dir.parent.name

    if toml_category and toml_category != dir_category:
        result.issues.append(
            ValidationIssue(
                "error",
                "category_mismatch",
                f"meta.category='{toml_category}' but pattern is in '{dir_category}/' directory",
            )
        )


# =============================================================================
# CHECK 10: Expected Location Snippet Verification
# =============================================================================


def check_expected_location_snippets(
    pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult
) -> None:
    """Check that expected_location snippets exist in the actual test files."""
    tests = toml_data.get("tests", {})

    for entry in tests.get("positive", []):
        expected_loc = entry.get("expected_location", {})
        snippet = expected_loc.get("snippet", "")
        file_path_str = entry.get("file", "")

        if not snippet or not file_path_str:
            continue

        file_path = pattern_dir / file_path_str
        if not file_path.exists():
            continue  # Already caught by toml_sync check

        try:
            content = file_path.read_text()
            # Normalize whitespace for comparison
            normalized_content = " ".join(content.split())
            normalized_snippet = " ".join(snippet.split())

            if normalized_snippet not in normalized_content:
                result.issues.append(
                    ValidationIssue(
                        "warning",
                        "snippet_mismatch",
                        f"Snippet not found in file: '{snippet[:50]}...'",
                        file=file_path_str,
                    )
                )
        except Exception:
            pass


# =============================================================================
# CHECK 11: Related Patterns Exist
# =============================================================================


def check_related_patterns_exist(
    pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult
) -> None:
    """Check that related_patterns references exist."""
    meta = toml_data.get("meta", {})
    related = meta.get("related_patterns", [])
    patterns_root = pattern_dir.parent.parent

    for related_id in related:
        # Search for the pattern in all categories
        # Patterns can be referenced by:
        # - Full name: "pt-007-inference-without-eval"
        # - ID prefix: "pt-007" (matches "pt-007-*")
        found = False
        for category_dir in patterns_root.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith((".", "_")):
                continue
            for pdir in category_dir.iterdir():
                # Match full name or ID prefix (e.g., "pt-007" matches "pt-007-...")
                if pdir.name == related_id or pdir.name.startswith(f"{related_id}-"):
                    found = True
                    break
            if found:
                break

        if not found:
            result.issues.append(
                ValidationIssue(
                    "error",
                    "related_pattern",
                    f"Related pattern '{related_id}' does not exist",
                )
            )


# =============================================================================
# CHECK 12: Test File Diversity (AST similarity)
# =============================================================================


def get_ast_hash(file_path: Path) -> str | None:
    """Get a normalized AST hash for a Python file."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)

        # Normalize: remove docstrings, normalize names
        for node in ast.walk(tree):
            # Remove docstrings
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body = node.body[1:]

        # Convert to string and hash
        ast_str = ast.dump(tree, annotate_fields=False)
        return hashlib.md5(ast_str.encode()).hexdigest()[:8]
    except Exception:
        return None


def check_test_diversity(pattern_dir: Path, result: ValidationResult) -> None:
    """Check test files aren't too similar (copy-paste detection)."""
    for test_type in ["positive", "negative"]:
        test_dir = pattern_dir / f"test_{test_type}"
        if not test_dir.exists():
            continue

        files = [f for f in test_dir.glob("*.py") if not f.name.startswith("_")]
        if len(files) < 2:
            continue

        # Get AST hashes
        hashes: dict[str, list[str]] = {}
        for f in files:
            h = get_ast_hash(f)
            if h:
                hashes.setdefault(h, []).append(f.name)

        # Check for duplicates
        for h, filenames in hashes.items():
            if len(filenames) > 1:
                result.issues.append(
                    ValidationIssue(
                        "warning",
                        "diversity",
                        f"Nearly identical files in test_{test_type}: {', '.join(filenames)}",
                    )
                )


# =============================================================================
# CHECK 13: Reference URL Validation and Caching
# =============================================================================


def get_cache_filename(url: str, pattern_id: str = "") -> str:
    """Get cache filename for a URL, optionally prefixed with pattern ID."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    try:
        from urllib.parse import urlparse

        domain = urlparse(url).netloc.replace(".", "_")
        if pattern_id:
            return f"{pattern_id}_{domain}_{url_hash}.md"
        return f"{domain}_{url_hash}.md"
    except Exception:
        if pattern_id:
            return f"{pattern_id}_{url_hash}.md"
        return f"{url_hash}.md"


def get_cache_path(url: str, pattern_id: str = "") -> Path:
    """Get clean cache file path for a URL (markdown format)."""
    return DOC_CACHE_CLEAN_DIR / get_cache_filename(url, pattern_id)


def get_raw_cache_path(url: str, pattern_id: str = "") -> Path:
    """Get raw cache file path for a URL (markdown format)."""
    return DOC_CACHE_RAW_DIR / get_cache_filename(url, pattern_id)


def is_cache_valid(cache_path: Path) -> bool:
    """Check if cache file exists and is not expired."""
    if not cache_path.exists():
        return False
    age_days = (time.time() - cache_path.stat().st_mtime) / (60 * 60 * 24)
    return age_days < DOC_CACHE_MAX_AGE_DAYS


def extract_doc_content_with_vllm(markdown_content: str) -> tuple[str, bool]:
    """Use local vLLM to extract documentation content with async chunked processing.

    Asks vLLM for line ranges to CUT (navigation/boilerplate), not content to keep.
    Uses existing VLLMClient infrastructure with guided_json for structured output.

    Returns:
        Tuple of (filtered_content, success). If vLLM fails, returns (original_content, False).
    """
    from scicode_lint.config import load_llm_config
    from scicode_lint.llm.client import LLMClient, create_client

    lines = markdown_content.split("\n")

    # If content is small, no vLLM needed
    if len(lines) < 50:
        return markdown_content, True

    # Check content length against max_input_tokens (~4 chars per token estimate)
    try:
        llm_config = load_llm_config()
        max_chars = llm_config.max_input_tokens * 4  # ~4 chars per token
        if len(markdown_content) > max_chars:
            # Content too large for vLLM context
            return markdown_content, False
    except Exception:
        return markdown_content, False

    # Build list of chunks with overlap
    chunks: list[tuple[int, int, str]] = []
    for start in range(0, len(lines), VLLM_CHUNK_SIZE - VLLM_CHUNK_OVERLAP):
        end = min(start + VLLM_CHUNK_SIZE, len(lines))
        chunk_lines = lines[start:end]
        # Number lines (line numbers are 1-indexed in the full file)
        numbered = "\n".join(f"{start + i + 1}: {line}" for i, line in enumerate(chunk_lines))
        chunks.append((start, end, numbered))

    async def process_chunk(client: LLMClient, numbered: str) -> tuple[set[int], bool]:
        """Process a single chunk and return (line numbers to CUT, success)."""
        system_prompt = (
            "You are a documentation parser. Identify navigation and boilerplate content to remove."
        )
        user_prompt = f"""CUT: nav menus, sidebars, footers, "skip to content", sign-in, social links, cookies.
KEEP: API docs, code, params, descriptions, warnings, notes.
Return {{"cut": [[start,end], ...]}} for line ranges to remove. Empty if nothing to cut.

{numbered}"""

        try:
            result = await client.async_complete_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=DocCutResponse,
            )
            cut_lines: set[int] = set()
            for range_pair in result.cut:
                if len(range_pair) == 2:
                    cut_lines.update(range(range_pair[0], range_pair[1] + 1))
            return cut_lines, True
        except Exception:
            return set(), False

    async def process_all_chunks(client: LLMClient) -> tuple[set[int], bool]:
        """Process all chunks concurrently. Returns (cut_lines, all_succeeded)."""
        tasks = [process_chunk(client, numbered) for _, _, numbered in chunks]
        results = await asyncio.gather(*tasks)
        cut_lines: set[int] = set()
        all_succeeded = True
        for chunk_cut_lines, success in results:
            cut_lines.update(chunk_cut_lines)
            if not success:
                all_succeeded = False
        return cut_lines, all_succeeded

    # Run async processing using existing VLLMClient (with auto-detection)
    try:
        client = create_client(llm_config)
        cut_lines, vllm_success = asyncio.run(process_all_chunks(client))
    except Exception:
        return markdown_content, False

    if not vllm_success:
        return markdown_content, False

    # Keep lines NOT in cut_lines
    filtered = "\n".join(line for i, line in enumerate(lines, 1) if i not in cut_lines)
    return filtered, True


def strip_html_nav_elements(html_content: str) -> str:
    """Strip navigation elements from HTML before markdown conversion.

    Removes: nav, footer, header (if it looks like site header), aside, and common nav classes.
    Uses Python's built-in html.parser for reliability.
    """
    from html.parser import HTMLParser

    # Tags to remove entirely (including content)
    nav_tags = {"nav", "footer", "aside", "header"}
    # Tags to check for nav-like classes (substring match)
    nav_classes = {
        "navbar",
        "sidebar",
        "site-header",
        "site-footer",
        "breadcrumb",
        "toc",
        "menu",
        "topnav",
        "header-nav",
        "footer-nav",
        "skip-link",
        "mobile-menu",
    }

    class NavStripper(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.result: list[str] = []
            self.skip_depth = 0  # When > 0, skip content
            self.skip_tag_stack: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            attrs_dict = dict(attrs)
            class_attr = attrs_dict.get("class", "") or ""
            id_attr = attrs_dict.get("id", "") or ""

            # Check if this is a nav element to skip
            should_skip = tag in nav_tags
            if not should_skip:
                # Check for nav-like classes/ids
                for nav_class in nav_classes:
                    if nav_class in class_attr.lower() or nav_class in id_attr.lower():
                        should_skip = True
                        break

            if should_skip:
                self.skip_depth += 1
                self.skip_tag_stack.append(tag)
            elif self.skip_depth == 0:
                # Rebuild the tag
                attr_str = " ".join(f'{k}="{v}"' if v else k for k, v in attrs)
                if attr_str:
                    self.result.append(f"<{tag} {attr_str}>")
                else:
                    self.result.append(f"<{tag}>")

        def handle_endtag(self, tag: str) -> None:
            if self.skip_depth > 0 and self.skip_tag_stack and self.skip_tag_stack[-1] == tag:
                self.skip_depth -= 1
                self.skip_tag_stack.pop()
            elif self.skip_depth == 0:
                self.result.append(f"</{tag}>")

        def handle_data(self, data: str) -> None:
            if self.skip_depth == 0:
                self.result.append(data)

        def handle_comment(self, data: str) -> None:
            if self.skip_depth == 0:
                self.result.append(f"<!--{data}-->")

        def handle_decl(self, decl: str) -> None:
            self.result.append(f"<!{decl}>")

        def get_result(self) -> str:
            return "".join(self.result)

    parser = NavStripper()
    try:
        parser.feed(html_content)
        return parser.get_result()
    except Exception:
        # If parsing fails, return original content
        return html_content


def fetch_and_cache_url(url: str, result: ValidationResult, pattern_id: str = "") -> bool:
    """Fetch URL content and cache it as markdown in raw/ and clean/ subdirectories."""
    import html2text
    import httpx

    clean_path = get_cache_path(url, pattern_id)
    raw_path = get_raw_cache_path(url, pattern_id)

    # Check if cache is still valid (prefer clean, fall back to raw)
    if is_cache_valid(clean_path) or is_cache_valid(raw_path):
        return True

    # Ensure cache directories exist
    DOC_CACHE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOC_CACHE_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.Client(
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "scicode-lint/1.0 (pattern-verification)"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            html_content = response.text

        # Setup markdown converter
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0  # Don't wrap lines

        # Convert raw HTML to markdown (for raw cache)
        markdown_content = h.handle(html_content)

        # Check for empty/minimal content (likely redirect or fetch failure)
        content_chars = len(markdown_content.strip())
        if content_chars < 100:
            result.issues.append(
                ValidationIssue(
                    "warning",
                    "reference_url",
                    f"Fetch returned minimal content ({content_chars} chars) for {url} - check for redirects",
                )
            )
            return False

        header = f"# Source: {url}\n# Fetched: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Save raw markdown
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(markdown_content)

        # Step 1: Strip HTML nav elements (deterministic, fast)
        stripped_html = strip_html_nav_elements(html_content)
        clean_markdown = h.handle(stripped_html)

        # Step 2: Pass through vLLM for additional cleaning
        # HTML stripping handles structural nav, vLLM catches remaining boilerplate
        original_len = len(markdown_content)
        html_stripped_len = len(clean_markdown)
        html_reduction = (1 - html_stripped_len / original_len) * 100 if original_len > 0 else 0

        # Pass HTML-stripped content to vLLM for further cleaning
        filtered_content, vllm_success = extract_doc_content_with_vllm(clean_markdown)

        if not vllm_success:
            # vLLM not available or content too large - warn and don't save to clean/
            result.issues.append(
                ValidationIssue(
                    "warning",
                    "reference_url",
                    f"vLLM unavailable for {url} (HTML stripped: {html_reduction:.0f}%)",
                )
            )
            return True  # Raw file still saved

        filtered_len = len(filtered_content)
        total_reduction = (1 - filtered_len / original_len) * 100 if original_len > 0 else 0

        # Save to clean/ if we achieved significant total reduction
        if filtered_len < original_len * 0.9:
            with open(clean_path, "w", encoding="utf-8") as f:
                f.write(header)
                f.write(filtered_content)

            # Check if doc is too large (suggests unfocused reference)
            line_count = filtered_content.count("\n") + 1
            if line_count > MAX_DOC_LINES:
                result.issues.append(
                    ValidationIssue(
                        "warning",
                        "reference_url",
                        f"Doc too large ({line_count} lines > {MAX_DOC_LINES}): {url} - find more focused page",
                    )
                )
        else:
            result.issues.append(
                ValidationIssue(
                    "warning",
                    "reference_url",
                    f"Doc cleaning ineffective for {url} (total: {total_reduction:.0f}%)",
                )
            )

        return True

    except httpx.HTTPStatusError as e:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"HTTP {e.response.status_code} fetching {url}",
            )
        )
        return False
    except httpx.RequestError as e:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"Failed to fetch {url}: {e}",
            )
        )
        return False
    except Exception as e:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"Error fetching {url}: {e}",
            )
        )
        return False


def check_url_reachable(url: str, result: ValidationResult) -> bool:
    """Check if URL is reachable with HEAD request. Returns True if OK."""
    import httpx

    try:
        with httpx.Client(
            timeout=5.0,
            follow_redirects=True,
            headers={"User-Agent": "scicode-lint/1.0 (pattern-verification)"},
        ) as client:
            response = client.head(url)
            response.raise_for_status()
            return True

    except httpx.HTTPStatusError as e:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"HTTP {e.response.status_code} for {url}",
            )
        )
        return False
    except httpx.RequestError as e:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"Cannot reach {url}: {e}",
            )
        )
        return False
    except Exception as e:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"Error checking {url}: {e}",
            )
        )
        return False


def check_reference_urls(
    toml_data: dict[str, Any], result: ValidationResult, fetch: bool = False
) -> None:
    """Check reference URLs (HEAD request by default, full fetch with --fetch-refs)."""
    meta = toml_data.get("meta", {})
    references = meta.get("references", [])
    if not references:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                "No reference URLs - add links to official documentation",
            )
        )
        return

    if len(references) > MAX_REFERENCE_URLS:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"Too many references ({len(references)}), max is {MAX_REFERENCE_URLS}",
            )
        )

    for url in references:
        if not isinstance(url, str):
            result.issues.append(
                ValidationIssue("error", "reference_url", f"Invalid reference: {url}")
            )
            continue

        if not url.startswith(("http://", "https://")):
            result.issues.append(
                ValidationIssue("error", "reference_url", f"Invalid URL format: {url}")
            )
            continue

        if fetch:
            # Full fetch and cache (prefix with pattern ID for easy lookup)
            fetch_and_cache_url(url, result, result.pattern_id)
        else:
            # Lightweight HEAD check (default)
            check_url_reachable(url, result)


def collect_all_pattern_references(patterns_dir: Path) -> list[tuple[str, str]]:
    """Collect all (pattern_id, url) pairs from all patterns."""
    refs: list[tuple[str, str]] = []
    for category in patterns_dir.iterdir():
        if not category.is_dir() or category.name.startswith((".", "_")):
            continue
        for pattern_dir in category.iterdir():
            if not pattern_dir.is_dir():
                continue
            toml_path = pattern_dir / "pattern.toml"
            if not toml_path.exists():
                continue
            pattern_id = pattern_dir.name
            try:
                with open(toml_path, "rb") as f:
                    data = tomllib.load(f)
                for url in data.get("meta", {}).get("references", []):
                    if isinstance(url, str):
                        refs.append((pattern_id, url))
            except Exception:
                continue
    return refs


def clean_orphaned_cache(patterns_dir: Path) -> list[str]:
    """Remove cached docs that are no longer referenced by any pattern.

    Returns list of removed files.
    """
    if not DOC_CACHE_DIR.exists():
        return []

    # Get all referenced (pattern_id, url) pairs and their cache filenames
    refs = collect_all_pattern_references(patterns_dir)
    referenced_filenames = {get_cache_filename(url, pattern_id) for pattern_id, url in refs}

    # Find and remove orphaned cache files from both raw/ and clean/ subdirs
    # Also clean up legacy files in root doc_cache/
    removed: list[str] = []
    dirs_to_check = [DOC_CACHE_DIR, DOC_CACHE_RAW_DIR, DOC_CACHE_CLEAN_DIR]

    for cache_dir in dirs_to_check:
        if not cache_dir.exists():
            continue
        for file_pattern in ("*.md", "*.txt"):
            for cache_file in cache_dir.glob(file_pattern):
                if cache_file.name not in referenced_filenames:
                    cache_file.unlink()
                    removed.append(f"{cache_dir.name}/{cache_file.name}")

    return removed


# =============================================================================
# FIX FUNCTIONS
# =============================================================================


def find_best_match(wrong_name: str, available_names: set[str]) -> str | None:
    """Find the best matching filename from available names."""
    best_match = None
    best_ratio = 0.6

    for name in available_names:
        ratio = SequenceMatcher(None, wrong_name, name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = name

    return best_match


def extract_first_function_or_class(file_path: Path) -> tuple[str, str, str]:
    """Extract first function/class name and a code snippet."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                snippet = ""
                for stmt in node.body:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                        continue
                    if hasattr(stmt, "lineno"):
                        lines = source.split("\n")
                        if stmt.lineno <= len(lines):
                            snippet = lines[stmt.lineno - 1].strip()
                            break
                return "function", node.name, snippet or f"def {node.name}"
            elif isinstance(node, ast.ClassDef):
                return "class", node.name, f"class {node.name}"

        return "module", file_path.stem, ""
    except Exception:
        return "module", file_path.stem, ""


def fix_toml_sync(
    pattern_dir: Path, sync_issues: dict[str, dict[str, list[str]]], result: ValidationResult
) -> None:
    """Fix TOML/file sync issues."""
    toml_path = pattern_dir / "pattern.toml"
    content = toml_path.read_text()

    missing_on_disk = sync_issues.get("missing_on_disk", {})
    missing_in_toml = sync_issues.get("missing_in_toml", {})

    # Try to rename mismatched entries
    for test_type in ["positive", "negative", "context_dependent"]:
        wrong_names = list(missing_on_disk.get(test_type, []))
        available_names = set(missing_in_toml.get(test_type, []))

        for wrong_name in wrong_names:
            best_match = find_best_match(wrong_name, available_names)
            if best_match:
                old_path = f"test_{test_type}/{wrong_name}"
                new_path = f"test_{test_type}/{best_match}"
                content = content.replace(f'"{old_path}"', f'"{new_path}"')
                result.fixed.append(f"Renamed {wrong_name} -> {best_match}")
                available_names.discard(best_match)
            else:
                # Remove orphaned entry
                lines = content.split("\n")
                new_lines: list[str] = []
                skip_block = False
                old_path = f"test_{test_type}/{wrong_name}"

                for line in lines:
                    if f'"{old_path}"' in line:
                        j = len(new_lines) - 1
                        while j >= 0 and not new_lines[j].strip().startswith("[[tests."):
                            j -= 1
                        new_lines = new_lines[:j]
                        skip_block = True
                        result.fixed.append(f"Removed orphaned entry: {wrong_name}")
                    elif skip_block:
                        starts_new = line.strip().startswith(("[[tests.", "[tests."))
                        if starts_new:
                            skip_block = False
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                content = "\n".join(new_lines)

    # Re-read to get updated missing_in_toml
    toml_path.write_text(content)

    # Re-check what's still missing
    with open(toml_path, "rb") as f:
        updated_toml = tomllib.load(f)
    on_disk = get_test_files_on_disk(pattern_dir)
    in_toml = get_test_files_in_toml(updated_toml)

    # Add missing entries
    new_entries = []
    for test_type in ["positive", "negative", "context_dependent"]:
        still_missing = on_disk[test_type] - in_toml[test_type]
        for filename in sorted(still_missing):
            if test_type == "positive":
                file_path = pattern_dir / "test_positive" / filename
                loc_type, loc_name, snippet = extract_first_function_or_class(file_path)
                snippet = snippet.replace('"', '\\"')
                new_entries.append(f'''
[[tests.positive]]
file = "test_positive/{filename}"
description = "TODO: Add description"
expected_issue = "TODO: Add expected issue"
min_confidence = 0.85

[tests.positive.expected_location]
type = "{loc_type}"
name = "{loc_name}"
snippet = "{snippet}"
''')
            elif test_type == "negative":
                new_entries.append(f"""
[[tests.negative]]
file = "test_negative/{filename}"
description = "TODO: Add description"
""")
            else:
                new_entries.append(f"""
[[tests.context_dependent]]
file = "test_context_dependent/{filename}"
description = "TODO: Add description"
context_notes = "TODO: Add context notes"
allow_detection = true
allow_skip = true
""")
            result.fixed.append(f"Added entry for {test_type}/{filename}")

    if new_entries:
        content = toml_path.read_text()
        with open(toml_path, "a") as f:
            f.write("\n# === Auto-generated entries (review and update) ===")
            for entry in new_entries:
                f.write(entry)


# =============================================================================
# MAIN VALIDATION
# =============================================================================


def validate_pattern(
    pattern_dir: Path, fix: bool = False, fetch_refs: bool = False
) -> ValidationResult:
    """Run all validation checks on a pattern."""
    result = ValidationResult(
        pattern_id=pattern_dir.name,
        category=pattern_dir.parent.name,
    )

    toml_path = pattern_dir / "pattern.toml"
    if not toml_path.exists():
        result.issues.append(ValidationIssue("error", "missing", "pattern.toml not found"))
        return result

    # Load TOML
    try:
        with open(toml_path, "rb") as f:
            toml_data = tomllib.load(f)
    except Exception as e:
        result.issues.append(ValidationIssue("error", "toml_parse", f"Failed to parse TOML: {e}"))
        return result

    # Run checks
    sync_issues = check_toml_file_sync(pattern_dir, toml_data, result)
    check_schema(pattern_dir, toml_data, result)
    check_data_leakage(pattern_dir, result)
    check_test_count(pattern_dir, result)
    check_todo_markers(toml_path, result)
    check_detection_question(toml_data, result)
    check_test_syntax(pattern_dir, result)
    check_empty_fields(toml_data, result)
    check_category_mismatch(pattern_dir, toml_data, result)
    check_expected_location_snippets(pattern_dir, toml_data, result)
    check_related_patterns_exist(pattern_dir, toml_data, result)
    check_test_diversity(pattern_dir, result)
    check_reference_urls(toml_data, result, fetch=fetch_refs)

    # Apply fixes if requested
    if fix and (sync_issues.get("missing_in_toml") or sync_issues.get("missing_on_disk")):
        fix_toml_sync(pattern_dir, sync_issues, result)

    return result


def find_all_patterns(patterns_dir: Path) -> list[Path]:
    """Find all pattern directories."""
    patterns = []
    for category in patterns_dir.iterdir():
        if not category.is_dir() or category.name.startswith((".", "_")):
            continue
        for pattern_dir in category.iterdir():
            if not pattern_dir.is_dir() or pattern_dir.name.startswith((".", "_")):
                continue
            if (pattern_dir / "pattern.toml").exists():
                patterns.append(pattern_dir)
    return sorted(patterns)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pattern", nargs="?", help="Specific pattern ID (e.g., ml-002)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix what's possible")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show issues")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument(
        "--fetch-refs",
        action="store_true",
        help=f"Fetch and cache reference docs (expires after {DOC_CACHE_MAX_AGE_DAYS} days)",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Remove orphaned doc cache files (no longer referenced by any pattern)",
    )
    args = parser.parse_args()

    patterns_dir = Path("patterns")
    if not patterns_dir.exists():
        print("Error: patterns/ directory not found", file=sys.stderr)
        return 1

    # Clean orphaned cache if requested
    if args.clean_cache:
        removed = clean_orphaned_cache(patterns_dir)
        if removed:
            print(f"Removed {len(removed)} orphaned cache file(s):")
            for f in removed:
                print(f"  - {f}")
        else:
            print("No orphaned cache files found.")

    # Find patterns
    if args.pattern:
        matches = list(patterns_dir.glob(f"*/{args.pattern}"))
        if not matches:
            print(f"Error: Pattern '{args.pattern}' not found", file=sys.stderr)
            return 1
        patterns = matches
    else:
        patterns = find_all_patterns(patterns_dir)

    # Validate
    error_count = 0
    warning_count = 0
    fixed_count = 0

    for pattern_dir in patterns:
        result = validate_pattern(pattern_dir, fix=args.fix, fetch_refs=args.fetch_refs)

        has_errors = result.has_errors
        has_warnings = result.has_warnings

        if has_errors:
            error_count += 1
        if has_warnings:
            warning_count += 1
        if result.fixed:
            fixed_count += 1

        # Print results
        if result.issues or result.fixed:
            prefix = "❌" if has_errors else ("⚠" if has_warnings else "✓")
            print(f"\n{prefix} {result.category}/{result.pattern_id}")

            for issue in result.issues:
                icon = "✗" if issue.level == "error" else "⚡"
                file_info = f" [{issue.file}]" if issue.file else ""
                print(f"   {icon} [{issue.check}]{file_info} {issue.message}")

            for fix_msg in result.fixed:
                print(f"   ✓ Fixed: {fix_msg}")

        elif not args.quiet:
            print(f"✓ {result.category}/{result.pattern_id}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Checked {len(patterns)} patterns")
    print(f"  Errors: {error_count}")
    print(f"  Warnings: {warning_count}")
    if args.fix:
        print(f"  Fixed: {fixed_count}")

    if args.strict:
        return 1 if (error_count > 0 or warning_count > 0) else 0
    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
