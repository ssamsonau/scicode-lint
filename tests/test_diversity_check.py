"""Tests for diversity_check module (semantic test file diversity)."""

from pathlib import Path

import pytest

from pattern_verification.deterministic.diversity_check import (
    BATCH_DIVERSITY_PROMPT,
    CROSS_TYPE_PAIR_TEMPLATE,
    SAME_TYPE_PAIR_TEMPLATE,
    PatternDiversityResult,
    _build_pairs_section,
    _parse_verdicts,
    format_results,
    load_pattern_info,
    load_test_files,
)


class TestPatternDiversityResult:
    """Tests for PatternDiversityResult dataclass."""

    def test_has_issues_false_when_empty(self) -> None:
        """Result with no issues should return has_issues=False."""
        result = PatternDiversityResult(
            pattern_id="ml-001",
            category="ai-training",
        )
        assert result.has_issues is False

    def test_has_issues_true_with_redundant_positive(self) -> None:
        """Result with redundant positive pairs should return has_issues=True."""
        result = PatternDiversityResult(
            pattern_id="ml-001",
            category="ai-training",
            redundant_positive_pairs=[("file1.py", "file2.py")],
        )
        assert result.has_issues is True

    def test_has_issues_true_with_redundant_negative(self) -> None:
        """Result with redundant negative pairs should return has_issues=True."""
        result = PatternDiversityResult(
            pattern_id="ml-001",
            category="ai-training",
            redundant_negative_pairs=[("file1.py", "file2.py")],
        )
        assert result.has_issues is True

    def test_has_issues_true_with_fixed_copies(self) -> None:
        """Result with fixed copy negatives should return has_issues=True."""
        result = PatternDiversityResult(
            pattern_id="ml-001",
            category="ai-training",
            fixed_copy_negatives=[("neg.py", "pos.py")],
        )
        assert result.has_issues is True

    def test_total_comparisons_default(self) -> None:
        """Total comparisons should default to 0."""
        result = PatternDiversityResult(
            pattern_id="ml-001",
            category="ai-training",
        )
        assert result.total_comparisons == 0


class TestLoadPatternInfo:
    """Tests for load_pattern_info function."""

    def test_load_valid_pattern(self, tmp_path: Path) -> None:
        """Should load name and question from valid pattern.toml."""
        pattern_dir = tmp_path / "ml-001"
        pattern_dir.mkdir()
        (pattern_dir / "pattern.toml").write_text(
            """
[meta]
name = "scaler-leakage"

[detection]
question = "Does this code have data leakage?"
"""
        )
        name, question = load_pattern_info(pattern_dir)
        assert name == "scaler-leakage"
        assert question == "Does this code have data leakage?"

    def test_load_missing_toml(self, tmp_path: Path) -> None:
        """Should raise error for missing pattern.toml."""
        pattern_dir = tmp_path / "ml-001"
        pattern_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_pattern_info(pattern_dir)


class TestLoadTestFiles:
    """Tests for load_test_files function."""

    def test_load_multiple_files(self, tmp_path: Path) -> None:
        """Should load all .py files from directory."""
        test_dir = tmp_path / "test_positive"
        test_dir.mkdir()
        (test_dir / "file1.py").write_text("# file 1")
        (test_dir / "file2.py").write_text("# file 2")

        files = load_test_files(test_dir)
        assert len(files) == 2
        assert files["file1.py"] == "# file 1"
        assert files["file2.py"] == "# file 2"

    def test_skip_underscore_files(self, tmp_path: Path) -> None:
        """Should skip files starting with underscore."""
        test_dir = tmp_path / "test_positive"
        test_dir.mkdir()
        (test_dir / "file1.py").write_text("# file 1")
        (test_dir / "_helper.py").write_text("# helper")
        (test_dir / "__init__.py").write_text("")

        files = load_test_files(test_dir)
        assert len(files) == 1
        assert "file1.py" in files

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Should return empty dict for empty directory."""
        test_dir = tmp_path / "test_positive"
        test_dir.mkdir()

        files = load_test_files(test_dir)
        assert files == {}

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Should return empty dict for nonexistent directory."""
        test_dir = tmp_path / "test_positive"

        files = load_test_files(test_dir)
        assert files == {}


class TestPrompts:
    """Tests for prompt templates."""

    def test_batch_prompt_has_placeholders(self) -> None:
        """Batch prompt should have all required placeholders."""
        assert "{pattern_name}" in BATCH_DIVERSITY_PROMPT
        assert "{detection_question}" in BATCH_DIVERSITY_PROMPT
        assert "{pairs_section}" in BATCH_DIVERSITY_PROMPT

    def test_same_type_pair_template_has_placeholders(self) -> None:
        """Same-type pair template should have all required placeholders."""
        assert "{pair_id}" in SAME_TYPE_PAIR_TEMPLATE
        assert "{test_type}" in SAME_TYPE_PAIR_TEMPLATE
        assert "{filename1}" in SAME_TYPE_PAIR_TEMPLATE
        assert "{code1}" in SAME_TYPE_PAIR_TEMPLATE
        assert "{filename2}" in SAME_TYPE_PAIR_TEMPLATE
        assert "{code2}" in SAME_TYPE_PAIR_TEMPLATE

    def test_cross_type_pair_template_has_placeholders(self) -> None:
        """Cross-type pair template should have all required placeholders."""
        assert "{pair_id}" in CROSS_TYPE_PAIR_TEMPLATE
        assert "{pos_filename}" in CROSS_TYPE_PAIR_TEMPLATE
        assert "{positive_code}" in CROSS_TYPE_PAIR_TEMPLATE
        assert "{neg_filename}" in CROSS_TYPE_PAIR_TEMPLATE
        assert "{negative_code}" in CROSS_TYPE_PAIR_TEMPLATE

    def test_same_type_pair_formatting(self) -> None:
        """Same-type pair template should format correctly."""
        text = SAME_TYPE_PAIR_TEMPLATE.format(
            pair_id=1,
            test_type="positive",
            filename1="file1.py",
            code1="# code 1",
            filename2="file2.py",
            code2="# code 2",
        )
        assert "PAIR 1" in text
        assert "file1.py" in text
        assert "# code 1" in text
        assert "positive-positive" in text

    def test_cross_type_pair_formatting(self) -> None:
        """Cross-type pair template should format correctly."""
        text = CROSS_TYPE_PAIR_TEMPLATE.format(
            pair_id=3,
            pos_filename="positive.py",
            positive_code="# buggy code",
            neg_filename="negative.py",
            negative_code="# correct code",
        )
        assert "PAIR 3" in text
        assert "positive.py" in text
        assert "# buggy code" in text
        assert "negative.py" in text


class TestBuildPairsSection:
    """Tests for _build_pairs_section function."""

    def test_no_files(self) -> None:
        """Empty files should produce empty pairs."""
        text, pairs = _build_pairs_section({}, {})
        assert pairs == []
        assert text == ""

    def test_positive_pairs_only(self) -> None:
        """Should generate positive-positive pairs."""
        pos = {"a.py": "code a", "b.py": "code b", "c.py": "code c"}
        text, pairs = _build_pairs_section(pos, {})
        assert len(pairs) == 3  # C(3,2) = 3
        assert all(p.kind == "positive" for p in pairs)

    def test_cross_pairs(self) -> None:
        """Should generate cross-type pairs."""
        pos = {"a.py": "code a"}
        neg = {"x.py": "code x", "y.py": "code y"}
        text, pairs = _build_pairs_section(pos, neg)
        # 0 pos-pos + 1 neg-neg + 2 cross = 3
        assert len(pairs) == 3
        cross = [p for p in pairs if p.kind == "cross"]
        assert len(cross) == 2

    def test_pair_ids_sequential(self) -> None:
        """Pair IDs should be sequential starting from 1."""
        pos = {"a.py": "code a", "b.py": "code b"}
        neg = {"x.py": "code x"}
        _, pairs = _build_pairs_section(pos, neg)
        ids = [p.pair_id for p in pairs]
        assert ids == list(range(1, len(pairs) + 1))


class TestParseVerdicts:
    """Tests for _parse_verdicts function."""

    def test_valid_json_array(self) -> None:
        """Should parse valid JSON array."""
        output = '[{"pair_id": 1, "verdict": "diverse"}, {"pair_id": 2, "verdict": "redundant"}]'
        verdicts = _parse_verdicts(output, [])
        assert verdicts == {1: "diverse", 2: "redundant"}

    def test_json_with_surrounding_text(self) -> None:
        """Should extract JSON from text with surrounding content."""
        output = 'Here are the results:\n[{"pair_id": 1, "verdict": "diverse"}]\nDone.'
        verdicts = _parse_verdicts(output, [])
        assert verdicts == {1: "diverse"}

    def test_invalid_json(self) -> None:
        """Should return None for invalid JSON."""
        verdicts = _parse_verdicts("not json at all", [])
        assert verdicts is None

    def test_invalid_verdict_ignored(self) -> None:
        """Should ignore entries with invalid verdict values."""
        output = '[{"pair_id": 1, "verdict": "unknown"}]'
        verdicts = _parse_verdicts(output, [])
        assert verdicts == {}


class TestFormatResults:
    """Tests for format_results function."""

    def test_format_no_issues(self) -> None:
        """Should format correctly when no issues found."""
        results = [
            PatternDiversityResult(
                pattern_id="ml-001",
                category="ai-training",
                total_comparisons=10,
            ),
            PatternDiversityResult(
                pattern_id="ml-002",
                category="ai-training",
                total_comparisons=15,
            ),
        ]
        output = format_results(results)
        assert "No diversity issues found" in output
        assert "0 pattern(s) with issues out of 2 checked" in output
        assert "25 total comparisons made" in output

    def test_format_with_redundant_pairs(self) -> None:
        """Should format redundant pairs correctly with full paths."""
        results = [
            PatternDiversityResult(
                pattern_id="ml-001",
                category="ai-training",
                redundant_positive_pairs=[("file1.py", "file2.py")],
                total_comparisons=10,
            ),
        ]
        output = format_results(results)
        assert "ISSUES FOUND:" in output
        assert "ai-training/ml-001" in output
        assert "Redundant positive pairs:" in output
        assert "test_positive/file1.py" in output
        assert "test_positive/file2.py" in output
        assert "1 pattern(s) with issues out of 1 checked" in output
        assert "1 redundant" in output

    def test_format_with_fixed_copies(self) -> None:
        """Should format fixed copy negatives correctly with full paths."""
        results = [
            PatternDiversityResult(
                pattern_id="ml-001",
                category="ai-training",
                fixed_copy_negatives=[("neg.py", "pos.py")],
                total_comparisons=10,
            ),
        ]
        output = format_results(results)
        assert "Non-diverse negatives" in output
        assert "test_negative/neg.py" in output
        assert "test_positive/pos.py" in output
        assert "1 non-diverse" in output

    def test_format_multiple_issues(self) -> None:
        """Should format multiple issue types correctly."""
        results = [
            PatternDiversityResult(
                pattern_id="ml-001",
                category="ai-training",
                redundant_positive_pairs=[("p1.py", "p2.py")],
                redundant_negative_pairs=[("n1.py", "n2.py")],
                fixed_copy_negatives=[("neg.py", "pos.py")],
                total_comparisons=20,
            ),
        ]
        output = format_results(results)
        assert "Redundant positive pairs:" in output
        assert "Redundant negative pairs:" in output
        assert "Non-diverse negatives" in output
        assert "2 redundant" in output
        assert "1 non-diverse" in output
