"""Tests for the eval runner (evals/run_eval.py)."""

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import from evals package
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.run_eval import LLMJudgeEvaluator
from scicode_lint.config import LLMConfig

# Use anyio for async tests (already installed as pytest plugin)
pytestmark = pytest.mark.anyio


class TestRunLinter:
    """Tests for the run_linter method."""

    @pytest.fixture
    def evaluator(self) -> LLMJudgeEvaluator:
        """Create evaluator with mocked LLM client."""
        mock_client = MagicMock()
        llm_config = LLMConfig(base_url="http://localhost:5001")
        patterns_dir = Path(__file__).parent.parent / "patterns"
        return LLMJudgeEvaluator(
            llm_client=mock_client,
            patterns_dir=patterns_dir,
            llm_config=llm_config,
            skip_judge=True,
        )

    async def test_run_linter_with_finding(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test run_linter returns correct format when finding is detected."""
        # Create mock linter with mock result
        mock_linter = MagicMock()
        mock_finding = MagicMock()
        mock_finding.detection_type = "yes"
        mock_finding.location = MagicMock()
        mock_finding.location.lines = [10, 11]
        mock_finding.location.snippet = "x = scaler.fit_transform(X)"
        mock_finding.reasoning = "Scaler fitted on test data"
        mock_finding.issue = "Data leakage detected"
        mock_finding.confidence = 0.95
        mock_finding.explanation = "This causes data leakage"
        mock_finding.thinking = "Analyzed the code..."

        mock_result = MagicMock()
        mock_result.findings = [mock_finding]
        mock_result.checked_patterns = []

        mock_linter._check_file_async = AsyncMock(return_value=mock_result)

        # Run linter
        result = await evaluator.run_linter(Path("dummy.py"), mock_linter)

        # Verify result format
        assert result["detected"] == "yes"
        assert result["lines"] == [10, 11]
        assert result["snippet"] == "x = scaler.fit_transform(X)"
        assert result["reasoning"] == "Scaler fitted on test data"
        assert result["confidence"] == 0.95
        assert result["thinking"] == "Analyzed the code..."

    async def test_run_linter_no_finding(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test run_linter returns correct format when no finding detected."""
        mock_linter = MagicMock()

        # Mock checked_patterns (no findings)
        mock_check = MagicMock()
        mock_check.detected = "no"
        mock_check.reasoning = "No issues found"
        mock_check.confidence = 0.9
        mock_check.thinking = "Code looks clean"

        mock_result = MagicMock()
        mock_result.findings = []
        mock_result.checked_patterns = [mock_check]

        mock_linter._check_file_async = AsyncMock(return_value=mock_result)

        result = await evaluator.run_linter(Path("dummy.py"), mock_linter)

        assert result["detected"] == "no"
        assert result["lines"] == []
        assert result["confidence"] == 0.9
        assert result["reasoning"] == "No issues found"

    async def test_run_linter_error_handling(self, evaluator: LLMJudgeEvaluator) -> None:
        """Test run_linter handles errors gracefully."""
        mock_linter = MagicMock()
        mock_linter._check_file_async = AsyncMock(side_effect=RuntimeError("Connection failed"))

        result = await evaluator.run_linter(Path("dummy.py"), mock_linter)

        assert result["detected"] == "no"
        assert result["confidence"] == 0.0
        assert "Connection failed" in result["explanation"]


class TestEvaluateDirect:
    """Tests for direct metrics evaluation."""

    @pytest.fixture
    def evaluator(self) -> LLMJudgeEvaluator:
        """Create evaluator with mocked LLM client."""
        mock_client = MagicMock()
        llm_config = LLMConfig(base_url="http://localhost:5001")
        patterns_dir = Path(__file__).parent.parent / "patterns"
        return LLMJudgeEvaluator(
            llm_client=mock_client,
            patterns_dir=patterns_dir,
            llm_config=llm_config,
        )

    def test_positive_test_detected(self, evaluator: LLMJudgeEvaluator) -> None:
        """Positive test passes when bug is detected with sufficient confidence."""
        linter_output = {"detected": "yes", "confidence": 0.95}
        passed, reason = evaluator._evaluate_direct("positive", linter_output)
        assert passed is True
        assert "detected" in reason.lower()

    def test_positive_test_not_detected(self, evaluator: LLMJudgeEvaluator) -> None:
        """Positive test fails when bug is not detected."""
        linter_output = {"detected": "no", "confidence": 0.0}
        passed, reason = evaluator._evaluate_direct("positive", linter_output)
        assert passed is False
        assert "not detected" in reason.lower()

    def test_positive_test_low_confidence(self, evaluator: LLMJudgeEvaluator) -> None:
        """Positive test fails when confidence is below threshold."""
        linter_output = {"detected": "yes", "confidence": 0.5}
        passed, reason = evaluator._evaluate_direct("positive", linter_output, min_confidence=0.8)
        assert passed is False
        assert "confidence" in reason.lower()

    def test_negative_test_no_detection(self, evaluator: LLMJudgeEvaluator) -> None:
        """Negative test passes when no bug is detected."""
        linter_output = {"detected": "no", "confidence": 0.0}
        passed, reason = evaluator._evaluate_direct("negative", linter_output)
        assert passed is True

    def test_negative_test_false_positive(self, evaluator: LLMJudgeEvaluator) -> None:
        """Negative test fails when bug is incorrectly detected."""
        linter_output = {"detected": "yes", "confidence": 0.9}
        passed, reason = evaluator._evaluate_direct("negative", linter_output)
        assert passed is False
        assert "false positive" in reason.lower()

    def test_context_dependent_always_passes(self, evaluator: LLMJudgeEvaluator) -> None:
        """Context-dependent test always passes."""
        for detected in ["yes", "no", "context-dependent"]:
            linter_output = {"detected": detected, "confidence": 0.5}
            passed, _ = evaluator._evaluate_direct("context_dependent", linter_output)
            assert passed is True


class TestSemaphoreUsage:
    """Tests for semaphore-based concurrency control."""

    async def test_semaphore_limits_concurrency(self) -> None:
        """Verify semaphore limits concurrent evaluations."""
        mock_client = MagicMock()
        llm_config = LLMConfig(base_url="http://localhost:5001")
        patterns_dir = Path(__file__).parent.parent / "patterns"

        # Create evaluator with very low concurrency limit
        evaluator = LLMJudgeEvaluator(
            llm_client=mock_client,
            patterns_dir=patterns_dir,
            llm_config=llm_config,
            max_concurrent=2,  # Only allow 2 concurrent
            skip_judge=True,
        )

        # Track concurrent executions
        current_concurrent = 0
        max_concurrent_seen = 0

        async def mock_check_file(path: Path) -> MagicMock:
            nonlocal current_concurrent, max_concurrent_seen
            current_concurrent += 1
            max_concurrent_seen = max(max_concurrent_seen, current_concurrent)
            await asyncio.sleep(0.1)  # Simulate work
            current_concurrent -= 1

            result = MagicMock()
            result.findings = []
            result.checked_patterns = [
                MagicMock(detected="no", reasoning="", confidence=0.9, thinking=None)
            ]
            return result

        # Create mock linter
        mock_linter = MagicMock()
        mock_linter._check_file_async = mock_check_file

        # Run multiple evaluations concurrently
        async def run_eval() -> dict[str, Any]:
            async with evaluator._semaphore:
                return await evaluator.run_linter(Path("dummy.py"), mock_linter)

        # Launch 10 concurrent tasks
        tasks = [run_eval() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Verify concurrency was limited
        assert max_concurrent_seen <= 2, f"Max concurrent was {max_concurrent_seen}, expected <= 2"
