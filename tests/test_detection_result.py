"""Tests for DetectionResult model validation."""

import pytest
from pydantic import ValidationError

from scicode_lint.llm.models import DetectionResult, NamedLocation


class TestDetectionResultValidation:
    """Tests for DetectionResult location validation."""

    def test_detected_yes_with_location_is_valid(self) -> None:
        """detected='yes' with location should be valid."""
        result = DetectionResult(
            detected="yes",
            location=NamedLocation(
                name="train_model",
                location_type="function",
                near_line=15,
            ),
            confidence=0.95,
            reasoning="Issue found in train_model function",
        )
        assert result.detected == "yes"
        assert result.location is not None
        assert result.location.name == "train_model"

    def test_detected_yes_without_location_raises_error(self) -> None:
        """detected='yes' with no location should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionResult(
                detected="yes",
                location=None,
                confidence=0.95,
                reasoning="Issue found but no location specified",
            )
        # Check error message mentions location
        error_msg = str(exc_info.value)
        assert "Location required" in error_msg
        assert "detected='yes'" in error_msg

    def test_detected_context_dependent_with_location_is_valid(self) -> None:
        """detected='context-dependent' with location should be valid."""
        result = DetectionResult(
            detected="context-dependent",
            location=NamedLocation(
                name="Trainer.fit",
                location_type="method",
                near_line=10,
            ),
            confidence=0.75,
            reasoning="Depends on context",
        )
        assert result.detected == "context-dependent"
        assert result.location is not None

    def test_detected_context_dependent_without_location_raises_error(self) -> None:
        """detected='context-dependent' with no location should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionResult(
                detected="context-dependent",
                location=None,
                confidence=0.75,
                reasoning="Context dependent but no location",
            )
        error_msg = str(exc_info.value)
        assert "Location required" in error_msg
        assert "detected='context-dependent'" in error_msg

    def test_detected_no_with_null_location_is_valid(self) -> None:
        """detected='no' with null location should be valid."""
        result = DetectionResult(
            detected="no",
            location=None,
            confidence=0.9,
            reasoning="No issue found",
        )
        assert result.detected == "no"
        assert result.location is None

    def test_detected_no_with_location_is_valid(self) -> None:
        """detected='no' with location (edge case) should be valid."""
        # Model might reference location even when saying "no issue"
        result = DetectionResult(
            detected="no",
            location=NamedLocation(
                name="process_data",
                location_type="function",
            ),
            confidence=0.85,
            reasoning="Checked process_data, no issue",
        )
        assert result.detected == "no"
        assert result.location is not None


class TestNamedLocation:
    """Tests for NamedLocation model."""

    def test_minimal_location(self) -> None:
        """Location with just name and type should be valid."""
        loc = NamedLocation(
            name="my_function",
            location_type="function",
        )
        assert loc.name == "my_function"
        assert loc.location_type == "function"
        assert loc.near_line is None

    def test_location_with_near_line(self) -> None:
        """Location with near_line should be valid."""
        loc = NamedLocation(
            name="Trainer.train",
            location_type="method",
            near_line=42,
        )
        assert loc.near_line == 42

    def test_module_level_location(self) -> None:
        """Module-level location should be valid."""
        loc = NamedLocation(
            name="<module>",
            location_type="module",
            near_line=5,
        )
        assert loc.location_type == "module"


class TestDetectionResultSchema:
    """Tests for DetectionResult JSON schema."""

    def test_location_in_schema(self) -> None:
        """location field should be in schema properties."""
        schema = DetectionResult.model_json_schema()
        assert "location" in schema.get("properties", {})

    def test_all_required_fields_present(self) -> None:
        """All non-optional fields should be required."""
        schema = DetectionResult.model_json_schema()
        required = schema.get("required", [])
        assert "detected" in required
        assert "confidence" in required
        assert "reasoning" in required
        # location has default=None, so it may not be in required
        # thinking has default=None, should NOT be required
        assert "thinking" not in required


class TestDetectionResultThinking:
    """Tests for thinking field handling."""

    def test_thinking_is_optional(self) -> None:
        """thinking field should default to None."""
        result = DetectionResult(
            detected="no",
            location=None,
            confidence=0.9,
            reasoning="No issue",
        )
        assert result.thinking is None

    def test_thinking_can_be_set(self) -> None:
        """thinking field can be explicitly set."""
        result = DetectionResult(
            detected="yes",
            location=NamedLocation(name="train", location_type="function"),
            confidence=0.95,
            reasoning="Found issue",
            thinking="Let me analyze this code...",
        )
        assert result.thinking == "Let me analyze this code..."
