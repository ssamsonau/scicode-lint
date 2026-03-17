"""Tests for AST utility functions."""

from scicode_lint.ast_utils import (
    ResolvedLocation,
    find_all_definitions,
    resolve_name,
    resolve_name_with_fallback,
)


class TestFindAllDefinitions:
    """Tests for find_all_definitions function."""

    def test_find_function(self) -> None:
        """Test finding a standalone function."""
        code = """
def train_model(data):
    pass
"""
        definitions = find_all_definitions(code)
        assert len(definitions) == 1
        assert definitions[0].name == "train_model"
        assert definitions[0].location_type == "function"
        assert definitions[0].start_line == 2

    def test_find_class(self) -> None:
        """Test finding a class."""
        code = """
class Trainer:
    pass
"""
        definitions = find_all_definitions(code)
        assert len(definitions) == 1
        assert definitions[0].name == "Trainer"
        assert definitions[0].location_type == "class"

    def test_find_method(self) -> None:
        """Test finding a method inside a class."""
        code = """
class Trainer:
    def fit(self, data):
        pass
"""
        definitions = find_all_definitions(code)
        assert len(definitions) == 2
        # Class comes first
        assert definitions[0].name == "Trainer"
        assert definitions[0].location_type == "class"
        # Method with qualified name
        assert definitions[1].name == "Trainer.fit"
        assert definitions[1].location_type == "method"

    def test_find_async_function(self) -> None:
        """Test finding an async function."""
        code = """
async def fetch_data(url):
    pass
"""
        definitions = find_all_definitions(code)
        assert len(definitions) == 1
        assert definitions[0].name == "fetch_data"
        assert definitions[0].location_type == "function"

    def test_find_nested_class(self) -> None:
        """Test finding nested class with qualified name."""
        code = """
class Outer:
    class Inner:
        def method(self):
            pass
"""
        definitions = find_all_definitions(code)
        names = [d.name for d in definitions]
        assert "Outer" in names
        assert "Outer.Inner" in names
        assert "Outer.Inner.method" in names

    def test_invalid_syntax(self) -> None:
        """Test that invalid syntax returns empty list."""
        code = "def broken("
        definitions = find_all_definitions(code)
        assert definitions == []


class TestResolveName:
    """Tests for resolve_name function."""

    def test_resolve_function(self) -> None:
        """Test resolving a function by name."""
        code = """
def process_data(x):
    return x * 2
"""
        result = resolve_name(code, "process_data")
        assert result is not None
        assert result.name == "process_data"
        assert result.start_line == 2

    def test_resolve_method_qualified(self) -> None:
        """Test resolving a method using qualified name."""
        code = """
class Model:
    def train(self, data):
        pass
"""
        result = resolve_name(code, "Model.train")
        assert result is not None
        assert result.name == "Model.train"
        assert result.location_type == "method"

    def test_resolve_method_partial(self) -> None:
        """Test resolving a method using just the method name."""
        code = """
class Model:
    def train(self, data):
        pass
"""
        result = resolve_name(code, "train")
        assert result is not None
        assert result.name == "Model.train"

    def test_resolve_with_type_filter(self) -> None:
        """Test resolving with location_type filter."""
        code = """
def train():
    pass

class train:
    pass
"""
        # Should find the function when filtering by function type
        result = resolve_name(code, "train", location_type="function")
        assert result is not None
        assert result.location_type == "function"

    def test_resolve_with_near_line(self) -> None:
        """Test using near_line to disambiguate duplicates."""
        code = """
def process():
    pass

def other():
    pass

def process():  # Duplicate name
    pass
"""
        # Without near_line, gets first match
        result1 = resolve_name(code, "process")
        assert result1 is not None
        assert result1.start_line == 2

        # With near_line closer to second definition
        result2 = resolve_name(code, "process", near_line=8)
        assert result2 is not None
        assert result2.start_line == 8

    def test_resolve_module_level(self) -> None:
        """Test resolving module-level code."""
        code = """
import numpy as np
x = np.array([1, 2, 3])
"""
        result = resolve_name(code, "anything", location_type="module", near_line=2)
        assert result is not None
        assert result.location_type == "module"
        assert result.name == "<module>"

    def test_resolve_not_found(self) -> None:
        """Test that non-existent name returns None."""
        code = """
def existing():
    pass
"""
        result = resolve_name(code, "nonexistent")
        assert result is None


class TestResolveNameWithFallback:
    """Tests for resolve_name_with_fallback function."""

    def test_successful_resolution(self) -> None:
        """Test that AST resolution works when possible."""
        code = """
def train_model():
    pass
"""
        result = resolve_name_with_fallback(code, "train_model")
        assert result is not None
        assert result.name == "train_model"
        assert result.start_line == 2

    def test_fallback_to_near_line(self) -> None:
        """Test fallback to near_line when AST resolution fails."""
        code = """
# Some code
x = 1
y = 2
z = 3
"""
        result = resolve_name_with_fallback(code, "nonexistent", near_line=3)
        assert result is not None
        # Should return context around line 3
        assert result.start_line == 1  # max(1, 3-3)
        assert result.end_line == 5  # min(5, 3+3)
        assert result.name == "nonexistent"

    def test_fallback_no_name_with_near_line(self) -> None:
        """Test fallback when name is None but near_line is provided."""
        code = """
x = 1
y = 2
"""
        result = resolve_name_with_fallback(code, None, near_line=2)
        assert result is not None
        assert result.name == "<unknown>"

    def test_no_fallback_possible(self) -> None:
        """Test that None is returned when neither AST nor fallback works."""
        code = "def f(): pass"
        result = resolve_name_with_fallback(code, "nonexistent", near_line=None)
        assert result is None


class TestResolvedLocation:
    """Tests for ResolvedLocation dataclass."""

    def test_lines_property(self) -> None:
        """Test that lines property returns full range."""
        loc = ResolvedLocation(
            name="test",
            location_type="function",
            start_line=5,
            end_line=10,
            snippet="...",
        )
        assert loc.lines == [5, 6, 7, 8, 9, 10]

    def test_single_line(self) -> None:
        """Test lines for single-line definition."""
        loc = ResolvedLocation(
            name="test",
            location_type="function",
            start_line=5,
            end_line=5,
            snippet="...",
        )
        assert loc.lines == [5]
