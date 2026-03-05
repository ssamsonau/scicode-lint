# Pydantic Migration for Validators

## Summary

Migrated `evals/validators.py` from `dataclasses` to Pydantic for better validation, cleaner code, and improved error handling.

**Date**: 2026-03-03

---

## Changes Made

### Before (dataclass)

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class Location:
    type: Literal["function", "class", "method", "module"]
    name: str
    snippet: str

@dataclass
class ActualFinding:
    id: str
    category: str
    severity: str
    location: Location
    issue: str
    explanation: str
    suggestion: str
    confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActualFinding":
        """Manual parsing - error-prone!"""
        loc_data = data["location"]  # KeyError if missing!
        location = Location(
            type=loc_data["type"],
            name=loc_data["name"],
            snippet=loc_data["snippet"]
        )
        return cls(
            id=data["id"],
            category=data["category"],
            severity=data["severity"],
            location=location,
            issue=data["issue"],
            explanation=data["explanation"],
            suggestion=data["suggestion"],
            confidence=data["confidence"],
        )
```

**Problems:**
- ❌ No validation (missing fields = KeyError crash)
- ❌ 20+ lines of boilerplate `from_dict` code
- ❌ No type coercion
- ❌ No range validation (confidence could be -1 or 100)
- ❌ Poor error messages

### After (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Literal

class Location(BaseModel):
    type: Literal["function", "class", "method", "module"]
    name: str
    snippet: str

class ActualFinding(BaseModel):
    id: str
    category: str
    severity: str
    location: Location  # Auto-parsed from nested dict!
    issue: str
    explanation: str
    suggestion: str
    confidence: float = Field(ge=0.0, le=1.0)  # Validated!

# Usage - ONE LINE!
finding = ActualFinding.model_validate(data)
```

**Benefits:**
- ✅ **Automatic validation** with helpful errors
- ✅ **Zero boilerplate** - removed all `from_dict` methods
- ✅ **Nested models** auto-parsed
- ✅ **Type coercion** (str → float, etc.)
- ✅ **Field validation** (ranges, patterns, custom validators)
- ✅ **Better error messages** with field paths

---

## Code Reduction

**Before:**
- `Location.from_dict`: 8 lines
- `ExpectedFinding.from_dict`: 11 lines
- `ActualFinding.from_dict`: 15 lines
- **Total**: 34 lines of boilerplate

**After:**
- **Total**: 0 lines (Pydantic handles it automatically!)

**Reduction**: -34 lines (-100% of boilerplate)

---

## Validation Examples

### 1. Missing Required Field

**Before:**
```python
# KeyError: 'severity'
finding = ActualFinding.from_dict({
    "id": "ml-001",
    "category": "ai-training",
    # Missing severity!
    ...
})
```

**After:**
```python
# ValidationError with clear message:
# "Field required [type=missing, input_value=..., input_type=dict]"
finding = ActualFinding.model_validate({
    "id": "ml-001",
    "category": "ai-training",
    # Missing severity!
    ...
})
```

### 2. Invalid Confidence Range

**Before:**
```python
# No validation - accepts invalid value!
finding = ActualFinding.from_dict({
    ...
    "confidence": 1.5  # Invalid but accepted!
})
```

**After:**
```python
# ValidationError: "Input should be less than or equal to 1"
finding = ActualFinding.model_validate({
    ...
    "confidence": 1.5  # Caught by Field(ge=0.0, le=1.0)
})
```

### 3. Invalid Location Type

**Before:**
```python
# No validation - accepts any string!
location = Location(
    type="invalid_type",  # Should be function/class/method/module
    name="test",
    snippet="x = 1"
)
```

**After:**
```python
# ValidationError: "Input should be 'function', 'class', 'method' or 'module'"
location = Location(
    type="invalid_type",  # Caught by Literal type!
    name="test",
    snippet="x = 1"
)
```

### 4. Nested Object Parsing

**Before:**
```python
# Manual nested object creation
loc_data = data["location"]
location = Location(
    type=loc_data["type"],
    name=loc_data["name"],
    snippet=loc_data["snippet"]
)
finding = ActualFinding(..., location=location)
```

**After:**
```python
# Automatic nested parsing!
finding = ActualFinding.model_validate({
    ...
    "location": {  # Pydantic auto-converts to Location object
        "type": "function",
        "name": "test",
        "snippet": "x = 1"
    }
})
```

---

## Files Modified

### 1. `evals/validators.py`
- Replaced `dataclass` with `BaseModel`
- Removed all `from_dict()` methods (34 lines)
- Added field validation with `Field()`
- Kept all business logic methods unchanged

### 2. `evals/run_eval.py`
- Changed `ActualFinding.from_dict()` → `ActualFinding.model_validate()`
- Changed `ExpectedFinding.from_dict()` → `ExpectedFinding.model_validate()`

---

## Testing

All validation tests pass:

```bash
✅ Valid finding created successfully
✅ Validation caught invalid confidence: ValidationError
✅ Validation caught missing field: ValidationError
✅ Validation caught invalid location type: ValidationError

🎉 All Pydantic validation tests passed!
```

Eval framework still works correctly:
```bash
python evals/run_eval.py --pattern ml-001-scaler-leakage
# ✅ Completes successfully with Pydantic models
```

---

## Benefits Summary

| Aspect | dataclass | Pydantic | Improvement |
|--------|-----------|----------|-------------|
| **Boilerplate code** | 34 lines | 0 lines | -100% |
| **Validation** | None | Automatic | ✅ |
| **Error messages** | KeyError | Clear ValidationError | ✅ |
| **Type coercion** | Manual | Automatic | ✅ |
| **Nested parsing** | Manual | Automatic | ✅ |
| **Field constraints** | None | Built-in (ge, le, etc.) | ✅ |
| **JSON schema** | Manual | Auto-generated | ✅ |

---

## Future Enhancements

With Pydantic, we can easily add:

1. **Custom validators**:
   ```python
   from pydantic import field_validator

   @field_validator('snippet')
   def validate_snippet(cls, v):
       if len(v) < 5:
           raise ValueError('Snippet too short')
       return v
   ```

2. **JSON Schema export**:
   ```python
   schema = ActualFinding.model_json_schema()
   # Use for API documentation, OpenAPI specs, etc.
   ```

3. **Strict mode** for exact type checking:
   ```python
   class StrictFinding(BaseModel):
       model_config = ConfigDict(strict=True)
   ```

4. **Computed fields**:
   ```python
   @computed_field
   @property
   def severity_level(self) -> int:
       return {"low": 1, "medium": 2, "high": 3, "critical": 4}[self.severity]
   ```

---

## Migration Checklist

- ✅ Migrate `Location` to Pydantic
- ✅ Migrate `ExpectedFinding` to Pydantic
- ✅ Migrate `ActualFinding` to Pydantic
- ✅ Migrate `ValidationResult` to Pydantic
- ✅ Remove all `from_dict()` methods
- ✅ Update `run_eval.py` to use `model_validate()`
- ✅ Add field validation (confidence range)
- ✅ Test validation with invalid data
- ✅ Test eval framework end-to-end
- ✅ Document migration

**Status**: ✅ Complete
