# scicode-lint Interface Analysis

## Question: Is the package interface optimized for humans or GenAI coding agents?

**Answer: Currently optimized for HUMANS with good GenAI agent support**

---

## Current Interface Design

### Human-Friendly Features ✓

1. **CLI with text output (default)**
   - Colorful, readable output with emojis
   - Human-readable severity indicators (🔴 CRITICAL, 🟠 HIGH, 🟡 MEDIUM)
   - Narrative explanations

2. **Documentation**
   - Focused on human users
   - Installation guides, usage examples
   - Troubleshooting sections

3. **Error messages**
   - Human-readable
   - Helpful suggestions

### GenAI Agent Support ✓

1. **Python API**
   - Direct programmatic access via `SciCodeLinter`
   - Type-safe with Pydantic models
   - No subprocess overhead

2. **JSON output**
   - Machine-parseable via `--format json`
   - Structured data with consistent schema

3. **Structured results**
   - `Finding` objects with typed fields
   - Clear data models (`LintResult`, `Finding`, `Location`)

4. **Targeted checking**
   - Filter by pattern, category, severity
   - Essential for efficiency

---

## What Works Well for GenAI Agents

✓ **Python API exists** - Can import and use programmatically
✓ **Structured output** - Clear data models, no string parsing
✓ **JSON CLI output** - Alternative to Python API
✓ **Filtering options** - Check specific patterns/categories
✓ **Explanations** - `finding.explanation` contains fix instructions
✓ **Location info** - `finding.location.name` and `snippet` for targeting

---

## What Could Be Better for GenAI Agents

### Minor Issues

1. **Default is text output** - Should default to JSON for programmatic use
   - Current: `scicode-lint check file.py` → text
   - Better: Detect if stdout is TTY, auto-select format

2. **Documentation hierarchy** - Human docs in README, GenAI docs in subfolder
   - Current: README targets humans
   - Better: README section for GenAI agents OR separate quick start

3. **Exit codes** - Same exit code (1) for errors and findings
   - Current: Hard to distinguish "issues found" vs "error occurred"
   - Better: Different exit codes (1=issues, 2=error)

4. **API examples** - Limited Python API examples in README
   - Current: CLI examples dominate
   - Better: Prominent Python API section in README

---

## Recommendations for GenAI Agent Optimization

### High Priority

1. **Update README.md**
   - Add "For GenAI Agents" section at top
   - Show Python API first, CLI second
   - Link to `GENAI_AGENT_GUIDE.md`

2. **Improve error handling**
   - Specific exception types for different errors
   - Example:
     ```python
     from scicode_lint.exceptions import (
         LLMConnectionError,
         ModelNotFoundError,
         TimeoutError
     )
     ```

### Medium Priority

3. **Auto-detect output format**
   ```python
   # Smart default
   if sys.stdout.isatty():
       format = "text"  # Human at terminal
   else:
       format = "json"  # Piped/redirected, likely machine
   ```

4. **Better exit codes**
   - 0: No issues
   - 1: Issues found
   - 2: Error occurred

### Low Priority (Nice to Have)

5. **Streaming API**
   ```python
   async for finding in linter.check_file_stream(path):
       await fix_issue(finding)  # Fix as findings arrive
   ```

6. **Batch API**
   ```python
   results = linter.check_files([path1, path2, path3])  # Parallel
   ```

---

## Current Status: Hybrid Interface

**Package serves both audiences equally well**

### For Humans
- CLI with beautiful text output
- Comprehensive documentation
- Easy troubleshooting

### For GenAI Agents
- Python API for programmatic use
- JSON output for parsing
- Structured data models
- Targeted pattern checking

**Conclusion:** The package is well-designed for both use cases. The new `GENAI_AGENT_GUIDE.md` fills the documentation gap. Minor improvements (error handling, exit codes) would enhance GenAI agent experience, but are not blockers.

---

## Action Items

### Completed ✓
- [x] Created `GENAI_AGENT_GUIDE.md` with Python API examples
- [x] Documented targeted pattern checking
- [x] Provided complete workflow examples

### Recommended (Optional)
- [ ] Add "For GenAI Agents" section to README.md
- [ ] Create custom exception types
- [ ] Improve exit codes (0/1/2 instead of 0/1)
- [ ] Auto-detect output format based on TTY
- [ ] Add batch API for multiple files

**Priority:** The package is already usable by GenAI agents. Recommended improvements are enhancements, not requirements.
