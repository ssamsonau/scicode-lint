# Concurrency Guide for vLLM

## Overview

scicode-lint always checks all patterns concurrently. vLLM handles batching and queuing internally - no configuration needed!

## How It Works

**All patterns checked concurrently:**
- All 44 patterns are sent to vLLM simultaneously
- vLLM's internal scheduler batches them efficiently
- Prefix caching works best with concurrent requests
- No client-side rate limiting or semaphores

**Implementation:**
```python
# Send all requests at once, let vLLM queue them
tasks = [llm.async_complete_structured(...) for pattern in patterns]
results = await asyncio.gather(*tasks)
```

## Performance

**Sequential vs Concurrent:**
- 44 patterns, sequential: ~132s
- 44 patterns, concurrent: ~15-20s
- **Speedup: 6-8x**

**Why concurrent is better:**
- ✅ vLLM automatically batches requests
- ✅ Prefix caching reuses common code across patterns
- ✅ Internal scheduler manages GPU resources
- ✅ No idle GPU time between requests

---

## Example: Large Codebase

```bash
# 100 files, 44 patterns each = 4,400 pattern checks
# Sequential: ~3.5 hours
# Concurrent (default): ~25 minutes
# Speedup: 8.4x
```

---

## Monitoring

Use vLLM metrics endpoint and GPU monitoring tools. Run with `-v` flag for detailed timing logs.
