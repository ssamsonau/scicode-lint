# Benchmarks

Performance benchmarking tools for scicode-lint with vLLM.

## Quick Start

```bash
# Main benchmark (~2 minutes) - RECOMMENDED
python benchmarks/benchmark.py

# Full system test (~30 minutes)
python benchmarks/benchmark_system.py
```

---

## Available Benchmarks

### 1. `benchmark.py` ⚡ **Main Benchmark**

**Purpose:** Quick test showing vLLM's concurrent batching advantage

**Runtime:** ~2 minutes

**What it tests:**
- Sequential: 5 patterns processed one at a time
- Concurrent: 5 patterns processed all at once
- Shows actual speedup from vLLM batching

**Example output:**
```
Sequential:  73.9s
Concurrent:  48.8s
Speedup:     1.5x
Time saved:  25.1s

Extrapolated for all 64 patterns:
  Sequential: ~650s
  Concurrent: ~429s
```

**When to use:**
- Quick verification that vLLM batching works
- After restarting vLLM
- Testing configuration changes
- Daily sanity checks

---

### 2. `benchmark_system.py` 📊 **Full System Benchmark**

**Purpose:** Comprehensive system performance test on real files

**Runtime:** ~30 minutes

**What it measures:**
- System information (CPU, memory, platform)
- LLM backend detection
- Full linting on 20 real Python files from patterns/
- Total time, average per file, throughput

**Output files:**
- `benchmark_results.json` - Machine-readable results
- `benchmark_results.txt` - Human-readable summary

**When to use:**
- Initial setup validation
- Hardware comparison
- Performance baseline measurements
- Sharing performance reports

**Example:**
```bash
python benchmarks/benchmark_system.py

# Results saved to:
cat benchmark_results.txt
```

---

## Requirements

### vLLM Running

Verify vLLM is running at `http://localhost:5001` before benchmarking.

### Dependencies

```bash
pip install -e .
```

---

## Understanding Results

### Good Performance

✅ **Sequential time:** 1-2s per pattern
✅ **Concurrent speedup:** 3-8x for multiple patterns
✅ **Throughput:** 15-20 files/minute on full system test

### Performance Issues

⚠️ **Slow sequential:** 5+ seconds per pattern
- Check GPU utilization and model size
- Try smaller model or better GPU

⚠️ **Low speedup:** <2x for concurrent
- Check vLLM logs for errors
- Verify vLLM batching is enabled
- Restart vLLM

⚠️ **Timeouts:** Requests hang
- Check vLLM server logs
- Increase timeout in config
- Restart vLLM server

---

## How It Works

### Why Concurrent is Faster

**Sequential (one at a time):**
```
Pattern 1 → Process → Result (1.4s)
Pattern 2 → Process → Result (1.0s)
Pattern 3 → Process → Result (1.3s)
Total: 3.7s
```

**Concurrent (vLLM batching):**
```
Pattern 1 ┐
Pattern 2 ├─→ [vLLM Batch] → All Results
Pattern 3 ┘
Total: 1.6s (2.3x faster!)
```

**Why vLLM is faster:**
1. **Continuous batching** - Processes multiple requests together
2. **Prefix caching** - Shared code sent once, reused for all patterns
3. **GPU parallelism** - GPU processes multiple patterns simultaneously

---

## Concurrency Strategy

Both benchmarks use **no semaphore** (unlimited concurrency):

```python
# Send all requests at once - vLLM batches automatically
tasks = [llm.async_complete(...) for pattern in patterns]
results = await asyncio.gather(*tasks)
```

**Why no semaphore for vLLM?**
- vLLM has built-in continuous batching
- Prefix caching works best with all requests queued at once
- vLLM manages GPU resources internally
- Client-side limiting adds unnecessary bottlenecks

**For shared vLLM servers:**
- Use semaphore (custom limit) if needed to avoid overloading shared resources

See `docs_use_human/performance/CONCURRENCY_GUIDE.md` for detailed strategies.

---

## Troubleshooting

### "No LLM backend detected"

Start vLLM server: `bash src/scicode_lint/vllm/start_vllm.sh`

### Benchmarks hang or timeout

1. Verify vLLM is responding
2. Check vLLM logs for errors
3. Restart vLLM server if needed

### Inconsistent results

Run benchmark 2-3 times and average results for accuracy.

---

## Related Documentation

- [Benchmarking Guide](../docs_use_human/performance/BENCHMARKING.md) - User-facing guide
- [Concurrency Guide](../docs_use_human/performance/CONCURRENCY_GUIDE.md) - Detailed strategies for different backends
- [Installation](../INSTALLATION.md) - Setup instructions
