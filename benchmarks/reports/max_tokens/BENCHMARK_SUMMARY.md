# Max Output Tokens Experiment

## Metadata

| Field | Value |
|-------|-------|
| **Run Date** | 2026-03-09 |
| **Package Version** | v0.1.4 |
| **Model** | RedHatAI/Qwen3-8B-FP8-dynamic |
| **GPU** | NVIDIA RTX 4000 Ada Generation (20GB VRAM) |
| **Driver Version** | 581.80 |
| **Results File** | `comparison_20260309_112324.json` |

## Experiment Design

Tested 7 different `max_completion_tokens` values to find the optimal tradeoff between detection accuracy and execution time. Each configuration ran the full eval suite (64 patterns, 401 tests).

## Results

| Tokens | Overall Acc | Positive Acc | Negative Acc | Time (s) | Notes |
|--------|-------------|--------------|--------------|----------|-------|
| 16384 | 93.02% | **92.19%** | 93.37% | **307.9** | Fastest at high accuracy |
| 8192 | 92.52% | 90.10% | 94.90% | 320.2 | Previous default |
| 6144 | 92.27% | 91.67% | 92.86% | 392.8 | |
| **4096** | **93.27%** | 91.15% | **95.41%** | 381.0 | **Current default** - highest accuracy |
| 2048 | 91.27% | 90.63% | 91.84% | 320.9 | |
| 1024 | 85.41% | 76.04% | 94.39% | 286.7 | Positive accuracy drops |
| 512 | 54.61% | 42.19% | 64.80% | 835.3 | Unusable - reasoning truncated |

## Key Findings

1. **Sweet spot: 4096-16384 tokens** - All values in this range achieve 92-93% accuracy
2. **Highest accuracy: 4096 tokens (93.27%)** - Slightly better than higher values
3. **Fastest execution: 16384 tokens (308s)** - 24% faster than 4096, near-best accuracy
4. **Sharp dropoff below 2048**: Model struggles to complete reasoning
5. **512 is pathological**: Slowest AND worst - truncated reasoning causes JSON parse failures and retries

## Observations

- **Positive accuracy is more sensitive** to token limits than negative accuracy
- At 1024 tokens, positive accuracy drops to 76% while negative stays at 94%
- Lower token limits cause more retries due to truncated thinking, paradoxically increasing total time (512 tokens took 835s)

## Recommendation

**Use 16384 tokens** as the new default:
- Near-best accuracy (93.02% vs 93.27% for 4096)
- Fastest execution (308s)
- Best positive accuracy (92.19%)
- Provides headroom for complex edge cases

Alternative: Keep 4096 if minimizing token generation matters (API cost optimization).
