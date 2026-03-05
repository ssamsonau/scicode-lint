# VRAM Requirements

scicode-lint requires **minimum 20GB VRAM** with **native FP8 support** (compute capability >= 8.9).

**Disk space:** Also requires **~15GB free disk space** for model weights, which are downloaded to `~/.cache/huggingface/` on first launch. See [INSTALLATION.md](../INSTALLATION.md#model-storage) for details.

## Quick Reference

**Requirements:**
- 20GB+ VRAM
- Native FP8 support (compute capability >= 8.9)

**Configuration:** 16K context (~1,500 line files), Gemma 3 12B FP8

## Setup

**Just run the script** - it automatically verifies your hardware meets requirements:

```bash
bash src/scicode_lint/vllm/start_vllm.sh
```

The script verifies:
1. **VRAM**: Minimum 20GB available
2. **Compute capability**: >= 8.9 (native FP8 support)
3. **GPU availability**: Rejects CPU-only systems

If requirements aren't met, the script exits with clear error messages and hardware recommendations.

## Configuration

**Model:** Gemma 3 12B FP8 (`RedHatAI/gemma-3-12b-it-FP8-dynamic`)

**Specs:**
- Model size: ~13.3GB
- Context: 16,000 tokens
- File coverage: ~1,500 lines (90-95th percentile)
- Quality: 75.21% (OpenLLM v1 average)
- VRAM usage: ~18GB total (90% utilization)

**Memory breakdown:**
- Model weights: 13.3GB
- Framework overhead: ~2GB
- KV cache (16K context): ~3GB
- Total: ~18GB

## Supported GPUs

**Native FP8 GPUs (compute capability >= 8.9):**
- **Consumer:** RTX 4090 (24GB)
- **Workstation:** RTX 4000 Ada (20GB), RTX 5000 Ada (32GB)
- **Cloud/HPC inference:** L4 (24GB), L40 (48GB), A10 (24GB)

## Why 20GB Minimum?

- FP8 model requires ~18GB for 16K context operation
- Model weights: 13.3GB
- Framework overhead: ~2GB
- KV cache: ~3GB
- Modern GPUs (2024+) standardize on 20GB+ for inference workloads

## Context Window Sizing

All configurations use **16K context** (standardized):

- Based on analysis of 10M+ GitHub repositories
- Covers **90-95%** of Python files in the wild
- Median file: 258 lines (~2,600 tokens)
- Mean file: 879 lines (~8,800 tokens)
- 90th percentile: ~1,500 lines (~15,000 tokens)

With prompt overhead (~500 tokens) and safety buffer (~400 tokens reserved), 16K context supports files up to **~1,500 lines**.

**Token allocation:**
- vLLM context window: 16,000 tokens (total)
- Reserved buffer: 400 tokens (output + safety margin)
- Maximum input: 15,600 tokens
  - System/detection prompts: ~500 tokens
  - Code content: ~15,100 tokens (~1,510 lines)

**vLLM paged attention:** Smaller files don't waste VRAM. Memory is allocated dynamically based on actual file size.

## Verification

Check your setup is working:

```bash
# 1. Verify requirements
bash src/scicode_lint/vllm/start_vllm.sh
# Will exit with error if hardware doesn't meet requirements

# 2. Check VRAM usage with nvidia-smi (should be ~18GB)
nvidia-smi

# 3. Test linting
scicode-lint check your_file.py
```

## Summary

**Requirements:**
- 20GB+ VRAM
- Native FP8 support (compute cap >= 8.9)
- Examples: RTX 4090, RTX 4000 Ada, L4, L40, A10

**Model:** Gemma 3 12B FP8 (75.21% OpenLLM v1 quality)
**Context:** 16K tokens (~1,500 line files)
