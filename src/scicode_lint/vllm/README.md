# vLLM Utilities

Lightweight utilities for vLLM server lifecycle management and system information.

## Contents

### Python Module (`__init__.py`)

Programmatic server management for GenAI agents:
- `VLLMServer()` - Context manager
- `start_server()` / `stop_server()` - Manual control
- `get_gpu_info()` - GPU and VRAM information
- `get_server_info()` - vLLM server status
- `print_system_info()` - Print complete system status

See [../../docs_use_genai/VLLM_UTILITIES.md](../../docs_use_genai/VLLM_UTILITIES.md) for complete documentation.

### Bash Script (`start_vllm.sh`)

Manual server startup for humans:
```bash
# Start with defaults
bash src/scicode_lint/vllm/start_vllm.sh

# Start with custom model
bash src/scicode_lint/vllm/start_vllm.sh "meta-llama/Llama-3.1-8B-Instruct"

# Start with full config
bash src/scicode_lint/vllm/start_vllm.sh \
    "RedHatAI/gemma-3-12b-it-FP8-dynamic" \
    5001 \
    12000 \
    0.9
```

## Quick Examples

Use `print_system_info()` to check GPU and vLLM status, or `get_gpu_info()` for programmatic access to VRAM information.

## Files

- `__init__.py` - Python utilities module
- `start_vllm.sh` - Bash script for manual startup
- `README.md` - This file
