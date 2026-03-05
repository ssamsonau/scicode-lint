#!/bin/bash
# Start vLLM server locally
# Run with: bash scripts/start_vllm.sh [OPTIONS] [MODEL] [PORT] [MAX_LEN] [GPU_MEM]
#
# Options:
#   --restart, --force    Kill any running vLLM server before starting
#
# Examples:
#   bash scripts/start_vllm.sh
#   bash scripts/start_vllm.sh --restart
#   bash scripts/start_vllm.sh "meta-llama/Llama-3.1-8B-Instruct"
#   bash scripts/start_vllm.sh --restart "RedHatAI/gemma-3-12b-it-FP8-dynamic" 5001 8000 0.9
#
# For background mode:
#   nohup bash scripts/start_vllm.sh > /tmp/vllm.log 2>&1 &
#
# Requirements:
#   pip install scicode-lint[vllm-server]
#
# Note: This configures the vLLM SERVER (what model it loads).
#       config.toml configures the CLIENT (what it expects - auto-detected by default).

set -e

# Parse flags
RESTART=false
while [[ "$1" == --* ]]; do
    case "$1" in
        --restart|--force)
            RESTART=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "✗ vLLM not found."
    echo "  Install with: pip install scicode-lint[vllm-server]"
    exit 1
fi

# Check vLLM version (require 0.16+)
VLLM_VERSION=$(vllm --version 2>&1 | grep -oP 'vllm \K[0-9.]+' | head -1)
if [ -n "$VLLM_VERSION" ]; then
    MAJOR=$(echo "$VLLM_VERSION" | cut -d. -f1)
    MINOR=$(echo "$VLLM_VERSION" | cut -d. -f2)

    if [ "$MAJOR" -eq 0 ] && [ "$MINOR" -lt 16 ]; then
        echo "✗ vLLM version $VLLM_VERSION is too old (requires 0.16+)"
        echo "  Upgrade with: pip install --upgrade vllm"
        exit 1
    fi
    echo "✓ vLLM version $VLLM_VERSION"
fi

# Auto-detect GPU capabilities and set defaults if not specified
if [ -z "$3" ] || [ -z "$4" ]; then
    # Detect compute capability for FP8 support
    COMPUTE_CAP=""
    NATIVE_FP8=false
    GPU_AVAILABLE=false

    if command -v nvidia-smi &> /dev/null; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$COMPUTE_CAP" ]; then
            GPU_AVAILABLE=true
            # Convert to comparable integer (e.g., "8.9" -> 89, "8.0" -> 80)
            COMPUTE_CAP_MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)
            COMPUTE_CAP_MINOR=$(echo "$COMPUTE_CAP" | cut -d. -f2)
            COMPUTE_CAP_INT=$((COMPUTE_CAP_MAJOR * 10 + COMPUTE_CAP_MINOR))

            # Native FP8 support requires compute capability >= 8.9 (Ada Lovelace, Hopper)
            if [ "$COMPUTE_CAP_INT" -ge 89 ]; then
                NATIVE_FP8=true
            fi
        fi
    fi

    # Allow override for testing (e.g., SCICODE_VRAM_MB=16384 for simulating 16GB)
    if [ -n "$SCICODE_VRAM_MB" ]; then
        VRAM_MB="$SCICODE_VRAM_MB"
        echo "⚙️  Using simulated VRAM: ${VRAM_MB}MB (for testing)"
    else
        VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    fi

    if [ -n "$VRAM_MB" ]; then
        # Round up to avoid misleading "19GB" for 20GB cards
        VRAM_GB=$(( (VRAM_MB + 512) / 1024 ))

        # Enforce minimum 20GB VRAM
        if [ $VRAM_MB -lt 19500 ]; then
            echo "✗ ERROR: Detected ${VRAM_GB}GB VRAM. Minimum requirement: 20GB VRAM"
            echo ""
            echo "scicode-lint requires 20GB+ VRAM with native FP8 support."
            echo "Supported GPUs:"
            echo "  • Consumer: RTX 4090 (24GB)"
            echo "  • Workstation: RTX 4000 Ada (20GB), RTX 5000 Ada (32GB)"
            echo "  • HPC/Cloud inference: L4 (24GB), L40 (48GB), A10 (24GB)"
            echo ""
            echo "Cloud GPU pricing: L4 ~\$0.39/hr, A10 ~\$0.24/hr"
            echo "See INSTALLATION.md for deployment options."
            exit 1
        fi

        # Enforce native FP8 support (compute capability >= 8.9)
        if [ "$NATIVE_FP8" = false ]; then
            echo "✗ ERROR: GPU lacks native FP8 support"
            echo ""
            echo "Detected compute capability: ${COMPUTE_CAP:-unknown}"
            echo "Required: compute capability >= 8.9"
            echo ""
            echo "Your GPU (likely A100, V100, or RTX 30-series) does not support"
            echo "native FP8 operations. scicode-lint requires modern GPUs with"
            echo "native FP8 tensor cores for reliable performance."
            echo ""
            echo "Supported GPUs (compute cap >= 8.9):"
            echo "  • RTX 40-series (4090, 4000 Ada, 5000 Ada)"
            echo "  • L4, L40, A10"
            echo ""
            echo "Cloud alternatives: L4 ~\$0.39/hr, upgrade from A100/V100"
            exit 1
        fi

        # Standard configuration: 16K context (standardized)
        # Covers 90-95th percentile based on 10M+ repo analysis
        # Median: 258 lines, Mean: 879 lines, 90th: ~1,500 lines
        # Paged attention means no waste on smaller files
        DEFAULT_MAX_LEN=16000
        DEFAULT_GPU_MEM=0.90
        # Only model that reliably fits in 20GB VRAM with good structured output
        DEFAULT_MODEL="RedHatAI/gemma-3-12b-it-FP8-dynamic"

        echo "✓ Detected ${VRAM_GB}GB VRAM (compute capability: ${COMPUTE_CAP:-unknown})"
        echo "  → Gemma 3 12B FP8 (optimized for 20GB VRAM)"
        echo "     16K context, ~1,500 line files, native FP8 support"
    else
        # No GPU detected
        echo "✗ ERROR: GPU not detected"
        echo ""
        echo "scicode-lint requires a GPU with:"
        echo "  • 20GB+ VRAM"
        echo "  • Native FP8 support (compute capability >= 8.9)"
        echo ""
        echo "CPU-only mode is not supported."
        echo "See INSTALLATION.md for cloud GPU options (L4 ~\$0.39/hr)."
        exit 1
    fi
else
    # User specified custom max_len and gpu_mem - use default FP8 model
    DEFAULT_MAX_LEN=16000
    DEFAULT_GPU_MEM=0.9
    DEFAULT_MODEL="RedHatAI/gemma-3-12b-it-FP8-dynamic"
fi

# Configuration (with optional parameters)
# If user didn't specify model, use hardware-detected default
MODEL="${1:-${DEFAULT_MODEL:-RedHatAI/gemma-3-12b-it-FP8-dynamic}}"
PORT="${2:-5001}"
MAX_LEN="${3:-$DEFAULT_MAX_LEN}"
GPU_MEM="${4:-$DEFAULT_GPU_MEM}"

# Set short model name - Gemma 3 12B FP8 is the standard model
if [[ "$MODEL" == *"gemma"* ]]; then
    SERVED_NAME="gemma-3-12b-fp8"
else
    # Fallback for any other model (not recommended - may OOM)
    SERVED_NAME=$(basename "$MODEL" | sed 's/-FP8.*//' | tr '[:upper:]' '[:lower:]')
    echo "⚠️  WARNING: Non-standard model. May not fit in 20GB VRAM."
fi

# Handle --restart flag
if [ "$RESTART" = true ]; then
    if pgrep -f "vllm serve" > /dev/null; then
        echo "🔄 Stopping existing vLLM server..."
        pkill -f "vllm serve"
        sleep 3
        echo "✓ Stopped"
    else
        echo "ℹ  No vLLM server running"
    fi
fi

# Check if vLLM is already running on this port
if curl -s "http://localhost:${PORT}/v1/models" &> /dev/null; then
    echo "✓ vLLM server already running on port ${PORT}"
    echo ""
    echo "Current model:"
    curl -s "http://localhost:${PORT}/v1/models" | python3 -c "import sys, json; data=json.load(sys.stdin); print('  ' + data['data'][0]['id'])" 2>/dev/null || echo "  (unable to detect)"
    echo ""
    echo "To restart with different settings, use:"
    echo "  bash $0 --restart [MODEL] [PORT] [MAX_LEN] [GPU_MEM]"
    exit 0
fi

echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  Served as: $SERVED_NAME"
echo "  Port: $PORT"
echo "  Max length: $MAX_LEN tokens"
echo "  GPU memory: ${GPU_MEM}%"
echo ""
echo "Model will download automatically on first run (~8GB)"
echo ""

vllm serve \
    --host 0.0.0.0 \
    --port $PORT \
    --model $MODEL \
    --served-model-name $SERVED_NAME \
    --trust-remote-code \
    --gpu-memory-utilization $GPU_MEM \
    --max-model-len $MAX_LEN
