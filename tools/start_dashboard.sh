#!/bin/bash
# Start vLLM monitoring dashboard
# Usage: ./tools/start_dashboard.sh [--url http://localhost:5001]

pkill -f "streamlit run.*vllm_dashboard" 2>/dev/null
exec streamlit run "$(dirname "$0")/vllm_dashboard.py" --server.headless true -- "$@"
