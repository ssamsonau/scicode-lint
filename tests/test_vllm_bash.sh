#!/bin/bash
# Integration tests for vLLM bash script with simulated VRAM
# Tests the 20GB minimum requirement and 16K standard context

set -e

SCRIPT="src/scicode_lint/vllm/start_vllm.sh"
FAILED=0

echo "Testing vLLM bash script with different VRAM configurations..."
echo ""

# Test 1: 20GB VRAM (should succeed with 16K context, 0.90 GPU mem, FP8 model)
echo "Test 1: 20GB VRAM detection"
output=$(SCICODE_VRAM_MB=20475 bash "$SCRIPT" 2>&1 || true)
if echo "$output" | grep -q "Detected.*GB VRAM" && echo "$output" | grep -q "16K"; then
    echo "✓ Correctly detected 20GB → FP8 model, 16K context"
else
    echo "✗ Failed to detect 20GB settings"
    echo "$output"
    FAILED=$((FAILED + 1))
fi

# Test 2: 16GB VRAM (should ERROR - below minimum)
echo "Test 2: 16GB VRAM detection (should error)"
output=$(SCICODE_VRAM_MB=16384 bash "$SCRIPT" 2>&1 || true)
if echo "$output" | grep -q "ERROR.*Minimum requirement: 20GB"; then
    echo "✓ Correctly rejected 16GB → Error (below minimum)"
else
    echo "✗ Failed to reject 16GB VRAM"
    echo "$output"
    FAILED=$((FAILED + 1))
fi

# Test 3: 12GB VRAM (should ERROR - below minimum)
echo "Test 3: 12GB VRAM detection (should error)"
output=$(SCICODE_VRAM_MB=12288 bash "$SCRIPT" 2>&1 || true)
if echo "$output" | grep -q "ERROR.*Minimum requirement: 20GB"; then
    echo "✓ Correctly rejected 12GB → Error (below minimum)"
else
    echo "✗ Failed to reject 12GB VRAM"
    echo "$output"
    FAILED=$((FAILED + 1))
fi

# Test 4: Boundary test - just below 20GB (19499MB → should ERROR)
echo "Test 4: Boundary test - 19499MB (just below 20GB)"
output=$(SCICODE_VRAM_MB=19499 bash "$SCRIPT" 2>&1 || true)
if echo "$output" | grep -q "ERROR.*Minimum requirement: 20GB"; then
    echo "✓ Correctly rejected 19499MB → Error (below minimum)"
else
    echo "✗ Failed boundary test at 19499MB"
    echo "$output"
    FAILED=$((FAILED + 1))
fi

# Test 5: Boundary test - at 20GB threshold (19500MB → should succeed)
echo "Test 5: Boundary test - 19500MB (at 20GB threshold)"
output=$(SCICODE_VRAM_MB=19500 bash "$SCRIPT" 2>&1 || true)
if echo "$output" | grep -q "Detected.*GB VRAM"; then
    echo "✓ Correctly handled 19500MB → FP8 model, 16K context"
else
    echo "✗ Failed boundary test at 19500MB"
    echo "$output"
    FAILED=$((FAILED + 1))
fi

# Test 6: 24GB VRAM (should succeed - above minimum)
echo "Test 6: 24GB VRAM detection"
output=$(SCICODE_VRAM_MB=24576 bash "$SCRIPT" 2>&1 || true)
if echo "$output" | grep -q "Detected.*GB VRAM" && echo "$output" | grep -q "16K"; then
    echo "✓ Correctly detected 24GB → FP8 model, 16K context"
else
    echo "✗ Failed to detect 24GB settings"
    echo "$output"
    FAILED=$((FAILED + 1))
fi

echo ""
echo "========================================="
if [ $FAILED -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ $FAILED test(s) failed"
    exit 1
fi
