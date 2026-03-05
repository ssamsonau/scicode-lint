# VRAM Testing Guide

This document explains how to test scicode-lint with different VRAM configurations without needing multiple GPUs.

## Overview

scicode-lint auto-detects VRAM and adjusts settings accordingly.

## Testing with Simulated VRAM

Run tests with `pytest tests/test_vllm.py -v` or `bash tests/test_vllm_bash.sh`.

## Manual Testing with Simulated VRAM

Use `SCICODE_VRAM_MB` environment variable to simulate different VRAM amounts.

## Test Coverage

Includes server lifecycle, VRAM auto-detection, and integration tests.

## VRAM Thresholds

System uses MB-level precision for accurate detection across GPU models.

## Adding New VRAM Configs

Update detection logic, tests, and documentation when adding new VRAM tiers.
