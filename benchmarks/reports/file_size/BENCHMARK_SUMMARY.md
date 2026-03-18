# File Size Benchmark Report

**Date:** 2026-03-18
**Mode:** single pattern (ml-001)
**Runs per file:** 3

## Results

| File | Lines | Findings | Mean (s) | Min (s) | Max (s) | Spread (s) |
|------|------:|--------:|---------:|--------:|--------:|-----------:|
| small_30_lines.py | 32 | 1 | 14.68 | 13.51 | 15.75 | 2.24 |
| medium_200_lines.py | 211 | -1 | 23.53 | 16.6 | 31.97 | 15.37 |
| large_500_lines.py | 496 | -1 | 29.93 | 22.43 | 42.66 | 20.23 |
| xlarge_1000_lines.py | 1001 | -1 | 43.95 | 34.81 | 52.88 | 18.07 |

## Summary

- **Smallest file:** 32 lines → 14.68s
- **Largest file:** 1001 lines → 43.95s
- **Line count ratio:** 31x
- **Time ratio:** 3.0x
- **Scaling:** Sub-linear (3.0x time for 31x lines)