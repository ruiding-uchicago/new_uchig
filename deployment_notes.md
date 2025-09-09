# Deployment Notes for DigitalOcean

## Problem
The app has 10,000+ JSON training files (318MB) that cause build failures on DigitalOcean.

## Solution
1. **Keep torch dependencies** - Required for PFAS inference
2. **Use CPU-only torch** - Smaller size (100MB vs 700MB)
3. **Exclude training data** - Via .dockerignore
4. **Keep inference code** - PFAS model Python files are preserved

## What's excluded during build:
- `/pfas_screen_model/JSON_data/retrieved_substances_data_enhanced_json_files_2nd_attempt_11052024/` (10,000+ files)
- ZIP files in JSON_data
- Git files, cache, logs

## What's kept:
- All Python code including `inference_engine.py`
- Model weights (if any)
- torch, torch_geometric for inference
- All other dependencies

## If build still fails:
1. Increase build timeout in DigitalOcean settings
2. Use build machine with more memory (upgrade plan)
3. Consider using pre-built Docker image