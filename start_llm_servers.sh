#!/bin/bash
# Standalone script to run LLM servers in the background
# Run this ONCE before starting Unity - it will keep running until you stop it

cd "$(dirname "$0")"

echo "Starting LLM servers..."
echo "This will run in the foreground. Press Ctrl+C to stop."
echo ""

# Run with Python from virtual environment
./ACRLPython/acrl/bin/python -u ACRLPython/LLMcommunication/RunAnalyzer.py --model gemma3 --interval 2.0
