#!/bin/bash
# Run RAG system tests
# Usage: ./run_rag_tests.sh

cd "$(dirname "$0")/.."

echo "==========================================="
echo "Running RAG System Tests"
echo "==========================================="
echo ""

# Run all RAG tests
./acrl/bin/pytest Tests/TestRAG*.py -v --tb=short

echo ""
echo "==========================================="
echo "Test Run Complete"
echo "==========================================="
