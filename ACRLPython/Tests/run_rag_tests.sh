#!/bin/bash
# Run RAG system tests
# Usage: ./run_rag_tests.sh [--integration]

cd "$(dirname "$0")/.."

echo "==========================================="
echo "Running RAG System Tests"
echo "==========================================="
echo ""

# Check if integration tests should be run
if [ "$1" = "--integration" ]; then
    echo "Running all tests (including integration tests)..."
    ./acrl/bin/pytest Tests/TestRAG*.py -v --tb=short
else
    echo "Running unit tests only (use --integration for all tests)..."
    ./acrl/bin/pytest Tests/TestRAG*.py -v --tb=short -m "not integration"
fi

echo ""
echo "==========================================="
echo "Test Run Complete"
echo "==========================================="
