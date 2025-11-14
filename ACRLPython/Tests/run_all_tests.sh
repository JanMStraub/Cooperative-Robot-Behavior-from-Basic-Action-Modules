#!/bin/bash
# Run complete system tests
# Usage: ./run_all_tests.sh

cd "$(dirname "$0")/.."

echo "==========================================="
echo "Running Complete System Tests"
echo "==========================================="
echo ""

echo "Running all tests..."
    ./acrl/bin/pytest Tests/Test*.py -v --tb=short

echo ""
echo "==========================================="
echo "Test Run Complete"
echo "==========================================="
