# LLMcommunication Test Suite

Comprehensive unit tests for the ACRLPython LLMcommunication package.

## Setup

### Install Test Dependencies

```bash
cd ACRLPython/LLMcommunication
pip install -r requirements-test.txt
```

### Install Main Dependencies

Make sure main dependencies are installed:

```bash
pip install numpy opencv-python ollama
```

## Running Tests

### Run All Tests

```bash
# From LLMcommunication directory
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
```

### Run Specific Test Files

```bash
# Test a specific module
pytest Tests/TestUnityProtocol.py -v

# Test configuration
pytest Tests/TestConfig.py -v

# Test object detection
pytest Tests/TestObjectDetector.py -v

# Test RAG system (all RAG tests)
pytest Tests/TestRAG*.py -v

# Test specific RAG component
pytest Tests/TestRAGVectorStore.py -v
pytest Tests/TestRAGEmbeddings.py -v
pytest Tests/TestRAGSystem.py -v
```

### Run RAG Tests

Run all RAG system tests using the helper script:

```bash
# From ACRLPython directory
./Tests/run_rag_tests.sh

# Or directly with pytest
./acrl/bin/pytest Tests/TestRAG*.py -v --tb=short
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
pytest tests/test_unity_protocol.py::TestImageMessageEncoding -v

# Run a specific test function
pytest tests/test_config.py::TestConfigConstants::test_network_config -v
```

### Run with Markers

```bash
# Run only fast tests (if markers are added)
pytest tests/ -m "not slow"

# Run only integration tests
pytest tests/ -m "integration"
```

## Test Organization

### Test Files

**Core Infrastructure:**
- `TestConfig.py` - Configuration constants and helpers
- `TestUnityProtocol.py` - Unity ↔ Python wire protocol
- `TestTcpServerBase.py` - TCP server base class

**Network Servers:**
- `TestStreamingServer.py` - Image streaming server and storage
- `TestResultsServer.py` - Results broadcasting server
- `TestDetectionServer.py` - Object detection server
- `TestStereoDetectionServer.py` - Stereo image server

**Vision & Detection:**
- `TestObjectDetector.py` - Color-based cube detection
- `TestAnalyzeImage.py` - Ollama vision processing
- `TestDepthEstimator.py` - Stereo depth estimation

**Orchestrators:**
- `TestRunAnalyzer.py` - LLM analyzer orchestrator
- `TestRunDetector.py` - Detection orchestrator
- `TestRunStereoDetector.py` - Stereo detection orchestrator

**RAG System (Semantic Search):**
- `TestRAGConfig.py` - RAG configuration and validation
- `TestRAGEmbeddings.py` - LM Studio embedding generation
- `TestRAGVectorStore.py` - Vector storage and similarity search
- `TestRAGIndexer.py` - Operation indexing from registry
- `TestRAGQueryEngine.py` - Semantic search and retrieval
- `TestRAGSystem.py` - RAG system integration tests

### Fixtures (Conftest.py)

Shared fixtures available to all tests:

**Network & Communication:**
- `mock_socket` - Mock socket object for network testing
- `server_config` - Test server configuration

**Image Processing:**
- `sample_image` - Sample RGB gradient image (640x480)
- `sample_red_cube_image` - Test image with red cube
- `sample_blue_cube_image` - Test image with blue cube
- `sample_stereo_pair` - Stereo image pair for depth testing

**LLM & Results:**
- `detection_result_dict` - Sample detection result
- `llm_result_dict` - Sample LLM analysis result
- `mock_lmstudio_client` - Mock LM Studio client for vision

**RAG System:**
- `mock_lmstudio_embeddings_client` - Mock LM Studio embeddings API
- `sample_operation` - Mock BasicOperation instance
- `mock_operation_registry` - Mock operation registry with sample ops
- `temp_vector_store_path` - Temporary path for vector store
- `cleanup_rag_singletons` - Reset RAG singleton instances

**Utilities:**
- `cleanup_singletons` - Resets singleton instances between tests
- `temp_output_dir` - Temporary directory for test output

## Test Coverage

### Current Coverage Status

**Core Infrastructure** (High Priority):
- ✅ `config.py` - 100% coverage
- ✅ `core/UnityProtocol.py` - 95%+ coverage
- ✅ `core/TCPServerBase.py` - 90%+ coverage

**Servers**:
- ✅ `StreamingServer.py` - 85%+ coverage
- ✅ `ResultsServer.py` - 85%+ coverage
- ⚠️ `DetectionServer.py` - Partial (needs completion)
- ⚠️ `StereoDetectionServer.py` - Partial (needs completion)

**Detection & Processing**:
- ✅ `ObjectDetector.py` - 85%+ coverage
- ✅ `AnalyzeImage.py` - 85%+ coverage
- ⚠️ `DepthEstimator.py` - Partial (needs completion)

**Orchestration**:
- ⚠️ `RunAnalyzer.py` - Partial (needs completion)
- ⚠️ `RunDetector.py` - Partial (needs completion)
- ⚠️ `RunStereoDetector.py` - Partial (needs completion)

**RAG System** (NEW):
- ✅ `rag/Config.py` - 95%+ coverage
- ✅ `rag/Embeddings.py` - 90%+ coverage (with mocks)
- ✅ `rag/VectorStore.py` - 95%+ coverage
- ✅ `rag/Indexer.py` - 85%+ coverage
- ✅ `rag/QueryEngine.py` - 85%+ coverage
- ✅ `rag/__init__.py` (RAGSystem) - 90%+ coverage

**Operations System** (NEW):
- ✅ `operations/Base.py` - 95%+ coverage
- ✅ `operations/MoveOperations.py` - 90%+ coverage
- ✅ `operations/Registry.py` - 90%+ coverage

### Generating Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/ --cov=. --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows

# Generate terminal coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

## Writing New Tests

### Test File Template

```python
#!/usr/bin/env python3
"""
Unit tests for YourModule.py

Brief description of what's being tested
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from YourModule import YourClass


class TestYourClassInitialization:
    """Test initialization"""

    def test_initialization(self):
        """Test basic initialization"""
        obj = YourClass()
        assert obj is not None


class TestYourClassFunctionality:
    """Test main functionality"""

    def test_some_method(self):
        """Test some_method"""
        obj = YourClass()
        result = obj.some_method()
        assert result == expected_value
```

### Best Practices

1. **Use descriptive test names**: `test_encode_valid_message` not `test_1`
2. **One assertion per test** (when possible): Easier to debug failures
3. **Use fixtures**: Reduce code duplication and improve readability
4. **Mock external dependencies**: Use `unittest.mock` for network, file I/O, etc.
5. **Test edge cases**: Empty inputs, None values, boundary conditions
6. **Test error handling**: Verify exceptions are raised correctly
7. **Clean up after tests**: Use `cleanup_singletons` fixture for stateful code
8. **Add docstrings**: Explain what each test verifies

### Testing Patterns

#### Testing with Mocks

```python
from unittest.mock import Mock, patch

@patch('module.external_function')
def test_with_mock(mock_external):
    mock_external.return_value = "mocked_value"
    result = my_function()
    assert result == "expected"
    mock_external.assert_called_once()
```

#### Testing Thread Safety

```python
import threading

def test_thread_safety():
    errors = []

    def worker():
        try:
            # Do concurrent operations
            pass
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
```

#### Testing Exceptions

```python
def test_raises_exception():
    with pytest.raises(ValueError, match="Expected error message"):
        function_that_should_raise()
```

## Continuous Integration

### Running Tests in CI

```yaml
# Example GitHub Actions workflow
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt
          pip install numpy opencv-python
      - name: Run tests
        run: pytest tests/ --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Make sure parent directory is in path
export PYTHONPATH="${PYTHONPATH}:/path/to/ACRLPython/LLMcommunication"
```

**Singleton State Issues**:
- Use `cleanup_singletons` fixture
- Run tests with `pytest -v` to see which test is failing

**OpenCV Errors**:
```bash
# Install OpenCV headless version for CI
pip install opencv-python-headless
```

**Threading Timeouts**:
- Increase timeout values in test config
- Use shorter timeouts in `ServerConfig` for tests

**Ollama Connection Errors**:
- Tests mock Ollama by default
- No actual Ollama server needed

**LM Studio Connection Errors (RAG Tests)**:
- RAG tests mock LM Studio embeddings API by default
- No actual LM Studio server needed for unit tests
- Use `mock_lmstudio_embeddings_client` fixture
- Integration tests with real LM Studio should be marked separately

### Debug Mode

```bash
# Run with debug output
pytest tests/ -v --log-cli-level=DEBUG

# Drop into debugger on failure
pytest tests/ --pdb

# Stop on first failure
pytest tests/ -x
```

## Contributing

When adding new features to the codebase:

1. **Write tests first** (TDD approach)
2. **Aim for 80%+ coverage** for new code
3. **Run full test suite** before committing
4. **Update this README** if adding new test categories

### Test Quality Checklist

- [ ] Tests pass consistently
- [ ] Edge cases covered
- [ ] Error cases tested
- [ ] Thread safety verified (if applicable)
- [ ] Mocks used for external dependencies
- [ ] Docstrings added to test functions
- [ ] No hardcoded values (use config or fixtures)
- [ ] Cleanup performed (singletons, temp files)

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Testing best practices](https://docs.python-guide.org/writing/tests/)
- Project documentation: `../CLAUDE.md`
