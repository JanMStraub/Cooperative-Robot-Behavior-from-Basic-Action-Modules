# RAG System for Robot Operations

A semantic search and retrieval system for robot operations using LM Studio embeddings and cosine similarity search.

## Overview

This RAG (Retrieval-Augmented Generation) system enables natural language search over robot operations, allowing LLMs to discover and use the right operations for tasks through semantic similarity.

### Key Features

- **Semantic Search**: Natural language queries like "move robot to position" → finds `move_to_coordinate`
- **LM Studio Integration**: Uses local LM Studio for embeddings (no API costs, privacy-preserving)
- **Persistent Index**: Cached vector index for fast startup
- **Metadata Filtering**: Filter by category, complexity, or custom criteria
- **LLM-Ready Output**: Returns full operation context for task planning
- **TF-IDF Fallback**: Works even if LM Studio is unavailable

## Architecture

```
┌─────────────────┐
│ Operations      │
│ Registry        │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Indexer         │ ──> Generates embeddings via LM Studio
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Vector Store    │ ──> Numpy arrays + Pickle persistence
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Query Engine    │ ──> Cosine similarity search
└────────┬────────┘
         │
         v
┌─────────────────┐
│ RAGSystem API   │ ──> Simple interface for LLM use
└─────────────────┘
```

## Prerequisites

### LM Studio Setup

1. **Install LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai)

2. **Download an Embedding Model**:
   - Recommended: `nomic-ai/nomic-embed-text-v1.5-GGUF`
   - Alternative: `BAAI/bge-small-en-v1.5-GGUF`
   - In LM Studio: Search > "nomic embed" > Download

3. **Load the Model**:
   - Go to "Local Server" tab
   - Select your embedding model
   - Click "Start Server"
   - Default endpoint: `http://localhost:1234`

4. **Verify Connection**:
   ```bash
   curl http://localhost:1234/v1/models
   ```

### Python Dependencies

Already installed in your environment:
- `openai` (2.6.1) - For LM Studio API
- `numpy` (2.1.3) - Vector operations
- `scikit-learn` (1.7.2) - Cosine similarity

## Quick Start

### 1. Basic Usage

```python
from LLMCommunication.rag import RAGSystem

# Initialize RAG system (connects to LM Studio)
rag = RAGSystem()
# ✓ Connected to LM Studio at http://localhost:1234/v1
# Loaded vector store from .rag_index.pkl (5 operations)

# If no cached index exists, build it
if not rag.is_ready():
    rag.index_operations()
    # Building index for 5 operations...
    # ✓ Index built with 5 operations

# Search for operations
results = rag.search("move robot to position", top_k=3)

# Print results
for result in results:
    print(f"{result['metadata']['name']}: {result['score']:.3f}")
# Output:
# move_to_coordinate: 0.892
# detect_object: 0.654
# grip_object: 0.521
```

### 2. Get Full Context for LLM

```python
# Get comprehensive operation details for LLM
context = rag.get_operation_context("move robot to pick up object", top_k=3)

print(context['summary'])
# Found 3 relevant operations for: move robot to pick up object

# Access operation details
for op in context['operations']:
    print(f"\nOperation: {op['name']}")
    print(f"  Description: {op['description']}")
    print(f"  Parameters: {[p['name'] for p in op['parameters']]}")
    print(f"  Similarity: {op['similarity_score']:.3f}")
```

### 3. Filter by Category

```python
# Get only navigation operations
nav_ops = rag.search(
    "robot movement",
    category="navigation",
    top_k=5
)

# Or get all operations in a category
all_nav = rag.get_operations_by_category("navigation")
```

### 4. Find Similar Operations

```python
# Find operations similar to move_to_coordinate
similar = rag.find_similar_operations("motion_move_to_coord_001", top_k=3)

for op in similar:
    print(f"{op['metadata']['name']}: {op['score']:.3f}")
```

## Configuration

### Environment Variables

```bash
# LM Studio endpoint (default: http://localhost:1234/v1)
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"

# Embedding model name (must match model loaded in LM Studio)
export LM_STUDIO_EMBEDDING_MODEL="nomic-embed-text"

# API key (LM Studio doesn't require real key)
export LM_STUDIO_API_KEY="lm-studio"
```

### Configuration File

Edit `ACRLPython/LLMCommunication/rag/config.py`:

```python
class RAGConfig:
    # LM Studio connection
    LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
    LM_STUDIO_MODEL = "nomic-embed-text"

    # Embedding settings
    EMBEDDING_DIMENSION = 768  # Depends on model
    EMBEDDING_BATCH_SIZE = 10  # Batch size for embeddings

    # Search settings
    DEFAULT_TOP_K = 5  # Default number of results
    MIN_SIMILARITY_SCORE = 0.5  # Minimum score threshold

    # Fallback settings
    USE_TFIDF_FALLBACK = True  # Use TF-IDF if LM Studio unavailable
```

## API Reference

### RAGSystem Class

#### `__init__(lm_studio_url=None, embedding_model=None, registry=None, auto_load_index=True)`

Initialize the RAG system.

**Parameters:**
- `lm_studio_url` (str, optional): LM Studio base URL
- `embedding_model` (str, optional): Embedding model name
- `registry` (OperationRegistry, optional): Custom operation registry
- `auto_load_index` (bool): Auto-load cached index if available

#### `index_operations(rebuild=False) -> bool`

Build or rebuild the operation index.

**Parameters:**
- `rebuild` (bool): Force rebuild even if index exists

**Returns:**
- `bool`: True if successful

#### `search(query, top_k=None, min_score=None, category=None, complexity=None) -> List[Dict]`

Search for operations using natural language.

**Parameters:**
- `query` (str): Natural language search query
- `top_k` (int, optional): Number of results
- `min_score` (float, optional): Minimum similarity score
- `category` (str, optional): Filter by category
- `complexity` (str, optional): Filter by complexity

**Returns:**
- List of dicts with `operation_id`, `score`, `metadata`

#### `get_operation_context(query, top_k=3) -> Dict`

Get full operation context for LLM consumption.

**Parameters:**
- `query` (str): Natural language query
- `top_k` (int): Number of operations to include

**Returns:**
- Dict with `query`, `operations` (full details), `summary`

#### `get_operations_by_category(category, top_k=None) -> List[Dict]`

Get all operations in a specific category.

#### `find_similar_operations(operation_id, top_k=5) -> List[Dict]`

Find operations similar to a given operation.

#### `get_stats() -> Dict`

Get statistics about the RAG system.

#### `is_ready() -> bool`

Check if RAG system is ready for queries.

## Example Queries

### Task-Based Queries

```python
# Movement tasks
rag.search("move the robot arm to a specific position")
rag.search("navigate to coordinates x=0.3, y=0.15")
rag.search("approach the detected object")

# Manipulation tasks
rag.search("pick up an object")
rag.search("grasp the cube gently")
rag.search("release the object")

# Perception tasks
rag.search("find a red cube")
rag.search("detect objects in workspace")
rag.search("identify target location")

# Composite tasks
rag.search("pick and place operation")
rag.search("move object from A to B")
```

### Result Format

```python
{
    "operation_id": "motion_move_to_coord_001",
    "score": 0.892,  # Cosine similarity score (0-1)
    "metadata": {
        "name": "move_to_coordinate",
        "category": "navigation",
        "complexity": "basic",
        "description": "Move the robot's end effector to...",
        "average_duration_ms": 1200.0,
        "success_rate": 0.96
    }
}
```

## Integration with LLM Task Planning

### Example: LLM-Driven Robot Control

```python
from LLMCommunication.rag import RAGSystem
from LLMCommunication.operations import get_global_registry

# Initialize RAG system
rag = RAGSystem()

# User task description
user_task = "Move the robot to position x=0.3, y=0.15, z=0.1"

# 1. Use RAG to find relevant operations
context = rag.get_operation_context(user_task, top_k=3)

# 2. Pass context to LLM
llm_prompt = f"""
Task: {user_task}

Available operations:
{context['operations']}

Generate the execution plan using these operations.
"""

# 3. LLM generates plan (pseudocode)
plan = llm_generates_plan(llm_prompt)
# Example output:
# move_to_coordinate(robot_id="Robot1", x=0.3, y=0.15, z=0.1)

# 4. Execute operations via registry
registry = get_global_registry()
result = registry.execute_operation_by_name(
    "move_to_coordinate",
    robot_id="Robot1",
    x=0.3, y=0.15, z=0.1
)
```

## Troubleshooting

### LM Studio Connection Failed

**Error**: `Failed to connect to LM Studio`

**Solutions:**
1. Ensure LM Studio is running: Check "Local Server" tab
2. Verify embedding model is loaded (not just downloaded)
3. Check endpoint: Should be `http://localhost:1234/v1`
4. Test connection: `curl http://localhost:1234/v1/models`
5. Check port 1234 is not blocked by firewall

**Fallback**: If LM Studio is unavailable, the system will automatically fall back to TF-IDF embeddings (keyword-based, less semantic understanding).

### Empty Search Results

**Cause**: Query doesn't match any operations

**Solutions:**
1. Check if index is built: `rag.is_ready()`
2. Lower similarity threshold: `min_score=0.3`
3. Increase top_k: `top_k=10`
4. Try broader query: "robot movement" instead of "move robot to exact position x=0.3"

### Index Not Loading

**Error**: `No cached index found`

**Solution**: Build the index:
```python
rag.index_operations()
```

### Embedding Dimension Mismatch

**Error**: `Embedding dimension mismatch`

**Cause**: Changed embedding model without rebuilding index

**Solution**: Rebuild index:
```python
rag.index_operations(rebuild=True)
```

## Performance

### Embedding Generation

- **LM Studio (Local)**:
  - Speed: ~50-100ms per text (depends on model and hardware)
  - Quality: High semantic understanding
  - Cost: Free (local)

- **TF-IDF (Fallback)**:
  - Speed: <1ms per text
  - Quality: Keyword-based (less semantic)
  - Cost: Free

### Search Performance

- **Index Size**: ~1KB per operation
- **Search Time**: <10ms for 100 operations
- **Memory**: ~1MB for 100 operations

### Scaling

- **Small (<100 ops)**: In-memory numpy arrays (current implementation)
- **Medium (100-10K ops)**: Consider ChromaDB for persistent storage
- **Large (>10K ops)**: Use production vector database (Pinecone, Weaviate)

## Advanced Usage

### Custom Embedding Generator

```python
from LLMCommunication.rag import RAGSystem, EmbeddingGenerator

# Create custom embedding generator
custom_embedder = EmbeddingGenerator(
    base_url="http://custom-server:8000/v1",
    model="custom-embedding-model"
)

# Use with RAG system
rag = RAGSystem()
rag.embedding_generator = custom_embedder
rag.index_operations(rebuild=True)
```

### Custom Vector Store Path

```python
from LLMCommunication.rag import RAGSystem
from LLMCommunication.rag.config import config

# Set custom path
config.VECTOR_STORE_PATH = "/path/to/my_index.pkl"

# Initialize RAG with custom path
rag = RAGSystem()
```

### Batch Indexing

```python
from LLMCommunication.rag.indexer import build_index_from_registry

# Build index with custom settings
store = build_index_from_registry(
    registry=my_custom_registry,
    save_path="./custom_index.pkl"
)
```

## Future Enhancements

- [ ] Hybrid search (semantic + keyword + BM25)
- [ ] Query expansion and reranking
- [ ] Operation chaining suggestions
- [ ] Failure mode retrieval
- [ ] ChromaDB integration for production
- [ ] Multi-modal embeddings (text + parameters)
- [ ] Incremental index updates
- [ ] Distributed search for large operation sets

## Files

```
ACRLPython/LLMCommunication/rag/
├── __init__.py          # RAGSystem class and public API
├── config.py            # Configuration settings
├── embeddings.py        # LM Studio embedding wrapper
├── vector_store.py      # Numpy-based vector storage
├── indexer.py           # Index builder from registry
├── query_engine.py      # Search and retrieval logic
├── README.md            # This file
└── .rag_index.pkl       # Cached index (auto-generated)
```

## Related Documentation

- [Operations System](../operations/README.md) - Operation definitions and registry
- [LM Studio Docs](https://lmstudio.ai/docs) - LM Studio documentation
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity) - Similarity metric explained

## Support

For issues or questions:
1. Check LM Studio is running with embedding model loaded
2. Verify index is built: `rag.is_ready()`
3. Check logs for detailed error messages
4. Try rebuilding index: `rag.index_operations(rebuild=True)`
