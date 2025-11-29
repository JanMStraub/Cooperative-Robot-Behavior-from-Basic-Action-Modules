# RAG System Documentation

## Overview

The **RAG (Retrieval-Augmented Generation) System** is a semantic search and retrieval framework for robot operations that enables LLM-driven robot control in the ACRL (Auto-Cooperative Robot Learning) project. It uses local embeddings via LM Studio and cosine similarity search to find relevant operations based on natural language queries, with multi-factor confidence scoring for reliable command execution.

### Purpose

The RAG system converts natural language commands like "move robot to position and close gripper" into structured robot operations by:
1. Semantically matching user intent against a registry of available operations
2. Providing ranked, relevant operations to LLMs for accurate command parsing
3. Enabling robust multi-command sequence execution with confidence scoring

### Key Features

- **Local Embeddings**: Uses LM Studio (nomic-embed-text) for 768-dimensional embeddings with no API costs
- **Semantic Search**: Cosine similarity-based vector search for natural language understanding
- **Confidence Scoring**: Multi-factor scoring combining similarity, metadata, parameters, and reliability
- **Persistent Index**: Cached vector store (`.rag_index.pkl`) for fast startup
- **Fallback Support**: TF-IDF embeddings when LM Studio unavailable
- **Category Filtering**: Filter operations by category (perception, navigation, manipulation, etc.)
- **Complexity Filtering**: Filter by operation complexity (atomic, basic, intermediate, complex)

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG SYSTEM ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│
│  User Query (Natural Language)
│         │
│         ▼
│  ┌──────────────────────────────────────────┐
│  │        QueryEngine                       │
│  │  - Generate query embedding              │
│  │  - Search vector store                   │
│  │  - Apply confidence scoring              │
│  └──────────────────────────────────────────┘
│         │
│         ▼
│  ┌──────────────────────────────────────────┐
│  │       VectorStore                        │
│  │  - In-memory numpy vectors (Nx768)       │
│  │  - Cosine similarity search              │
│  │  - Pickle persistence (.rag_index.pkl)   │
│  └──────────────────────────────────────────┘
│         │
│         ▼
│  ┌──────────────────────────────────────────┐
│  │   Operations + Metadata                  │
│  │  - Operation ID, name, category          │
│  │  - Complexity, description               │
│  │  - Parameters, success rate              │
│  └──────────────────────────────────────────┘
│
│  Initialization Flow:
│
│  Operations Registry ──▶ OperationIndexer ──▶ EmbeddingGenerator
│                             │                   (LM Studio)
│                             ▲                        │
│                             │                        │
│                             └────── embeddings ──────┘
│                             │
│                             └─────────▶ VectorStore.add_operation()
│                                              │
│                                              └─────▶ VectorStore.save()
│                                                           │
│                                                           └─▶ .rag_index.pkl
│
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

#### Indexing Flow
```
1. OperationIndexer gets all operations from Operations Registry
2. For each operation:
   - Generate rich text document (description, params, examples, etc.)
3. OperationIndexer calls EmbeddingGenerator.generate_embeddings(all_texts)
   - EmbeddingGenerator returns List[np.ndarray] embeddings (doesn't save)
4. OperationIndexer receives embeddings back
5. For each (operation, embedding):
   - OperationIndexer calls VectorStore.add_operation(id, embedding, metadata)
6. OperationIndexer calls VectorStore.save()
   - VectorStore creates .rag_index.pkl with vectors + metadata + IDs
```

#### Search Flow
```
1. User submits natural language query
2. QueryEngine generates query embedding
3. VectorStore computes cosine similarity with all operation embeddings
4. Apply category/complexity filters (if specified)
5. Apply confidence scoring (multi-factor boosting)
6. Return top-k results sorted by confidence score
```

---

## Core Components

### RAGSystem (`__init__.py`)

Main entry point and facade for the RAG system.

**Key Methods**:
```python
from rag import RAGSystem

rag = RAGSystem()  # Auto-loads cached index

# Search for operations
results = rag.search(
    query="move robot to position",
    top_k=5,
    min_score=0.5,
    category_filter="navigation"  # Optional
)

# Get full context for LLM
context = rag.get_operation_context(
    query="pick up object",
    top_k=3
)

# Find similar operations
similar = rag.find_similar_operations(
    operation_id="move_to_coordinate",
    top_k=3
)

# Rebuild index
rag.index_operations(rebuild=True)

# Check readiness
if rag.is_ready():
    print("RAG system ready!")

# Get statistics
stats = rag.get_stats()
print(f"Indexed operations: {stats['total_operations']}")
```

**Responsibilities**:
- Orchestrates all RAG components
- Manages initialization and lifecycle
- Provides public API for search and context retrieval
- Auto-loads cached index on startup

---

### EmbeddingGenerator (`Embeddings.py`)

Generates embeddings using LM Studio with TF-IDF fallback.

**Configuration**:
- **Primary**: LM Studio at `http://localhost:1234/v1` with `nomic-embed-text` model
- **Fallback**: TF-IDF vectorizer (500-dim max)
- **Embedding dimension**: 768 (LM Studio) or 500 (TF-IDF)
- **Batch size**: 10 texts per request
- **Timeout**: 30 seconds

**Usage**:
```python
from rag.Embeddings import EmbeddingGenerator

generator = EmbeddingGenerator()

# Single embedding
embedding = generator.generate_embedding("move to position")  # Shape: (768,)

# Batch embeddings
texts = ["move to position", "close gripper", "detect object"]
embeddings = generator.generate_embeddings(texts)  # List of (768,) arrays

# Check backend
if generator.is_using_lm_studio():
    print("Using LM Studio embeddings")
else:
    print("Using TF-IDF fallback")
```

**Error Handling**:
- Gracefully falls back to TF-IDF if LM Studio unavailable
- Returns zero vectors on complete failure
- Logs warnings with error details

---

### VectorStore (`VectorStore.py`)

In-memory vector database with cosine similarity search and persistence.

**Data Structure**:
```python
VectorStore {
    vectors: np.ndarray,          # Shape: (n_operations, 768)
    operation_ids: List[str],     # Operation IDs
    metadata: List[Dict],         # Operation metadata
    embedding_dimension: int      # Embedding size
}
```

**Key Methods**:
```python
from rag.VectorStore import VectorStore

store = VectorStore(embedding_dimension=768)

# Add operation
store.add_operation(
    operation_id="move_to_coordinate",
    embedding=embedding_vector,  # numpy array (768,)
    metadata={
        "name": "move_to_coordinate",
        "category": "navigation",
        "complexity": "basic",
        "success_rate": 0.95
    }
)

# Search
results = store.search(
    query_embedding=query_vector,
    top_k=5,
    min_score=0.5,
    filters={"category": "navigation"}
)

# Save/load
store.save(".rag_index.pkl")
loaded_store = VectorStore.load(".rag_index.pkl")

# Statistics
stats = store.get_stats()
print(f"Total operations: {stats['total_operations']}")
print(f"Categories: {stats['categories']}")
```

**Search Algorithm**:
- Uses sklearn `cosine_similarity` for efficient computation
- Applies category/complexity filters
- Applies confidence boosting (if enabled)
- Returns results sorted by confidence score

**Performance**:
- Search: O(n) where n = number of operations
- Typical: <10ms for 100 operations
- Memory: ~6KB per operation (768 × 8 bytes)

---

### OperationIndexer (`Indexer.py`)

Builds searchable index from operations registry.

**Usage**:
```python
from rag.Indexer import OperationIndexer

indexer = OperationIndexer()

# Build index
vector_store = indexer.build_index(save=True)  # Saves to .rag_index.pkl

# Rebuild index
vector_store = indexer.rebuild_index()

# Get statistics
stats = indexer.get_indexer_stats()
print(f"Operations indexed: {stats['operations_count']}")
```

**What Gets Indexed**:

Each operation's `to_rag_document()` method generates rich text including:
- Operation name and ID
- Category and complexity
- Short and long descriptions
- Usage examples
- Parameters (name, type, description, required, defaults, valid ranges)
- Preconditions and postconditions
- Performance metrics (duration, success rate)
- Known failure modes
- Related operations (required, commonly paired, mutually exclusive)

**Index Persistence**:
- Auto-saves to `.rag_index.pkl` if `AUTO_SAVE_INDEX=true` in config
- Pickle format (~500KB for 100 operations)
- Load time: <100ms

---

### QueryEngine (`QueryEngine.py`)

Semantic search orchestrator with confidence scoring.

**Key Methods**:
```python
from rag.QueryEngine import QueryEngine

engine = QueryEngine(vector_store, embedding_generator)

# Search with filters
results = engine.search(
    query="move robot to position",
    top_k=5,
    min_score=0.5,
    category_filter="navigation",
    complexity_filter="basic",
    include_full_operation=True  # Include full operation object
)

# Get LLM-ready context
context = engine.get_operation_context(
    query="pick up object",
    top_k=3
)

# Find similar operations
similar = engine.find_similar_operations(
    operation_id="move_to_coordinate",
    top_k=3
)
```

**Search Result Format**:
```python
{
    "operation_id": "move_to_coordinate",
    "score": 0.87,  # Confidence score after boosting
    "metadata": {
        "name": "move_to_coordinate",
        "category": "navigation",
        "complexity": "basic",
        "description": "Move robot end-effector to target position",
        "parameters": ["robot_id", "x", "y", "z", "speed"],
        "average_duration_ms": 2500.0,
        "success_rate": 0.95
    },
    "confidence": {
        "final_score": 0.87,
        "confidence_level": "high",  # high/medium/low/uncertain
        "breakdown": {
            "similarity": 0.82,
            "metadata_match": 0.85,
            "parameter_match": 0.90,
            "reliability": 0.95
        },
        "weights": {
            "similarity": 0.4,
            "metadata_match": 0.3,
            "parameter_match": 0.2,
            "reliability": 0.1
        }
    }
}
```

---

### ConfidenceScorer (`ConfidenceScorer.py`)

Multi-factor confidence computation for improved result ranking.

**Confidence Scoring Algorithm**:
```
Final Score = (
    0.4 × similarity_score +
    0.3 × metadata_match_score +
    0.2 × parameter_match_score +
    0.1 × reliability_score
)
```

**Component Scores**:

1. **Similarity Score** (40% weight):
   - Base cosine similarity from vector search
   - Range: 0.0 to 1.0

2. **Metadata Match Score** (30% weight):
   - Category filter match: +0.3 if matches, -0.2 if doesn't
   - Complexity filter match: +0.2 if matches, -0.1 if doesn't
   - Base: 0.5 (neutral when no filters)

3. **Parameter Match Score** (20% weight):
   - Analyzes query text for mentions of operation parameters
   - Splits snake_case parameters into terms
   - Scores based on term overlap: 0.3 (no match) to 1.0 (full match)

4. **Reliability Score** (10% weight):
   - Based on operation's historical `success_rate`
   - Direct mapping: reliability = success_rate

**Confidence Levels**:
- **HIGH**: ≥ 0.75
- **MEDIUM**: 0.5 to 0.75
- **LOW**: 0.25 to 0.5
- **UNCERTAIN**: < 0.25

**Category-Specific Thresholds**:
```python
CATEGORY_MIN_SCORES = {
    "navigation": 0.6,
    "manipulation": 0.55,
    "perception": 0.50,
    "coordination": 0.60,
    "state_query": 0.55
}
```

**Usage**:
```python
from rag.ConfidenceScorer import apply_confidence_boosting

# Boost search results
enhanced_results = apply_confidence_boosting(
    results=search_results,
    query_text="move to position",
    filters={"category": "navigation"}
)

# Results are re-ranked by confidence score
for result in enhanced_results:
    print(f"{result['operation_id']}: {result['confidence']['final_score']:.2f} ({result['confidence']['confidence_level']})")
```

---

### Config (`Config.py`)

Centralized configuration with environment variable support.

**Configuration Options**:

```python
from rag.Config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_SCORE,
    ENABLE_CONFIDENCE_SCORING
)

# LM Studio
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_EMBEDDING_MODEL = "nomic-embed-text"
LM_STUDIO_API_KEY = "lm-studio"  # Dummy, not required

# Embeddings
EMBEDDING_DIMENSION = 768
EMBEDDING_BATCH_SIZE = 10
EMBEDDING_TIMEOUT = 30  # seconds

# Vector Store
VECTOR_STORE_PATH = ".rag_index.pkl"
AUTO_SAVE_INDEX = True

# Search
DEFAULT_TOP_K = 5
MIN_SIMILARITY_SCORE = 0.5

# Confidence Scoring
ENABLE_CONFIDENCE_SCORING = True
CONFIDENCE_STRATEGY = "balanced"  # strict/balanced/permissive
CONFIDENCE_TIERS = {"high": 0.75, "medium": 0.5, "low": 0.25}

# Fallback
USE_TFIDF_FALLBACK = True
TFIDF_MAX_FEATURES = 500
```

**Environment Variables**:

Override defaults by setting environment variables:
```bash
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
export LM_STUDIO_EMBEDDING_MODEL="bge-small-en-v1.5"
export EMBEDDING_DIMENSION=384
export DEFAULT_TOP_K=10
```

---

## Quick Start Guide

### Prerequisites

1. **LM Studio** (recommended):
   - Download from [https://lmstudio.ai/](https://lmstudio.ai/)
   - Load an embedding model (e.g., `nomic-embed-text`)
   - Start the local server (default port: 1234)
   - Verify: `curl http://localhost:1234/v1/models`

2. **Python Dependencies**:
   ```bash
   pip install numpy scikit-learn openai
   ```

### Basic Usage

```python
from rag import RAGSystem

# Initialize (auto-loads cached index if available)
rag = RAGSystem()

# First time: build the index
if not rag.is_ready():
    rag.index_operations()

# Search for operations
results = rag.search("move robot to position", top_k=5)

for result in results:
    print(f"{result['metadata']['name']}: {result['score']:.2f}")
    print(f"  Category: {result['metadata']['category']}")
    print(f"  Confidence: {result['confidence']['confidence_level']}")
```

### Common Use Cases

#### 1. Natural Language Command Parsing

```python
from rag import RAGSystem

rag = RAGSystem()

# Get operations relevant to user command
command = "detect the blue cube and move to it"
results = rag.search(command, top_k=3)

# Use results to inform LLM or regex-based parser
for result in results:
    print(f"Relevant operation: {result['metadata']['name']}")
    print(f"Parameters: {result['metadata']['parameters']}")
```

#### 2. Operation Discovery

```python
# Find all perception operations
perception_ops = rag.search(
    query="detect objects",
    category_filter="perception",
    top_k=10
)

# Find basic navigation operations
navigation_ops = rag.search(
    query="move robot",
    category_filter="navigation",
    complexity_filter="basic"
)
```

#### 3. Get Full Operation Context for LLM

```python
# Get detailed context for LLM task planning
context = rag.get_operation_context(
    query="pick up object",
    top_k=3
)

# Context includes full operation details, parameters, examples
llm_prompt = f"""
Available operations:
{context['operations_text']}

User request: Pick up the red cube
Generate a plan:
"""
```

#### 4. Find Similar Operations

```python
# Find operations similar to move_to_coordinate
similar = rag.find_similar_operations(
    operation_id="move_to_coordinate",
    top_k=3
)

# Useful for discovering alternative or complementary operations
```

---

## Integration Points

### Operations Registry Integration

The RAG system indexes operations from the Operations Registry:

```python
# Operations are defined in operations/
from operations import BasicOperation, OperationCategory, OperationComplexity

# Example operation definition
move_op = BasicOperation(
    operation_id="move_to_coordinate",
    name="move_to_coordinate",
    category=OperationCategory.NAVIGATION,
    complexity=OperationComplexity.BASIC,
    description="Move robot end-effector to target position",
    # ... parameters, examples, preconditions, etc.
)

# RAG indexes all registered operations
from operations import get_global_registry
registry = get_global_registry()  # 9 operations currently registered
```

### CommandParser Integration

CommandParser uses RAG for semantic operation discovery:

```python
# orchestrators/CommandParser.py
from rag import RAGSystem

class CommandParser:
    def __init__(self, use_rag=True):
        if use_rag:
            self.rag = RAGSystem()
            self.rag.index_operations(rebuild=True)

    def parse(self, command_text):
        # Get relevant operations from RAG
        rag_results = self.rag.search(command_text, top_k=5)

        # Prioritize RAG results in LLM prompt
        # This improves parsing accuracy
        return self._parse_with_context(command_text, rag_results)
```

### RAGServer Integration (Network Service)

RAG is exposed as a TCP network service on port 5011:

```python
# orchestrators/RunRAGServer.py
from servers.RAGServer import RAGQueryHandler, run_rag_server_background

# Initialize with validation
RAGQueryHandler.initialize(rebuild_index=False, validate=True)

# Start server
run_rag_server_background(config)

# Unity can now query via RAGClient
```

**Protocol**: Query/response over TCP with request ID correlation

**Unity C# Integration**:
```csharp
// Unity side: PythonCommunication/RAGClient.cs
using PythonCommunication;

var ragClient = RAGClient.Instance;
ragClient.Connect("localhost", 5011);

// Query RAG
var result = await ragClient.QueryAsync(
    queryText: "move robot to position",
    topK: 5,
    filters: new Dictionary<string, string> { ["category"] = "navigation" }
);

foreach (var op in result.Operations) {
    Debug.Log($"Operation: {op.Name}, Score: {op.Score}");
}
```

### SequenceServer Integration

SequenceServer (port 5013) uses RAG for multi-command parsing:

```python
# orchestrators/SequenceServer.py
from rag import RAGSystem

rag = RAGSystem()
rag.index_operations()

# Parse compound command
command = "move to (0.3, 0.2, 0.1) and close gripper"
relevant_ops = rag.search(command, top_k=5)

# Use relevant ops to guide LLM/regex parsing
# Returns: [move_to_coordinate, control_gripper]
```

---

## Configuration

### LM Studio Setup

1. **Install LM Studio**:
   - Download from [https://lmstudio.ai/](https://lmstudio.ai/)
   - Install and launch

2. **Load Embedding Model**:
   - Recommended: `nomic-embed-text` (768-dim, high quality)
   - Alternative: `bge-small-en-v1.5` (384-dim, faster)
   - Click "Download" tab, search for model, download

3. **Start Local Server**:
   - Click "Local Server" tab
   - Load the downloaded embedding model
   - Start server on port 1234 (default)
   - Verify with: `curl http://localhost:1234/v1/models`

4. **Update Config** (if using non-default model):
   ```python
   # In rag/Config.py or set environment variable
   export LM_STUDIO_EMBEDDING_MODEL="bge-small-en-v1.5"
   export EMBEDDING_DIMENSION=384  # Match model dimension
   ```

### Custom Embedding Models

To use a different embedding model:

```python
import os

# Set before importing RAG
os.environ["LM_STUDIO_EMBEDDING_MODEL"] = "your-model-name"
os.environ["EMBEDDING_DIMENSION"] = "model-dimension"

from rag import RAGSystem

# Rebuild index with new embeddings
rag = RAGSystem()
rag.index_operations(rebuild=True)
```

### Confidence Scoring Strategies

```python
from rag import Config

# Strict: Higher thresholds for all categories
Config.CONFIDENCE_STRATEGY = "strict"
Config.CATEGORY_MIN_SCORES["navigation"] = 0.7
Config.CATEGORY_MIN_SCORES["manipulation"] = 0.65

# Permissive: Lower thresholds for broader matches
Config.CONFIDENCE_STRATEGY = "permissive"
Config.MIN_SIMILARITY_SCORE = 0.3

# Balanced (default): Moderate thresholds
Config.CONFIDENCE_STRATEGY = "balanced"
```

### Disable Confidence Scoring

```python
from rag import Config

# Use raw similarity scores only
Config.ENABLE_CONFIDENCE_SCORING = False
```

---

## API Reference

### RAGSystem

```python
class RAGSystem:
    def __init__(self):
        """Initialize RAG system, auto-load cached index"""

    def index_operations(self, rebuild: bool = False) -> bool:
        """Build/rebuild operation index"""

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.5,
        category_filter: str = None,
        complexity_filter: str = None
    ) -> List[Dict]:
        """Search for operations"""

    def get_operation_context(self, query: str, top_k: int = 3) -> Dict:
        """Get LLM-ready operation context"""

    def find_similar_operations(self, operation_id: str, top_k: int = 3) -> List[Dict]:
        """Find similar operations by ID"""

    def is_ready(self) -> bool:
        """Check if indexed and ready"""

    def get_stats(self) -> Dict:
        """Get system statistics"""
```

### EmbeddingGenerator

```python
class EmbeddingGenerator:
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate single embedding"""

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate batch embeddings"""

    def is_using_lm_studio(self) -> bool:
        """Check if using LM Studio"""
```

### VectorStore

```python
class VectorStore:
    def add_operation(self, operation_id: str, embedding: np.ndarray, metadata: Dict):
        """Add operation to store"""

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.5,
        filters: Dict = None
    ) -> List[Dict]:
        """Search for similar operations"""

    def get_operation(self, operation_id: str) -> Dict:
        """Get operation by ID"""

    def save(self, file_path: str):
        """Save to pickle file"""

    @staticmethod
    def load(file_path: str) -> 'VectorStore':
        """Load from pickle file"""

    def get_stats(self) -> Dict:
        """Get store statistics"""
```

### QueryEngine

```python
class QueryEngine:
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.5,
        category_filter: str = None,
        complexity_filter: str = None,
        include_full_operation: bool = False
    ) -> List[Dict]:
        """Semantic search"""

    def get_operation_context(self, query: str, top_k: int = 3) -> Dict:
        """Get full operation context for LLM"""

    def find_similar_operations(self, operation_id: str, top_k: int = 3) -> List[Dict]:
        """Find similar operations"""
```

---

## Algorithms

### Cosine Similarity Search

The RAG system uses cosine similarity to measure semantic similarity between query and operation embeddings:

```python
from sklearn.metrics.pairwise import cosine_similarity

# Query embedding: (1, 768)
# Operation embeddings: (n, 768)
similarities = cosine_similarity(query_embedding, operation_embeddings)

# Returns scores in range [-1, 1]
# 1.0 = identical vectors (perfect match)
# 0.0 = orthogonal vectors (no similarity)
# -1.0 = opposite vectors (rare with normalized embeddings)
```

**Complexity**: O(n × d) where n = operations, d = embedding dimension
**Performance**: ~10ms for 100 operations with 768-dim embeddings

### Confidence Boosting Algorithm

Multi-factor scoring to improve ranking accuracy:

```python
def compute_confidence_score(similarity, metadata, query, filters):
    # 1. Similarity score (40% weight)
    similarity_score = similarity  # From cosine similarity

    # 2. Metadata match score (30% weight)
    metadata_score = calculate_metadata_match(metadata, filters)

    # 3. Parameter match score (20% weight)
    param_score = calculate_parameter_match(query, metadata['parameters'])

    # 4. Reliability score (10% weight)
    reliability_score = metadata.get('success_rate', 0.5)

    # Weighted combination
    final_score = (
        0.4 * similarity_score +
        0.3 * metadata_score +
        0.2 * param_score +
        0.1 * reliability_score
    )

    return max(0.0, min(1.0, final_score))
```

### Parameter Matching

Analyzes query text for parameter mentions:

```python
def calculate_parameter_match_score(query_text, parameters):
    # Tokenize query
    query_terms = set(re.findall(r'\w+', query_text.lower()))

    # Extract parameter terms (handle snake_case)
    param_terms = set()
    for param in parameters:
        parts = param.lower().split('_')
        param_terms.update(parts)
        param_terms.add(param.lower().replace('_', ''))

    # Calculate overlap
    matches = len(param_terms & query_terms)

    if matches == 0:
        return 0.3  # Low but not zero
    else:
        # Scale from 0.5 to 1.0 based on match ratio
        return min(1.0, 0.5 + (matches / len(param_terms)) * 0.5)
```

---

## Performance & Optimization

### Embedding Generation

| Backend | Time per Embedding | Dimension | Quality |
|---------|-------------------|-----------|---------|
| LM Studio (nomic-embed-text) | 50-100ms | 768 | High |
| TF-IDF (fallback) | 1-10ms | 500 | Medium |

**Optimization**: Batch processing (10 texts per request) reduces overhead

### Vector Search

| Operations | Search Time | Memory Usage |
|-----------|-------------|--------------|
| 10 | <1ms | ~60KB |
| 100 | <10ms | ~600KB |
| 1000 | <100ms | ~6MB |

**Complexity**: O(n) linear search with numpy/sklearn optimizations

### Index Persistence

| Metric | Value |
|--------|-------|
| File size (100 ops) | ~500KB |
| Save time | <100ms |
| Load time | <100ms |
| Format | Pickle (binary) |

**Best Practice**: Use cached index (`.rag_index.pkl`) for fast startup

### Memory Usage

```
Per Operation:
- Embedding: 768 floats × 8 bytes = 6,144 bytes
- Metadata: ~1KB (JSON)
- Total: ~7KB per operation

For 100 Operations:
- Vectors: ~600KB
- Metadata: ~100KB
- Total: ~700KB
```

### Optimization Tips

1. **Use cached index**: Don't rebuild on every startup
2. **Batch embeddings**: Process multiple texts together
3. **Filter early**: Apply category/complexity filters before scoring
4. **Adjust top_k**: Only retrieve what you need (default: 5)
5. **Use TF-IDF for offline**: When LM Studio not critical

---

## Troubleshooting

### LM Studio Connection Failed

**Symptom**: `Failed to connect to LM Studio` warning in logs

**Solutions**:
1. Verify LM Studio is running: `curl http://localhost:1234/v1/models`
2. Check embedding model is loaded in LM Studio
3. Verify port (default: 1234) is not blocked
4. Check `LM_STUDIO_BASE_URL` in config
5. Allow fallback to TF-IDF: `USE_TFIDF_FALLBACK=True`

### Embedding Dimension Mismatch

**Symptom**: `ValueError: shapes not aligned` during search

**Cause**: Index built with different embedding dimension than current model

**Solution**:
```python
from rag import RAGSystem

rag = RAGSystem()
rag.index_operations(rebuild=True)  # Force rebuild with current model
```

### Index Corruption

**Symptom**: `Failed to load index` or pickle errors

**Solution**:
```bash
# Delete corrupted index
rm ACRLPython/rag/.rag_index.pkl

# Rebuild
python -c "from rag import RAGSystem; rag = RAGSystem(); rag.index_operations()"
```

### No Results Returned

**Symptom**: Search returns empty list `[]`

**Causes & Solutions**:
1. **min_score too high**: Lower threshold `min_score=0.3`
2. **Wrong category filter**: Check category name spelling
3. **Empty index**: Rebuild with `index_operations()`
4. **Very different query**: Try simpler query terms

### Low Confidence Scores

**Symptom**: All results have `confidence_level="low"`

**Solutions**:
1. **Adjust strategy**: Use `CONFIDENCE_STRATEGY="permissive"`
2. **Lower thresholds**: Reduce `CATEGORY_MIN_SCORES` values
3. **Disable confidence scoring**: `ENABLE_CONFIDENCE_SCORING=False`
4. **Check success_rate**: Ensure operations have realistic success rates

### Slow Search Performance

**Symptom**: Search takes >1 second for small operation set

**Causes & Solutions**:
1. **Large vector store**: Expected for 10,000+ operations
2. **Slow embeddings**: Check LM Studio response time
3. **Debug mode**: Disable verbose logging
4. **CPU-bound**: Use batch processing for multiple queries

---

## Development Guide

### Adding New Operations

1. **Define operation** in `operations/`:
   ```python
   from operations import BasicOperation, OperationCategory

   my_op = BasicOperation(
       operation_id="my_operation",
       name="my_operation",
       category=OperationCategory.NAVIGATION,
       description="My custom operation",
       # ... add parameters, examples, etc.
   )
   ```

2. **Register operation**:
   ```python
   from operations import get_global_registry

   registry = get_global_registry()
   registry.register_operation(my_op)
   ```

3. **Rebuild RAG index**:
   ```python
   from rag import RAGSystem

   rag = RAGSystem()
   rag.index_operations(rebuild=True)
   ```

### Rebuilding the Index

**When to rebuild**:
- Added/removed operations
- Changed operation descriptions or metadata
- Changed embedding model
- Corrupted index file

**How to rebuild**:
```python
from rag import RAGSystem

rag = RAGSystem()
rag.index_operations(rebuild=True)  # Force rebuild

# Or via command line
python -m orchestrators.RunRAGServer --rebuild-index
```

### Testing RAG Queries

**Interactive testing**:
```python
from rag import RAGSystem

rag = RAGSystem()

# Test queries
queries = [
    "move robot to position",
    "detect objects",
    "close gripper",
    "get robot status"
]

for query in queries:
    print(f"\nQuery: {query}")
    results = rag.search(query, top_k=3)

    for r in results:
        print(f"  {r['metadata']['name']}: {r['score']:.2f} ({r['confidence']['confidence_level']})")
```

**Unit testing**:
```bash
# Run RAG tests
pytest ACRLPython/tests/TestRAGServer.py -v
```

### Debugging Tips

1. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check embedding quality**:
   ```python
   from rag.Embeddings import EmbeddingGenerator

   gen = EmbeddingGenerator()
   emb1 = gen.generate_embedding("move to position")
   emb2 = gen.generate_embedding("navigate to location")

   from sklearn.metrics.pairwise import cosine_similarity
   similarity = cosine_similarity([emb1], [emb2])[0][0]
   print(f"Similarity: {similarity:.3f}")  # Should be high (>0.7)
   ```

3. **Inspect index contents**:
   ```python
   from rag.VectorStore import VectorStore

   store = VectorStore.load(".rag_index.pkl")
   stats = store.get_stats()

   print(f"Operations: {stats['total_operations']}")
   print(f"Categories: {stats['categories']}")
   print(f"Operations: {stats['operation_ids']}")
   ```

4. **Test confidence scoring**:
   ```python
   from rag.ConfidenceScorer import compute_confidence_score

   score = compute_confidence_score(
       similarity=0.8,
       metadata={"parameters": ["x", "y", "z"], "success_rate": 0.95},
       query_text="move to x y z position",
       filters={"category": "navigation"}
   )
   print(f"Confidence score: {score}")
   ```

---

## File Structure

```
ACRLPython/rag/
├── __init__.py              # RAGSystem main facade (298 lines)
├── Config.py                # Configuration management (114 lines)
├── Embeddings.py            # LM Studio + TF-IDF embeddings (236 lines)
├── VectorStore.py           # Vector database + search (295 lines)
├── Indexer.py               # Index builder (186 lines)
├── QueryEngine.py           # Semantic search (226 lines)
├── ConfidenceScorer.py      # Multi-factor scoring (270 lines)
├── .rag_index.pkl           # Cached vector index (auto-generated)
└── README.md                # This file
```

**Total**: ~1625 lines of code

---

## References

### Related Documentation
- [Operations System](../operations/README.md) - Robot operation definitions
- [OBJECT_DETECTION_README.md](../OBJECT_DETECTION_README.md) - Vision system
- [STATUS_OPERATION_README.md](../STATUS_OPERATION_README.md) - Status queries
- [REFACTORING_PLAN.md](../REFACTORING_PLAN.md) - Architecture overview

### External Resources
- [LM Studio](https://lmstudio.ai/) - Local LLM server
- [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) - Embedding model
- [scikit-learn cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

### Key Dependencies
- `numpy` - Vector operations
- `scikit-learn` - Cosine similarity, TF-IDF
- `openai` - LM Studio API client
- `pickle` - Index persistence

---

## License

Part of the ACRL (Auto-Cooperative Robot Learning) project - Master's thesis
