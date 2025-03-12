# ByteMeSumAI

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Development Status](https://img.shields.io/badge/status-alpha-orange)

**Intelligent document processing with advanced chunking and summarization for the AI era.**

## 🚀 Overview

SumWiseAI tackles one of the most critical challenges in building effective AI solutions: **processing long, unstructured documents**. It's not just a summarization tool - it's an essential component in your RAG (Retrieval-Augmented Generation) pipeline that transforms how your AI systems handle complex documents.

**The hard truth**: Your AI solution is only as good as the data foundation it's built upon. Traditional document processing approaches break down when faced with lengthy, complex documents that span multiple topics, entities, and timelines.

SumWiseAI changes the game by implementing:

- **Boundary-Aware Chunking**: Intelligently splits documents at natural semantic boundaries, not arbitrary character counts
- **Multi-Strategy Summarization**: Applies different summarization approaches based on document complexity
- **Entity and Temporal Analysis**: Tracks key entities and chronological information across document sections
- **Hierarchical Processing**: Maintains context across document sections for coherent understanding

## 💡 Why Your Data Strategy Matters

In the rush to deploy LLM solutions, many teams overlook a fundamental truth: **document processing is not a commodity**. The way you chunk, summarize, and represent documents directly impacts:

- Retrieval accuracy
- Context retention
- Hallucination reduction
- Computational efficiency
- User experience

SumWiseAI is built on the principle that **your data strategy is as important as your solution architecture**. It provides the tools to transform raw documents into structured, coherent, and contextually rich inputs for your AI systems.

## 🔍 Key Features

- **Intelligent Chunking**:
  - Fixed-size chunking (baseline)
  - Boundary-aware chunking (detects natural boundaries)
  - Semantic chunking (preserves meaning)

- **Multi-Strategy Summarization**:
  - Basic summarization (concise, detailed, narrative)
  - Extractive summarization (key sentences)
  - Entity-focused summarization (key entities)
  - Temporal summarization (chronological information)

- **Advanced Processing**:
  - Hierarchical document handling
  - Entity tracking across document sections
  - Cross-reference preservation
  - Context window optimization

- **Practical Tools**:
  - Document loading utilities
  - Report generation
  - Evaluation metrics
  - Framework integration

## 🏗️ Use Cases

### Enterprise Document Processing

Process long corporate documents like:
- Earnings call transcripts
- Annual reports
- Technical documentation
- Legal contracts

```python
from sumwiseai import Document, DocumentProcessor

# Load an earnings call transcript
doc = Document.from_file("q4_earnings_call.txt")

# Process with intelligent chunking and summarization
processor = DocumentProcessor()
results = processor.process_document(
    document=doc,
    chunking_strategy="boundary_aware",
    summarization_strategies=["basic", "entity_focused"]
)
```

### RAG Enhancement

Supercharge your RAG pipeline with advanced document handling:

```python
# Traditional RAG breaks context on long documents
# SumWiseAI preserves semantic structure
from sumwiseai import chunk, summarize

# Intelligent chunking that respects document structure
chunks = chunk(document, strategy="boundary_aware")

# Generate targeted summaries for each chunk
summaries = [summarize(chunk.text, strategy="entity_focused") for chunk in chunks]

# Use in your RAG pipeline
for chunk, summary in zip(chunks, summaries):
    knowledge_base.add(
        text=chunk.text,
        metadata={"summary": summary.summary, "entities": summary.metadata["entities"]}
    )
```

### Research and Analysis

Extract insights from complex academic or research documents:

```python
from sumwiseai import Document, process_document

# Load a complex research paper
paper = Document.from_file("research_paper.pdf")

# Process with focus on entities and temporal information
results = process_document(
    paper,
    chunking_strategy="semantic",
    summarization_strategies=["entity_focused", "temporal"]
)

# Extract key research entities
entities = results["summarization_result"]["entity_focused"]["entities"]
print(f"Key research entities: {entities}")
```

## 📊 The SumWiseAI Advantage

| Challenge | Traditional Approach | SumWiseAI Approach |
|-----------|----------------------|-------------------|
| Long documents | Fixed-size chunking breaks context | Boundary-aware chunking preserves semantic structure |
| Multiple topics | Context mixing across unrelated sections | Topic-aware processing maintains separation |
| Entity tracking | Entities fragmented across chunks | Entity-focused summarization tracks across document |
| Temporal sequences | Chronological information lost | Temporal summaries preserve time-based relationships |
| Structure preservation | Document structure ignored | Format-aware processing respects original structure |

## 🌱 Getting Started



```
sumwiseai/
├── src/
│   └── sumwiseai/
│       ├── __init__.py                 # Package initialization
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py               # Provider-agnostic LLM client
│       │   └── providers/              # Provider implementations
│       │       ├── __init__.py
│       │       └── base.py             # Base provider interface
│       ├── chunking/
│       │   ├── __init__.py
│       │   ├── processor.py            # Main chunking processor
│       │   ├── fixed.py                # Fixed-size chunking
│       │   ├── boundary.py             # Boundary-aware chunking
│       │   ├── semantic.py             # Semantic chunking
│       │   └── models.py               # Chunk and Boundary models
│       ├── summarization/
│       │   ├── __init__.py
│       │   ├── processor.py            # Main summarization processor
│       │   ├── basic.py                # Basic summarization
│       │   ├── extractive.py           # Extractive summarization
│       │   ├── entity.py               # Entity-focused summarization
│       │   ├── temporal.py             # Temporal summarization
│       │   └── models.py               # Summary result models
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── document.py             # Document processor
│       │   └── hierarchical.py         # Hierarchical processing
│       ├── loaders/
│       │   ├── __init__.py
│       │   └── document.py             # Document loader
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── markdown.py             # Markdown reporter
│       │   └── evaluation.py           # Evaluation utilities
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── cache.py                # Caching utilities
│       │   └── metrics.py              # Evaluation metrics
│       └── models/
│           ├── __init__.py
│           └── document.py             # Document model
├── tests/                              # Test suite
│   ├── unit/                           # Unit tests
│   ├── integration/                    # Integration tests
│   └── fixtures/                       # Test fixtures
├── examples/                           # Example scripts
│   ├── basic_summarization.py
│   ├── boundary_aware_chunking.py
│   └── notebooks/                      # Jupyter notebooks
├── docs/                               # Documentation
├── pyproject.toml                      # Package configuration
├── LICENSE                             # License file
└── README.md                           # This file
```


### Installation

```bash
pip install sumwiseai
```

### Quick Start

```python
from sumwiseai import summarize, chunk

# Simple document summarization
summary = summarize("This is a long document that contains important information...")
print(summary.summary)

# Intelligent chunking
chunks = chunk(long_document, strategy="boundary_aware")
print(f"Document divided into {len(chunks)} semantic chunks")
```

For detailed examples, check the [examples](./examples) directory.

## 🚧 Project Status

SumWiseAI is currently in **alpha stage**. While the core functionality is available and useful, we're actively enhancing the library based on real-world usage and feedback.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
<strong>Your AI is only as good as the documents it processes.</strong><br>
Make document processing a strategic advantage with SumWiseAI.
</p>

