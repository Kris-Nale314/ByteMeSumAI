# ByteMeSumAI

<p align="center">
  <img src="images/logo.svg" alt="ByteMeSumAI Logo" width="200"/>
</p>

Building Blocks for Robust and Context-Aware Retrieval-Augmented Generation

## Overview

ByteMeSumAI is a toolkit for developers and technical practitioners implementing Retrieval-Augmented Generation (RAG) systems. It addresses critical document architecture challenges that impact RAG performance, offering specialized components for semantic document processing, context preservation, and intelligent content extraction.

This library provides the infrastructure necessary to transform raw documents into contextually rich, semantically structured inputs for downstream RAG systems - a foundational requirement for context-aware AI applications.

## Key Features

- **Boundary-Aware Chunking**: Preserve semantic integrity by detecting natural document boundaries
- **Multi-Strategy Summarization**: Extract key information using various techniques
- **Entity Tracking**: Maintain entity references across document sections
- **Hierarchical Processing**: Process documents with awareness of their structural hierarchy
- **Evaluation Framework**: Quantitatively assess output quality

## Quick Start

```python
import bytemesumai as bm

# Load a document
doc = bm.Document.from_file("my_document.txt")

# Process with boundary-aware chunking
chunker = bm.ChunkingProcessor()
chunks = chunker.chunk_text_boundary_aware(doc.content)

# Create summaries with different strategies
summarizer = bm.SummarizationProcessor()
basic_summary = summarizer.basic_summary(doc.content)
entity_summary = summarizer.entity_focused_summary(doc.content)

print(f"Generated {len(chunks)} chunks")
print(f"Basic Summary: {basic_summary.summary[:100]}...")
```

## Documentation

Visit the full documentation to learn more about ByteMeSumAI's capabilities and how to use them effectively.

- [Installation Guide](installation.md)
- [Usage Examples](usage/quickstart.md)
- [API Reference](api/document.md)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/kris-nale314/bytemesumai/blob/main/LICENSE) file for details.