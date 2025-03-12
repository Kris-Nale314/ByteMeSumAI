# Quick Start Guide

This guide will help you get started with ByteMeSumAI by walking through some common use cases.

## Basic Document Processing

```python
import bytemesumai as bm

# Create a document from text
text = """
# Project Overview

This document provides an overview of our latest project.

## Goals and Objectives

Our main goals are to improve user experience and increase conversion rates.

## Timeline

The project is scheduled to be completed within 3 months.
"""

# Create a document
doc = bm.Document(content=text)

# Process the document
processor = bm.DocumentProcessor()
result = processor.process_document(doc)

# Access the summary
summary = result["summarization_result"]["basic_summary"]
print(summary)
```

## Working with Files

```python
import bytemesumai as bm

# Load a document from a file
doc = bm.Document.from_file("my_document.txt")

# Create a chunking processor with custom settings
chunker = bm.ChunkingProcessor(
    default_chunk_size=1200,
    default_chunk_overlap=250
)

# Chunk the document
chunking_result = chunker.chunk_document(
    text=doc,
    strategy="boundary_aware",
    compute_metrics=True
)

# Print chunking metrics
print(f"Created {len(chunking_result.chunks)} chunks")
print(f"Average chunk size: {chunking_result.metrics['avg_chunk_size']:.1f} characters")
print(f"Boundary preservation score: {chunking_result.metrics.get('boundary_preservation_score', 'N/A')}")

# Print first chunk
if chunking_result.chunks:
    print("\nFirst chunk:")
    print(chunking_result.chunks[0].text[:150] + "...")
```

## Multi-Strategy Summarization

```python
import bytemesumai as bm

# Load a document
doc = bm.Document.from_file("earnings_call.txt")

# Create a summarization processor
summarizer = bm.SummarizationProcessor()

# Generate different types of summaries
basic_summary = summarizer.basic_summary(
    doc.content,
    style="concise"
)

detailed_summary = summarizer.basic_summary(
    doc.content,
    style="detailed"
)

entity_summary = summarizer.entity_focused_summary(
    doc.content,
    entities=["Revenue", "Growth", "Q4", "CEO"]
)

temporal_summary = summarizer.temporal_summary(
    doc.content
)

# Print the summaries
print("Concise Summary:")
print(basic_summary.summary)

print("\nEntity-Focused Summary:")
print(entity_summary.summary)

print("\nTemporal Summary:")
print(temporal_summary.summary)
```

## Comparing Chunking Strategies

```python
import bytemesumai as bm

# Load a document
doc = bm.Document.from_file("long_document.txt")

# Create a chunking processor
chunker = bm.ChunkingProcessor()

# Compare different chunking strategies
comparison = chunker.compare_chunking_strategies(doc)

# Print the comparison results
print("Chunking Strategy Comparison:")
print(f"Best overall strategy: {comparison['comparison']['summary']['best_overall_strategy']}")

print("\nBoundary Preservation Ranking:")
for strategy, score in comparison["comparison"]["boundary_preservation_ranking"]:
    print(f"- {strategy}: {score:.2f}")

print("\nSentence Integrity Ranking:")
for strategy, score in comparison["comparison"]["sentence_integrity_ranking"]:
    print(f"- {strategy}: {score:.2f}")
```

## Next Steps

Now that you've seen the basics, explore more specific usage examples:

- [Advanced Chunking Techniques](chunking.md)
- [Summarization Strategies](summarization.md)
- [Document Processing Pipeline](document-processing.md)

Or check out the complete [API Reference](../api/document.md) for detailed information on all available functions and classes.