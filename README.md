<p align="center">
  <img src="https://raw.githubusercontent.com/kris-nale314/bytemesumai/main/docs/images/logo.svg" alt="ByteMeSumAI Logo" width="250"/>
</p>

<h1 align="center">ByteMeSumAI</h1>
<p align="center"><strong>Building Blocks for Robust and Context-Aware Retrieval-Augmented Generation</strong></p>

<p align="center">
  <a href="https://github.com/kris-nale314/bytemesumai/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-alpha-orange" alt="Development Status"></a>
</p>

## Why Document Architecture Matters in RAG

Most RAG implementations treat documents as flat, unstructured text, leading to:

- **Context fragmentation** when chunks break across natural document boundaries
- **Entity amnesia** when references are lost between chunks
- **Semantic degradation** when document structure is ignored

ByteMeSumAI addresses these issues by preserving document architecture:

- **Boundary-aware chunking** respects natural document divisions
- **Entity tracking** maintains references across sections
- **Semantic awareness** preserves meaning and relationships
- **Hierarchical processing** maintains document structure

## Key Capabilities

<table>
<tr>
<td width="50%">

### Intelligent Chunking
- Boundary-aware segmentation
- Semantic coherence preservation
- Sentence integrity protection
- Document structure analysis

</td>
<td width="50%">

### Advanced Summarization
- Multi-strategy summarization
- Entity-focused analysis
- Temporal relationship preservation
- Cross-document comparison

</td>
</tr>
<tr>
<td colspan="2" align="center">
<img src="https://raw.githubusercontent.com/kris-nale314/bytemesumai/main/docs/images/advanced_workflow.svg" alt="Advanced Workflow" width="80%"/>
</td>
</tr>
</table>

## Quick Start

```python
import bytemesumai as bm

# Load a document
doc = bm.Document.from_file("my_document.txt")

# Process with boundary-aware chunking
chunker = bm.ChunkingProcessor()
chunking_result = chunker.chunk_document(
    text=doc,
    strategy="boundary_aware",
    compute_metrics=True
)

# Print chunking metrics
print(f"Created {len(chunking_result.chunks)} chunks")
print(f"Boundary preservation score: {chunking_result.metrics.get('boundary_preservation_score', 'N/A')}")

# Create a multi-strategy summary
summarizer = bm.SummarizationProcessor()
basic_summary = summarizer.basic_summary(doc.content, style="concise")
entity_summary = summarizer.entity_focused_summary(doc.content)

print(f"Basic Summary: {basic_summary.summary[:100]}...")
```

## Examples of Problems ByteMeSumAI Solves

- **Chunking that respects meaning**: When a legal document's sections are split mid-paragraph, key context is lost. ByteMeSumAI preserves these natural boundaries.

- **Entity tracking**: When "Company X" is referenced across different sections of a document, traditional RAG systems may lose track of which company is being discussed. ByteMeSumAI's entity tracking maintains these references.

- **Temporal coherence**: When events in a document are chronological, traditional chunking can scramble this timeline. ByteMeSumAI preserves temporal relationships.

- **Structure preservation**: When document hierarchy matters (e.g., headings, subsections), ByteMeSumAI maintains this structure for improved context.

## Core Components

```
ByteMeSumAI
├── Chunking Engine        # Document segmentation with semantic awareness
├── Summarization Engine   # Multi-strategy content distillation
├── Document Processors    # Hierarchical document handling
├── Entity Tracking        # Cross-document entity reference management
└── Evaluation Framework   # Quantitative assessment of output quality
```

## Installation

```bash
pip install bytemesumai
```

## Documentation

Visit the full documentation to learn more about ByteMeSumAI's capabilities:

- [Installation Guide](https://github.com/kris-nale314/bytemesumai/blob/main/docs/installation.md)
- [Quick Start Guide](https://github.com/kris-nale314/bytemesumai/blob/main/docs/quickstart.md)
- [API Reference](https://github.com/kris-nale314/bytemesumai/blob/main/docs/api/document.md)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/kris-nale314/bytemesumai/blob/main/LICENSE) file for details.

---

<p align="center">
<strong>Document architecture is the foundation of effective RAG systems.</strong><br>
ByteMeSumAI: Building the blocks for semantically-aware document processing.
</p>