# ByteMeSumAI Examples

This directory contains example scripts demonstrating the advanced capabilities of ByteMeSumAI for intelligent document processing. These examples showcase how the library can be leveraged for Retrieval-Augmented Generation (RAG), Agentic AI, and other advanced AI applications.

## Example Scripts

### 1. Advanced Summarization Demo (`advanced_summarization_demo.py`)

This comprehensive demo showcases ByteMeSumAI's advanced summarization capabilities:

```bash
python advanced_summarization_demo.py --document-dir your_docs_folder --output output_folder
```

Optional flags:
- `--evaluate`: Run quality evaluation on each summary type (can be slow)

#### What This Demonstrates

- **Multi-Strategy Summarization**: Generates and compares different summary types (concise, detailed, bullet points, entity-focused, temporal)
- **Hierarchical Processing**: Shows how to chunk documents semantically and then build layered summaries
- **Metrics & Evaluation**: Provides quality metrics for different summarization approaches
- **Visualization**: Creates visualizations comparing summary quality metrics

### 2. Multi-Transcript Analyzer (`multi_transcript_analyzer.py`)

This example shows how to analyze and compare multiple documents (such as earnings call transcripts):

```bash
python multi_transcript_analyzer.py --transcript-dir your_transcripts_folder --output output_folder
```

#### What This Demonstrates

- **Cross-Document Analysis**: Compares content across multiple documents
- **Entity Tracking**: Identifies and follows key entities across documents
- **Financial Metrics Extraction**: Extracts and compares structured data
- **Comparative Reporting**: Generates comparison reports highlighting differences and similarities

### 3. Earnings Call Analyzer (`earnings_analyzer.py`)

A specialized example for financial document analysis:

```bash
python earnings_analyzer.py --transcript path/to/transcript.txt --output output_folder
```

#### What This Demonstrates

- **Financial Context Awareness**: Specialized processing for earnings calls
- **Entity & Metric Extraction**: Pulls out key financial data
- **Timeline Analysis**: Tracks temporal references and forecasts
- **Structured Output**: Generates organized reports with the most important information

## Why This Matters for RAG and Agentic AI

### Smart Document Architecture Improves RAG Performance

Traditional RAG systems often suffer from several limitations:

1. **Context Fragmentation**: Simple chunking methods break natural document boundaries, leading to loss of context
2. **Entity Amnesia**: Entities mentioned in one chunk are forgotten in another
3. **Temporal Confusion**: Chronological relationships are lost in naive splitting
4. **Structure Blindness**: Document hierarchy and formatting are ignored

ByteMeSumAI addresses these issues through:

- **Boundary-Aware Chunking**: Preserves natural document sections
- **Entity-Focused Summarization**: Maintains entity context across chunks
- **Temporal Coherence**: Preserves chronological relationships
- **Structure Preservation**: Respects document hierarchy

### Empowering Agentic AI Systems

For AI agents to effectively reason about and act upon document content, they need:

1. **Multi-Scale Understanding**: Both detailed and high-level document views
2. **Cross-Document Relationships**: Understanding how information connects across documents 
3. **Contextual Awareness**: Knowledge of what information belongs together
4. **Strategic Information Extraction**: Ability to pull out what matters most

These examples demonstrate how ByteMeSumAI provides building blocks that enable:

- **Adaptive Knowledge Granularity**: Agents can zoom in/out from detailed to summary views
- **Contextual Reasoning**: Maintain proper context around facts and claims
- **Information Synthesis**: Combine insights across multiple documents
- **Strategic Focus**: Target the most relevant information for a specific task

## Data Preparation Flow for RAG Systems

A typical RAG enhancement workflow using ByteMeSumAI:

1. **Document Loading**: Load documents from various sources
2. **Boundary Analysis**: Detect natural document boundaries
3. **Intelligent Chunking**: Split content while preserving context and boundaries
4. **Multi-Layered Summarization**: Generate summaries at different levels of detail
5. **Entity & Relationship Tracking**: Identify and track key entities and relationships
6. **Metrics & Evaluation**: Assess quality of processing steps
7. **Vector Storage**: Store processed chunks and summaries in vector database
8. **RAG Integration**: Connect to LLM system with enhanced retrieval capabilities

## Getting Started

To run these examples:

1. Install ByteMeSumAI:
   ```bash
   pip install bytemesumai
   ```

2. Prepare sample documents in a folder:
   - Text files (.txt)
   - Markdown files (.md)
   - Other supported formats

3. Run the example scripts pointing to your document directory:
   ```bash
   python examples/advanced_summarization_demo.py --document-dir my_docs --output results
   ```

4. Explore the generated reports and visualizations in the output directory

## Creating Your Own Examples

To build on these examples for your own use cases:

1. Import the key ByteMeSumAI components:
   ```python
   from bytemesumai.models import Document
   from bytemesumai.chunking import ChunkingProcessor
   from bytemesumai.summarization import SummarizationProcessor
   ```

2. Load your documents:
   ```python
   doc = Document.from_file("my_document.txt")
   ```

3. Process with appropriate strategies for your use case:
   ```python
   chunker = ChunkingProcessor()
   chunks = chunker.chunk_text_boundary_aware(doc.content)
   
   summarizer = SummarizationProcessor()
   summary = summarizer.entity_focused_summary(doc.content)
   ```

4. Integrate with your RAG or Agentic AI system:
   ```python
   # Example: Store in vector database
   for chunk in chunks:
       vector = embed_text(chunk.text)
       vector_db.store(vector, chunk.text, metadata=chunk.metadata)
   ```