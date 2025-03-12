# ByteMeSumAI Documentation

This directory contains documentation for the ByteMeSumAI package, including comprehensive guides, API references, and visual explanations of the package architecture and workflows.

## Visualizations

### 1. Architecture Overview

The architecture diagram (`images/architecture.svg`) provides a high-level overview of ByteMeSumAI's components and how they interact:

- **Document Processing**: The central system that coordinates all components
- **Chunking Engine**: Responsible for intelligent document segmentation
- **Summarization Engine**: Handles various summarization strategies
- **Evaluation Framework**: Assesses quality of outputs
- **LLM Client**: Provider-agnostic interface to language models

This visualization helps users understand how the components fit together and the different capabilities each component provides.

### 2. Advanced Summarization Workflow

The workflow diagram (`images/advanced_workflow.svg`) illustrates the detailed process flow in the advanced summarization example:

1. **Phase 1: Boundary Detection**
   - Detects section headers, paragraph breaks, and format shifts
   - Identifies natural boundaries in the document structure
   - Preserves document architecture for smarter processing

2. **Phase 2: Intelligent Chunking**
   - Applies boundary-aware chunking that respects document structure
   - Preserves sentences and coherent sections
   - Produces semantically meaningful document segments

3. **Phase 3: Multi-Strategy Summarization**
   - Applies different summarization techniques to the same content:
     - Basic summaries (concise/detailed)
     - Extractive summaries (important sentences)
     - Entity-focused summaries (organized by key entities)
     - Temporal summaries (chronological organization)
     - Contrastive summaries (comparing different sections/documents)

4. **Phase 4: Quality Evaluation**
   - Assesses output quality across multiple dimensions:
     - Completeness: Does it capture key information?
     - Conciseness: Is it appropriately brief?
     - Accuracy: Does it preserve factual content?
     - Coherence: Does it flow logically?

5. **LLM Integration**
   - Powers intelligence across all phases
   - Provider-agnostic (works with different LLM backends)

## Documentation Structure

- **API Reference**: Detailed documentation of all classes and methods
- **Guides**: Step-by-step tutorials for common use cases
- **Examples**: Annotated examples of the package in action
- **Images**: Visual explanations of concepts and workflows

## Building the Documentation

The documentation is built using MkDocs with the Material theme. To build the documentation locally:

1. Install the required dependencies:
   ```bash
   pip install -e ".[docs]"
   ```

2. Build the documentation:
   ```bash
   mkdocs build
   ```

3. Serve the documentation locally:
   ```bash
   mkdocs serve
   ```

## Contributing to Documentation

When adding to the documentation, please follow these guidelines:

1. **API Documentation**: Use Google-style docstrings in the code
2. **Code Examples**: Include working examples for key functionality
3. **Visual Explanations**: Add diagrams for complex workflows
4. **User Focus**: Consider both new users and advanced users