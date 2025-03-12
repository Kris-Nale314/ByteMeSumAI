"""
ByteMeSumAI - Building Blocks for Robust and Context-Aware RAG

This package provides advanced document processing capabilities for Retrieval-Augmented
Generation (RAG) systems, focusing on document architecture awareness, context
preservation, and multi-strategy summarization.
"""

__version__ = "0.1.0"

# Import main classes and functions for easier access
from bytemesumai.models.document import Document
from bytemesumai.chunking.processor import ChunkingProcessor
from bytemesumai.summarization.processor import SummarizationProcessor, summarize
from bytemesumai.reporting.markdown import MarkdownReporter
from bytemesumai.processing.document import DocumentProcessor

# Convenience functions for evaluation
from bytemesumai.evaluation.metrics import evaluate_summary, evaluate_chunking