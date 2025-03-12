"""
ByteMeSumAI: Intelligent document processing with advanced chunking and summarization.
"""

__version__ = "0.1.0"

# Import core components for easy access
from bytemesumai.models.document import Document
from bytemesumai.chunking.processor import ChunkingProcessor
from bytemesumai.chunking.models import Chunk, DocumentBoundary, ChunkingResult

from bytemesumai.llm.client import LLMClient

from bytemesumai.summarization.processor import SummarizationProcessor