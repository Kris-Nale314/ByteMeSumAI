"""
SumWiseAI: Intelligent document processing with advanced chunking and summarization.
"""

__version__ = "0.1.0"

# Import core components for easy access
from sumwiseai.models.document import Document
from sumwiseai.chunking.processor import ChunkingProcessor
from sumwiseai.chunking.models import Chunk, DocumentBoundary, ChunkingResult

from sumwiseai.llm.client import LLMClient

from sumwiseai.summarization.processor import SummarizationProcessor