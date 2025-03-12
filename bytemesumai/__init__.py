"""
ByteMeSumAI: Building Blocks for Robust and Context-Aware Retrieval-Augmented Generation
"""

__version__ = "0.1.0"

# Import core functionality for easy access
from bytemesumai.models.document import Document
from bytemesumai.chunking.processor import ChunkingProcessor
from bytemesumai.summarization.processor import SummarizationProcessor
from bytemesumai.processing.document import DocumentProcessor
from bytemesumai.llm.client import LLMClient


# Import evaluation functions
#from bytemesumai.evaluation.metrics import evaluate_summary