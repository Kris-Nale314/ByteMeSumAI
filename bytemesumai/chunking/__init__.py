"""
Document chunking functionality for ByteMeSumAI.
"""

from bytemesumai.chunking.models import Chunk, DocumentBoundary, ChunkingResult
from bytemesumai.chunking.processor import ChunkingProcessor

# Convenience function
def chunk(document, **kwargs):
    """
    Chunk a document using default settings.
    
    Args:
        document: Document or text to chunk
        **kwargs: Additional parameters for chunking
        
    Returns:
        ChunkingResult object
    """
    processor = ChunkingProcessor()
    return processor.chunk(document, **kwargs)