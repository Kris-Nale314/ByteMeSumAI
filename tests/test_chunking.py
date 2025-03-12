"""
Tests for the chunking functionality in ByteMeSumAI.
"""

import pytest
from bytemesumai.chunking import ChunkingProcessor, Chunk, DocumentBoundary
from bytemesumai.models import Document


def test_chunking_processor_initialization():
    """Test ChunkingProcessor initialization with default parameters."""
    processor = ChunkingProcessor()
    assert processor.default_chunk_size == 1000
    assert processor.default_chunk_overlap == 200
    assert processor.boundary_detection_threshold == 0.7


def test_fixed_size_chunking():
    """Test fixed-size chunking with default parameters."""
    processor = ChunkingProcessor()
    text = "This is a test document. " * 100  # Create a longer document
    
    chunks = processor.chunk_text_fixed_size(text)
    
    # Basic validation
    assert chunks
    assert isinstance(chunks[0], Chunk)
    assert len(chunks) > 1
    
    # Check chunk properties
    for chunk in chunks:
        assert chunk.text
        assert chunk.start_idx >= 0
        assert chunk.end_idx <= len(text)
        assert chunk.metadata.get("strategy") == "fixed_size"


def test_boundary_detection():
    """Test document boundary detection."""
    processor = ChunkingProcessor()
    
    # Create a text with clear section boundaries
    text = """# Introduction
    
    This is the introduction section.
    
    # Background
    
    This is the background section with some context.
    
    # Methodology
    
    This describes our approach in detail.
    """
    
    boundaries = processor.detect_document_boundaries(text)
    
    # Basic validation
    assert boundaries
    assert isinstance(boundaries[0], DocumentBoundary)
    
    # At least the section headers should be detected
    assert len(boundaries) >= 2
    
    # Check boundary properties
    for boundary in boundaries:
        assert boundary.position >= 0
        assert boundary.position < len(text)
        assert boundary.boundary_type
        assert 0 <= boundary.confidence <= 1


def test_boundary_aware_chunking():
    """Test boundary-aware chunking."""
    processor = ChunkingProcessor()
    
    # Create a text with clear section boundaries
    text = """# Introduction
    
    This is the introduction section.
    
    # Background
    
    This is the background section with some context.
    
    # Methodology
    
    This describes our approach in detail.
    """
    
    chunks = processor.chunk_text_boundary_aware(text)
    
    # Basic validation
    assert chunks
    assert isinstance(chunks[0], Chunk)
    
    # Check chunk properties
    for chunk in chunks:
        assert chunk.text
        assert chunk.start_idx >= 0
        assert chunk.end_idx <= len(text)
        assert "boundary_aware" in chunk.metadata.get("strategy", "")


def test_chunking_metrics():
    """Test computation of chunking metrics."""
    processor = ChunkingProcessor()
    text = "This is a test document. " * 100
    
    chunks = processor.chunk_text_fixed_size(text)
    metrics = processor.compute_chunking_metrics(chunks, text)
    
    # Check metrics
    assert "chunk_count" in metrics
    assert metrics["chunk_count"] == len(chunks)
    assert "avg_chunk_size" in metrics
    assert "min_chunk_size" in metrics
    assert "max_chunk_size" in metrics


def test_document_chunking():
    """Test chunking a Document object."""
    processor = ChunkingProcessor()
    doc = Document(content="This is a test document. " * 100)
    
    result = processor.chunk_document(doc, strategy="fixed_size")
    
    # Check result properties
    assert result.chunks
    assert result.strategy == "fixed_size"
    assert result.document_length == len(doc.content)
    assert result.metrics is not None