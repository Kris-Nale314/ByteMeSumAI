"""
Models for document chunking in SumWiseAI.

This module defines data models for document chunks and boundaries.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class DocumentBoundary:
    """Class representing a detected document boundary."""
    position: int
    boundary_type: str  # e.g., 'paragraph', 'section', 'document'
    confidence: float  # 0.0 to 1.0
    context: Optional[str] = None  # surrounding text for context
    
    def __str__(self) -> str:
        return f"Boundary({self.position}, {self.boundary_type}, conf={self.confidence:.2f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert boundary to dictionary representation."""
        return {
            "position": self.position,
            "type": self.boundary_type,
            "confidence": self.confidence,
            "context": self.context
        }

@dataclass
class Chunk:
    """Class representing a document chunk."""
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        """Return the length of the chunk in characters."""
        return len(self.text)
    
    def __str__(self) -> str:
        return f"Chunk({self.start_idx}:{self.end_idx}, {len(self.text)} chars)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "text": self.text,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "length": self.length,
            "metadata": self.metadata or {}
        }

@dataclass
class ChunkingResult:
    """Class representing the result of a chunking operation."""
    chunks: List[Chunk]
    strategy: str
    document_length: int
    boundaries: Optional[List[DocumentBoundary]] = None
    metrics: Optional[Dict[str, Any]] = None
    
    @property
    def chunk_count(self) -> int:
        """Return the number of chunks."""
        return len(self.chunks)
    
    @property
    def boundary_count(self) -> int:
        """Return the number of boundaries."""
        return len(self.boundaries) if self.boundaries else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunking result to dictionary representation."""
        return {
            "strategy": self.strategy,
            "document_length": self.document_length,
            "chunk_count": self.chunk_count,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "boundaries": [boundary.to_dict() for boundary in self.boundaries] if self.boundaries else None,
            "metrics": self.metrics
        }