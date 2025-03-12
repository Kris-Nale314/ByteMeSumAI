"""
Document model for ByteMeSumAI.

This module defines the Document class, which represents a text document
with metadata and provides methods for loading and manipulating documents.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union, ClassVar
from pathlib import Path

class Document:
    """
    Represents a document with content and metadata.
    
    This class is the central data model for documents in ByteMeSumAI,
    providing methods for loading various file formats and accessing content.
    """
    
    # Class variable for common file extensions and their corresponding MIME types
    MIME_TYPES: ClassVar[Dict[str, str]] = {
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.json': 'application/json',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.csv': 'text/csv',
    }
    
    def __init__(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        filepath: Optional[str] = None,
        mimetype: Optional[str] = None
    ):
        """
        Initialize a document.
        
        Args:
            content: Document text content
            metadata: Optional metadata about the document
            filepath: Optional source filepath
            mimetype: Optional MIME type of the document
        """
        self.content = content
        self.metadata = metadata or {}
        self.filepath = filepath
        
        # Determine MIME type if not specified
        if mimetype:
            self.mimetype = mimetype
        elif filepath:
            ext = os.path.splitext(filepath)[1].lower()
            self.mimetype = self.MIME_TYPES.get(ext, 'text/plain')
        else:
            self.mimetype = 'text/plain'
    
    @property
    def length(self) -> int:
        """Return the length of the document in characters."""
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        """Return an approximate word count."""
        return len(self.content.split())
    
    @property
    def filename(self) -> Optional[str]:
        """Return the filename if available."""
        return os.path.basename(self.filepath) if self.filepath else None
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path], encoding: str = 'utf-8') -> 'Document':
        """
        Load a document from a file.
        
        Args:
            filepath: Path to the document file
            encoding: File encoding (default: utf-8)
            
        Returns:
            Document instance
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        filepath = str(filepath)  # Convert Path to string if needed
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Get file extension
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        # Handle different file formats
        if ext in ('.txt', '.md', '.html', '.htm'):
            # Plain text formats
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            metadata = {'source': filepath}
            
        elif ext == '.json':
            # JSON files may contain text or be structured
            with open(filepath, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, str):
                # Simple string JSON
                content = data
                metadata = {'source': filepath}
            elif isinstance(data, dict):
                # Dictionary JSON - look for content fields
                for field in ['content', 'text', 'body', 'contents']:
                    if field in data:
                        content = data[field]
                        break
                else:
                    # If no content field, use the whole JSON
                    content = json.dumps(data, indent=2)
                
                # Use other fields as metadata
                metadata = {k: v for k, v in data.items() if k != field}
                metadata['source'] = filepath
            else:
                content = json.dumps(data, indent=2)
                metadata = {'source': filepath}
                
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return cls(content=content, metadata=metadata, filepath=filepath)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary representation.
        
        Returns:
            Dictionary with document content and metadata
        """
        return {
            "content": self.content,
            "metadata": self.metadata,
            "length": self.length,
            "word_count": self.word_count,
            "mimetype": self.mimetype,
            "filename": self.filename
        }
    
    def __str__(self) -> str:
        """Return a string representation of the document."""
        filename = self.filename or "Unnamed document"
        return f"Document({filename}, {self.length} chars, {self.word_count} words)"
    
    def __len__(self) -> int:
        """Return the length of the document in characters."""
        return self.length
    
    def slice(self, start: int, end: Optional[int] = None) -> 'Document':
        """
        Create a new document from a slice of this document.
        
        Args:
            start: Start index
            end: End index (optional)
            
        Returns:
            New Document instance
        """
        sliced_content = self.content[start:end]
        
        # Clone metadata and add slice information
        new_metadata = self.metadata.copy()
        new_metadata['slice_start'] = start
        new_metadata['slice_end'] = end if end is not None else self.length
        new_metadata['original_length'] = self.length
        
        return Document(
            content=sliced_content,
            metadata=new_metadata,
            filepath=self.filepath,
            mimetype=self.mimetype
        )