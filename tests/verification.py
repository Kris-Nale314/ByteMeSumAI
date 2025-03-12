#!/usr/bin/env python
"""
ByteMeSumAI Package Verification Script

This script verifies that all major components of ByteMeSumAI can be imported
and basic functionality works as expected.
"""

def verify_imports():
    """Verify that all key imports work correctly."""
    print("Verifying ByteMeSumAI package imports...")
    
    # Main package imports
    print("Testing main package imports...")
    import bytemesumai
    from bytemesumai import Document, ChunkingProcessor, SummarizationProcessor
    from bytemesumai import DocumentProcessor, MarkdownReporter, evaluate_summary
    
    print(f"ByteMeSumAI version: {bytemesumai.__version__}")
    
    # Sub-package imports
    print("\nTesting sub-package imports...")
    from bytemesumai.models import Document
    from bytemesumai.chunking import Chunk, DocumentBoundary, ChunkingProcessor, chunk
    from bytemesumai.summarization import SummarizationProcessor, summarize
    from bytemesumai.llm.client import LLMClient
    from bytemesumai.processing import DocumentProcessor
    from bytemesumai.reporting import MarkdownReporter
    from bytemesumai.evaluation import evaluate_summary, evaluate_chunking
    
    print("All imports successful!")
    return True

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    from bytemesumai import Document, ChunkingProcessor, SummarizationProcessor
    
    # Create a test document
    test_text = """
    # Test Document
    
    This is a simple test document for ByteMeSumAI.
    
    ## Section 1
    
    This is the first section with some content.
    
    ## Section 2
    
    This is the second section with additional content.
    """
    
    doc = Document(content=test_text)
    print(f"Created document with {len(doc.content)} characters")
    
    # Test chunking
    chunker = ChunkingProcessor()
    chunking_result = chunker.chunk_document(doc, strategy="boundary_aware")
    print(f"Chunked document into {len(chunking_result.chunks)} parts")
    
    # Test summarization (mock mode to avoid API calls)
    summarizer = SummarizationProcessor(use_mock=True)
    summary = summarizer.basic_summary(test_text)
    print(f"Generated summary with {summary.word_count} words")
    
    print("Basic functionality tests passed!")
    return True

if __name__ == "__main__":
    verify_imports()
    test_basic_functionality()
    print("\nPackage verification complete!")