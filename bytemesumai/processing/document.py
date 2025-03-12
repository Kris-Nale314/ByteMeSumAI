# Copyright (c) 2025 Kris Naleszkiewicz
# Licensed under the MIT License - see LICENSE file for details
"""
Document processor for ByteMeSumAI.

This module provides the DocumentProcessor class, which combines chunking and
summarization to process documents in an intelligent way.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union

from bytemesumai.models.document import Document
from bytemesumai.chunking.processor import ChunkingProcessor
from bytemesumai.summarization.processor import SummarizationProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Process documents with intelligent chunking and multi-strategy summarization.
    
    This class combines chunking and summarization processors to provide
    a complete document processing pipeline.
    """
    
    def __init__(
        self,
        chunking_processor: Optional[ChunkingProcessor] = None,
        summarization_processor: Optional[SummarizationProcessor] = None,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the document processor.
        
        Args:
            chunking_processor: Optional custom chunking processor
            summarization_processor: Optional custom summarization processor
            model: Default model for processors (if not provided)
        """
        # Initialize processors
        self.chunking_processor = chunking_processor or ChunkingProcessor()
        self.summarization_processor = summarization_processor or SummarizationProcessor(model=model)
        
        logger.info(f"Initialized DocumentProcessor with model {model}")
    
    def process_document(
        self,
        document: Union[str, Document],
        chunking_strategy: str = "boundary_aware",
        summarization_strategies: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a document with chunking and multi-strategy summarization.
        
        Args:
            document: Document or text to process
            chunking_strategy: Strategy for chunking ('fixed_size', 'boundary_aware', 'semantic')
            summarization_strategies: List of summarization strategies to apply
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary with processing results
        """
        # Default summarization strategies if not provided
        if summarization_strategies is None:
            summarization_strategies = ["basic", "extractive"]
        
        # Convert string to Document if needed
        if isinstance(document, str):
            document = Document(content=document)
            
        logger.info(f"Processing document ({len(document.content)} chars) with {chunking_strategy} chunking")
        start_time = time.time()
        
        # Basic document info
        results = {
            "document_info": {
                "length": len(document.content),
                "word_count": len(document.content.split()),
                "filename": getattr(document, "filename", None)
            },
            "processing_info": {
                "chunking_strategy": chunking_strategy,
                "summarization_strategies": summarization_strategies,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
            }
        }
        
        # Determine if document is long
        is_long_document = len(document.content) > 4000
        results["processing_info"]["is_long_document"] = is_long_document
        
        # Apply chunking if document is long
        if is_long_document:
            chunking_result = self.chunking_processor.chunk_document(
                text=document.content,
                strategy=chunking_strategy,
                compute_metrics=True
            )
            
            results["chunking_result"] = {
                "strategy": chunking_result.strategy,
                "chunk_count": len(chunking_result.chunks),
                "metrics": chunking_result.metrics
            }
            
            # Process each chunk for long documents
            chunk_texts = [chunk.text for chunk in chunking_result.chunks]
            combined_summary = self.summarization_processor.basic_summary(
                "\n\n".join(chunk_texts[:3]),  # Use first few chunks for simplicity
                style="concise"
            )
            
            results["summarization_result"] = {
                "basic_summary": combined_summary.summary,
                "word_count": len(combined_summary.summary.split())
            }
        else:
            # Direct summarization for shorter documents
            summary_results = {}
            
            for strategy in summarization_strategies:
                summary = self.summarization_processor.summarize(
                    document.content,
                    method=strategy
                )
                
                summary_results[strategy] = {
                    "summary": summary.summary,
                    "word_count": len(summary.summary.split())
                }
            
            results["summarization_result"] = summary_results
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        results["processing_info"]["total_processing_time"] = processing_time
        
        return results