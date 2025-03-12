"""
Processor for document summarization in SumWiseAI.

This module provides the main SummarizationProcessor class that implements
different document summarization strategies.
"""

import logging
import time
from typing import Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SummaryResult:
    """Basic class for summarization results."""
    
    def __init__(
        self,
        summary: str,
        method: str,
        processing_time: float,
        model: str,
        **kwargs
    ):
        """Initialize a summary result."""
        self.summary = summary
        self.method = method
        self.processing_time = processing_time
        self.model = model
        self.metadata = kwargs
        
    @property
    def word_count(self) -> int:
        """Return the approximate word count of the summary."""
        return len(self.summary.split())

class SummarizationProcessor:
    """
    Processor for generating summaries with multiple strategies.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize the summarization processor.
        
        Args:
            model: Default model to use for summarization
        """
        self.default_model = model
        logger.info(f"Initialized SummarizationProcessor with model {model}")
        
        # Mock responses for testing without API calls
        self.mock_responses = {
            "basic": "This is a mock basic summary of the document. It covers the main points in a concise manner.",
            "extractive": "This is a mock extractive summary. It contains the most important sentences from the original document.",
            "entity_focused": "This is a mock entity-focused summary. It focuses on key entities mentioned in the document.",
            "temporal": "This is a mock temporal summary. It organizes information chronologically."
        }
    
    def basic_summary(
        self, 
        text: str, 
        max_length: Optional[int] = None, 
        style: str = "concise"
    ) -> SummaryResult:
        """
        Generate a basic document summary.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            style: Summarization style ("concise", "detailed", "bullet", "narrative")
            
        Returns:
            SummaryResult with summary and metadata
        """
        logger.info(f"Generating {style} summary for text ({len(text)} chars)")
        start_time = time.time()
        
        # In a real implementation, we would call an LLM here.
        # For this simplified version, we'll return a mock response.
        summary = f"This is a {style} summary of the document. "
        
        if len(text) > 1000:
            summary += "The document discusses quarterly earnings, mentioning revenue and projections."
        else:
            summary += "The document is relatively short and contains high-level information."
            
        if style == "detailed":
            summary += " Additional details about product performance and market trends are included."
        elif style == "bullet":
            summary = "• Key financial results discussed\n• Product roadmap outlined\n• Market challenges addressed"
        elif style == "narrative":
            summary += " The narrative flows through financial results, then discusses products, and concludes with outlook."
        
        # Apply max_length constraint if specified
        if max_length and len(summary.split()) > max_length:
            summary = " ".join(summary.split()[:max_length])
        
        processing_time = time.time() - start_time
        
        return SummaryResult(
            summary=summary,
            method="basic",
            processing_time=processing_time,
            model=self.default_model,
            style=style
        )
    
    def extractive_summary(
        self, 
        text: str, 
        ratio: float = 0.2, 
        min_length: int = 100
    ) -> SummaryResult:
        """
        Generate an extractive summary by selecting the most important sentences.
        
        Args:
            text: Text to summarize
            ratio: Target ratio of summary to original text length
            min_length: Minimum summary length in characters
            
        Returns:
            SummaryResult with extractive summary and metadata
        """
        logger.info(f"Generating extractive summary for text ({len(text)} chars)")
        start_time = time.time()
        
        # For this simplified version, use mock response
        summary = self.mock_responses["extractive"]
        processing_time = time.time() - start_time
        
        return SummaryResult(
            summary=summary,
            method="extractive",
            processing_time=processing_time,
            model=self.default_model,
            ratio=ratio,
            min_length=min_length
        )
    
    def entity_focused_summary(
        self, 
        text: str, 
        entities: Optional[list] = None
    ) -> SummaryResult:
        """
        Generate a summary focused on specific entities.
        
        Args:
            text: Text to summarize
            entities: List of entities to focus on
            
        Returns:
            SummaryResult with entity-focused summary and metadata
        """
        logger.info(f"Generating entity-focused summary for text ({len(text)} chars)")
        start_time = time.time()
        
        # For this simplified version, use mock response
        summary = self.mock_responses["entity_focused"]
        
        # Mock entities if not provided
        if entities is None:
            entities = ["Dell", "Jeff Clarke", "Revenue", "AI", "Server Business"]
            
        processing_time = time.time() - start_time
        
        return SummaryResult(
            summary=summary,
            method="entity_focused",
            processing_time=processing_time,
            model=self.default_model,
            entities=entities,
            entity_count=len(entities)
        )
    
    def temporal_summary(
        self, 
        text: str, 
        chrono_order: bool = True
    ) -> SummaryResult:
        """
        Generate a summary organized by time periods/events.
        
        Args:
            text: Text to summarize
            chrono_order: Whether to present events in chronological order
            
        Returns:
            SummaryResult with temporal summary and metadata
        """
        logger.info(f"Generating temporal summary for text ({len(text)} chars)")
        start_time = time.time()
        
        # For this simplified version, use mock response
        summary = self.mock_responses["temporal"]
        
        # Mock time periods
        time_periods = ["Q4 2024", "FY 2024", "2023", "First Half FY 2025", "Second Half FY 2025"]
            
        processing_time = time.time() - start_time
        
        return SummaryResult(
            summary=summary,
            method="temporal",
            processing_time=processing_time,
            model=self.default_model,
            time_periods=time_periods,
            chronological_order=chrono_order
        )
    
    def summarize(
        self, 
        text: str, 
        method: str = "basic", 
        **kwargs
    ) -> SummaryResult:
        """
        Generate a summary using the specified method.
        
        Args:
            text: Text to summarize
            method: Summarization method to use
            **kwargs: Additional parameters for the specific method
            
        Returns:
            SummaryResult with summary and metadata
        """
        if method == "basic":
            return self.basic_summary(
                text=text,
                max_length=kwargs.get("max_length"),
                style=kwargs.get("style", "concise")
            )
        elif method == "extractive":
            return self.extractive_summary(
                text=text,
                ratio=kwargs.get("ratio", 0.2),
                min_length=kwargs.get("min_length", 100)
            )
        elif method == "entity_focused":
            return self.entity_focused_summary(
                text=text,
                entities=kwargs.get("entities")
            )
        elif method == "temporal":
            return self.temporal_summary(
                text=text,
                chrono_order=kwargs.get("chrono_order", True)
            )
        else:
            # For unknown methods, use a generic mock response
            logger.info(f"Using generic mock response for unknown method: {method}")
            start_time = time.time()
            
            summary = f"This is a mock {method} summary of the document."
            
            processing_time = time.time() - start_time
            
            return SummaryResult(
                summary=summary,
                method=method,
                processing_time=processing_time,
                model=self.default_model,
                **kwargs
            )