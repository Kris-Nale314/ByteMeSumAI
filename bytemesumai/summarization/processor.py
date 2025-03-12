# Copyright (c) 2025 Kris Naleszkiewicz
# Licensed under the MIT License - see LICENSE file for details
"""
Enhanced processor for document summarization in ByteMeSumAI.

This module provides the main SummarizationProcessor class that implements
different document summarization strategies with advanced capabilities for
entity tracking, temporal analysis, extractive summarization, and multi-document
comparison.
"""

import re
import logging
import time
import json
from typing import List, Dict, Any, Optional, Union, Tuple

from bytemesumai.llm.client import LLMClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SummaryResult:
    """Class for summarization results with metadata."""
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary result to a dictionary."""
        return {
            "summary": self.summary,
            "method": self.method,
            "processing_time": self.processing_time,
            "model": self.model,
            "word_count": self.word_count,
            **self.metadata
        }


class SummarizationProcessor:
    """
    Advanced processor for generating summaries with multiple strategies.
    
    Supports both mock mode for testing and real LLM-based summarization
    with various strategies like basic, extractive, entity-focused,
    temporal, multi-document, and contrastive summarization.
    """
    
    def __init__(
        self, 
        model: str = "gpt-3.5-turbo", 
        use_mock: bool = False,
        mock_responses: Optional[Dict[str, str]] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize the summarization processor.
        
        Args:
            model: Default model to use for summarization
            use_mock: Whether to use mock responses instead of LLM
            mock_responses: Custom mock responses (if using mock mode)
            llm_client: Optional custom LLM client
        """
        self.default_model = model
        self.use_mock = use_mock
        self.llm_client = llm_client or LLMClient(model=model)
        
        # Default mock responses for testing without API calls
        self._default_mock_responses = {
            "basic": "This is a mock basic summary of the document. It covers the main points in a concise manner.",
            "extractive": "This is a mock extractive summary. It contains the most important sentences from the original document.",
            "entity_focused": "This is a mock entity-focused summary. It focuses on key entities mentioned in the document.",
            "temporal": "This is a mock temporal summary. It organizes information chronologically.",
            "multi_document": "This is a mock summary comparing multiple documents. It highlights similarities and differences.",
            "contrastive": "This is a mock contrastive summary highlighting differences between documents."
        }
        
        # Use provided mock responses or defaults
        self.mock_responses = mock_responses or self._default_mock_responses
        
        logger.info(f"Initialized SummarizationProcessor with model {model}, mock mode: {use_mock}")
    
    def basic_summary(
        self, 
        text: str, 
        max_length: Optional[int] = None, 
        style: str = "concise",
        system_message: Optional[str] = None
    ) -> SummaryResult:
        """
        Generate a basic document summary.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            style: Summarization style ("concise", "detailed", "bullet", "narrative")
            system_message: Optional custom system message for the LLM
            
        Returns:
            SummaryResult with summary and metadata
        """
        logger.info(f"Generating {style} summary for text ({len(text)} chars)")
        start_time = time.time()
        
        if self.use_mock:
            # Use mock response
            summary = self.mock_responses.get("basic", self._default_mock_responses["basic"])
            
            # Add style variations
            if style == "detailed":
                summary += " Additional details about performance and trends are included."
            elif style == "bullet":
                summary = "• Key point 1\n• Key point 2\n• Key point 3"
            elif style == "narrative":
                summary += " The narrative flows naturally, telling a cohesive story."
        else:
            # Create custom instructions based on style
            style_instructions = {
                "concise": "Create a brief, high-level summary of the key points. Be concise and direct.",
                "detailed": "Provide a comprehensive summary that captures main points and important details.",
                "bullet": "Create a bullet-point summary of the key information, organized by topic.",
                "narrative": "Generate a narrative summary that flows naturally and maintains the original tone."
            }.get(style, "Create a concise summary of the main points.")
            
            # Add length constraint if specified
            length_constraint = f" Keep your summary under {max_length} words." if max_length else ""
            
            # Create system message if not provided
            if system_message is None:
                system_message = f"""You are a specialized summarization AI that creates accurate, informative summaries.
                Focus only on information present in the provided text.
                Do not add information, opinions, or analysis not present in the original."""
            
            # Build prompt
            prompt = f"""
            {style_instructions}{length_constraint}
            
            TEXT TO SUMMARIZE:
            {text}
            """
            
            try:
                # Generate summary using LLM
                summary = self.llm_client.generate_completion(
                    prompt=prompt,
                    system_message=system_message,
                    max_tokens=max_length * 3 if max_length else None  # Approximate token count
                )
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                summary = f"Error generating summary: {str(e)}"
        
        # Apply max_length constraint if specified
        if max_length and len(summary.split()) > max_length:
            summary = " ".join(summary.split()[:max_length])
        
        processing_time = time.time() - start_time
        
        return SummaryResult(
            summary=summary,
            method="basic",
            processing_time=processing_time,
            model=self.default_model,
            style=style,
            max_length=max_length
        )
    
    def extractive_summary(
        self, 
        text: str, 
        ratio: float = 0.2, 
        min_length: int = 100,
        system_message: Optional[str] = None
    ) -> SummaryResult:
        """
        Generate an extractive summary by selecting the most important sentences.
        
        Args:
            text: Text to summarize
            ratio: Target ratio of summary to original text length
            min_length: Minimum summary length in characters
            system_message: Optional custom system message for the LLM
            
        Returns:
            SummaryResult with extractive summary and metadata
        """
        logger.info(f"Generating extractive summary for text ({len(text)} chars)")
        start_time = time.time()
        
        if self.use_mock:
            # Use mock response
            summary = self.mock_responses.get("extractive", self._default_mock_responses["extractive"])
            
            # Add some metadata for mock mode
            original_sentence_count = len(re.split(r'[.!?]+', text))
            summary_sentence_count = len(re.split(r'[.!?]+', summary))
            
            return SummaryResult(
                summary=summary,
                method="extractive",
                processing_time=time.time() - start_time,
                model=self.default_model,
                ratio=ratio,
                min_length=min_length,
                original_sentences=original_sentence_count,
                summary_sentences=summary_sentence_count,
                compression_ratio=summary_sentence_count / max(1, original_sentence_count)
            )
        else:
            # Approach: Use LLM to identify important sentences
            # Create system message if not provided
            if system_message is None:
                system_message = """You are an extractive summarization engine. Your task is to identify and extract the most informative 
                sentences from the text without modifying them. Provide only the extracted sentences."""
            
            # Calculate target sentence count based on ratio
            sentences = re.split(r'[.!?]+\s+', text)
            target_sentence_count = max(3, int(len(sentences) * ratio))
            
            # Build prompt
            prompt = f"""
            Identify the {target_sentence_count} most important sentences from the following text that capture its key information.
            Return ONLY the exact sentences, exactly as they appear in the original text, with no modifications.
            Do not add your own words or explanations.
            
            TEXT TO ANALYZE:
            {text}
            """
            
            try:
                # Generate extractive summary using LLM
                extracted_sentences = self.llm_client.generate_completion(
                    prompt=prompt,
                    system_message=system_message
                )
                
                # Count original and summary sentences for metrics
                original_sentence_count = len(sentences)
                summary_sentence_count = len(re.split(r'[.!?]+', extracted_sentences))
                
                return SummaryResult(
                    summary=extracted_sentences,
                    method="extractive",
                    processing_time=time.time() - start_time,
                    model=self.default_model,
                    ratio=ratio,
                    min_length=min_length,
                    original_sentences=original_sentence_count,
                    summary_sentences=summary_sentence_count,
                    compression_ratio=summary_sentence_count / max(1, original_sentence_count)
                )
            except Exception as e:
                logger.error(f"Error generating extractive summary: {e}")
                
                return SummaryResult(
                    summary=f"Error generating extractive summary: {str(e)}",
                    method="extractive",
                    processing_time=time.time() - start_time,
                    model=self.default_model,
                    ratio=ratio,
                    min_length=min_length,
                    error=str(e)
                )
    
    def entity_focused_summary(
        self, 
        text: str, 
        entities: Optional[List[str]] = None,
        system_message: Optional[str] = None
    ) -> SummaryResult:
        """
        Generate a summary focused on specific entities.
        
        Args:
            text: Text to summarize
            entities: List of entities to focus on (detected automatically if None)
            system_message: Optional custom system message for the LLM
            
        Returns:
            SummaryResult with entity-focused summary and metadata
        """
        logger.info(f"Generating entity-focused summary for text ({len(text)} chars)")
        start_time = time.time()
        
        # If entities not provided, detect them first
        if entities is None:
            entities = self._extract_key_entities(text)
        
        if not entities:
            logger.warning("No entities found for entity-focused summary")
            return SummaryResult(
                summary="No entities identified in the text for entity-focused summary.",
                method="entity_focused",
                processing_time=time.time() - start_time,
                model=self.default_model,
                entities=[],
                entity_count=0,
                error="No entities identified"
            )
        
        if self.use_mock:
            # Use mock response
            summary = self.mock_responses.get("entity_focused", self._default_mock_responses["entity_focused"])
            summary += f" Focused on these entities: {', '.join(entities[:3])}..."
            
            return SummaryResult(
                summary=summary,
                method="entity_focused",
                processing_time=time.time() - start_time,
                model=self.default_model,
                entities=entities,
                entity_count=len(entities)
            )
        else:
            # Create system message if not provided
            if system_message is None:
                system_message = """You are a specialized entity-focused summarization AI. Your summaries organize information 
                by key entities while maintaining factual accuracy. Focus only on information present 
                in the source text."""
            
            # Build prompt
            entities_list = ", ".join(entities)
            prompt = f"""
            Create a summary of the text that focuses on these key entities: {entities_list}.
            
            For each entity, summarize the most important information mentioned about it.
            Organize your summary by entity, with clear headings for each one.
            Include only information that is explicitly mentioned in the text.
            
            TEXT TO SUMMARIZE:
            {text}
            """
            
            try:
                # Generate entity-focused summary using LLM
                summary = self.llm_client.generate_completion(
                    prompt=prompt,
                    system_message=system_message
                )
                
                return SummaryResult(
                    summary=summary,
                    method="entity_focused",
                    processing_time=time.time() - start_time,
                    model=self.default_model,
                    entities=entities,
                    entity_count=len(entities)
                )
            except Exception as e:
                logger.error(f"Error generating entity-focused summary: {e}")
                
                return SummaryResult(
                    summary=f"Error generating entity-focused summary: {str(e)}",
                    method="entity_focused",
                    processing_time=time.time() - start_time,
                    model=self.default_model,
                    entities=entities,
                    entity_count=len(entities),
                    error=str(e)
                )
    
    def _extract_key_entities(self, text: str, max_entities: int = 5) -> List[str]:
        """
        Extract key entities from text.
        
        Args:
            text: Text to analyze
            max_entities: Maximum number of entities to extract
            
        Returns:
            List of key entity names
        """
        if self.use_mock:
            # Return mock entities
            return ["Entity1", "Entity2", "Entity3", "Entity4", "Entity5"]
        
        try:
            # Use a simple LLM approach for entity extraction
            prompt = f"""
            Identify the {max_entities} most important named entities (people, organizations, products, locations) 
            in the following text. Return only a comma-separated list of entity names, with no additional text.
            
            TEXT TO ANALYZE:
            {text[:6000] if len(text) > 6000 else text}
            """
            
            system_message = """
            You are an entity recognition specialist. Extract only the most important named entities 
            from the text. Return only the entity names as a comma-separated list.
            """
            
            result = self.llm_client.generate_completion(
                prompt=prompt,
                system_message=system_message
            )
            
            # Clean up and parse result
            result = result.strip()
            entities = [entity.strip() for entity in result.split(",") if entity.strip()]
            
            return entities[:max_entities]
        except Exception as e:
            logger.warning(f"Error extracting entities: {e}")
            return []
    
    def temporal_summary(
        self, 
        text: str, 
        chrono_order: bool = True,
        system_message: Optional[str] = None
    ) -> SummaryResult:
        """
        Generate a summary organized by time periods/events.
        
        Args:
            text: Text to summarize
            chrono_order: Whether to present events in chronological order
            system_message: Optional custom system message for the LLM
            
        Returns:
            SummaryResult with temporal summary and metadata
        """
        logger.info(f"Generating temporal summary for text ({len(text)} chars)")
        start_time = time.time()
        
        if self.use_mock:
            # Use mock response
            summary = self.mock_responses.get("temporal", self._default_mock_responses["temporal"])
            time_periods = ["Period1", "Period2", "Period3", "Period4"]
            
            return SummaryResult(
                summary=summary,
                method="temporal",
                processing_time=time.time() - start_time,
                model=self.default_model,
                time_periods=time_periods,
                chronological_order=chrono_order,
                period_count=len(time_periods)
            )
        else:
            # Create system message if not provided
            if system_message is None:
                system_message = """You are a specialized temporal summarization AI. Your summaries organize information 
                by time periods and events, maintaining a clear temporal structure while preserving 
                factual accuracy."""
            
            # Build prompt
            time_order = "chronological" if chrono_order else "reverse chronological"
            prompt = f"""
            Create a summary of the text organized by time periods or key events.
            
            Identify the main time periods, dates, or temporal events in the text.
            Organize the information in {time_order} order.
            For each time period or event, summarize the relevant information.
            Use clear temporal markers and headings in your summary.
            
            TEXT TO SUMMARIZE:
            {text}
            """
            
            try:
                # Generate temporal summary using LLM
                summary = self.llm_client.generate_completion(
                    prompt=prompt,
                    system_message=system_message
                )
                
                # Extract time periods mentioned
                time_periods = self._extract_time_periods(summary)
                
                return SummaryResult(
                    summary=summary,
                    method="temporal",
                    processing_time=time.time() - start_time,
                    model=self.default_model,
                    time_periods=time_periods,
                    chronological_order=chrono_order,
                    period_count=len(time_periods)
                )
            except Exception as e:
                logger.error(f"Error generating temporal summary: {e}")
                
                return SummaryResult(
                    summary=f"Error generating temporal summary: {str(e)}",
                    method="temporal",
                    processing_time=time.time() - start_time,
                    model=self.default_model,
                    time_periods=[],
                    chronological_order=chrono_order,
                    period_count=0,
                    error=str(e)
                )
    
    def _extract_time_periods(self, text: str) -> List[str]:
        """
        Extract time periods mentioned in a summary.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of time periods
        """
        # Simple regex-based approach
        date_pattern = r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})'
        year_pattern = r'\b((?:19|20)\d{2})\b'
        quarter_pattern = r'\b(Q[1-4]\s+\d{4}|(?:first|second|third|fourth) quarter of \d{4})'
        
        # Find all matches
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        years = re.findall(year_pattern, text)
        quarters = re.findall(quarter_pattern, text, re.IGNORECASE)
        
        # Combine and remove duplicates
        all_periods = list(set(dates + years + quarters))
        
        return all_periods
    
    def multi_document_summary(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        focus: str = "comparative",
        query: Optional[str] = None,
        system_message: Optional[str] = None
    ) -> SummaryResult:
        """
        Generate a summary comparing multiple documents.
        
        Args:
            documents: List of document texts or document objects with content and metadata
            focus: Focus type ("comparative", "integrative", "contrastive")
            query: Optional query to focus the summary on specific aspects
            system_message: Optional custom system message for the LLM
            
        Returns:
            SummaryResult with multi-document summary and metadata
        """
        logger.info(f"Generating multi-document summary for {len(documents)} documents with {focus} focus")
        start_time = time.time()
        
        # Process document inputs to get text and metadata
        processed_docs = []
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                # If document is a string, use it as content with a generic title
                processed_docs.append({
                    "content": doc,
                    "title": f"Document {i+1}",
                    "length": len(doc)
                })
            elif isinstance(doc, dict):
                # If document is a dict, extract content and metadata
                processed_docs.append({
                    "content": doc.get("content", doc.get("text", "")),
                    "title": doc.get("title", doc.get("filename", f"Document {i+1}")),
                    "length": len(doc.get("content", doc.get("text", "")))
                })
            else:
                # Skip invalid documents
                logger.warning(f"Skipping invalid document at index {i}: {type(doc)}")
                continue
        
        if not processed_docs:
            return SummaryResult(
                summary="No valid documents provided for multi-document summary.",
                method="multi_document",
                processing_time=time.time() - start_time,
                model=self.default_model,
                focus=focus,
                query=query,
                document_count=0,
                is_query_focused=query is not None
            )
        
        if self.use_mock:
            # Use mock response
            summary = self.mock_responses.get("multi_document", self._default_mock_responses["multi_document"])
            
            if query:
                summary += f" Focused on query: '{query}'."
            
            return SummaryResult(
                summary=summary,
                method="multi_document",
                processing_time=time.time() - start_time,
                model=self.default_model,
                focus=focus,
                query=query,
                document_count=len(processed_docs),
                document_titles=[doc["title"] for doc in processed_docs],
                is_query_focused=query is not None
            )
        else:
            # If texts are too long, summarize each one first
            combined_text = ""
            if sum(doc["length"] for doc in processed_docs) > 8000:
                # Summarize each document individually first
                for i, doc in enumerate(processed_docs):
                    doc_summary = self.basic_summary(doc["content"], style="concise").summary
                    combined_text += f"DOCUMENT {i+1}: {doc['title']}\n{doc_summary}\n\n"
            else:
                # Combine texts with clear document markers
                for i, doc in enumerate(processed_docs):
                    # Limit each document to 3000 chars if very long
                    excerpt_length = min(doc["length"], 3000)
                    excerpt = doc["content"][:excerpt_length]
                    combined_text += f"DOCUMENT {i+1}: {doc['title']}\n{excerpt}\n\n"
            
            # Create system message if not provided
            if system_message is None:
                system_message = """You are a specialized multi-document summarization AI. Your task is to synthesize information 
                across multiple documents while maintaining factual accuracy and highlighting important 
                relationships between documents."""
            
            # Build prompt based on focus and query
            if query:
                prompt = f"""
                Create a summary of the following documents that answers this query: "{query}"
                
                Focus on information relevant to the query, while noting any contradictions or differences 
                between documents. Maintain factual accuracy and cite document numbers when appropriate.
                
                DOCUMENTS:
                {combined_text}
                """
            else:
                focus_instructions = {
                    "comparative": "Highlight similarities and differences between documents. Compare and contrast key information.",
                    "integrative": "Synthesize information across documents into a unified summary. Focus on creating a coherent narrative.",
                    "contrastive": "Emphasize differences and opposing viewpoints between documents. Highlight contradictions."
                }.get(focus, "Highlight key information from all documents.")
                
                prompt = f"""
                Create a {focus} summary of the following documents.
                
                {focus_instructions}
                
                Include:
                1. Key points shared across multiple documents
                2. Important unique information from individual documents
                3. Any contradictions or differences between documents
                
                Maintain factual accuracy and cite document numbers when appropriate.
                
                DOCUMENTS:
                {combined_text}
                """
            
            try:
                # Generate multi-document summary using LLM
                summary = self.llm_client.generate_completion(
                    prompt=prompt,
                    system_message=system_message
                )
                
                return SummaryResult(
                    summary=summary,
                    method="multi_document",
                    processing_time=time.time() - start_time,
                    model=self.default_model,
                    focus=focus,
                    query=query,
                    document_count=len(processed_docs),
                    document_titles=[doc["title"] for doc in processed_docs],
                    is_query_focused=query is not None
                )
            except Exception as e:
                logger.error(f"Error generating multi-document summary: {e}")
                
                return SummaryResult(
                    summary=f"Error generating multi-document summary: {str(e)}",
                    method="multi_document",
                    processing_time=time.time() - start_time,
                    model=self.default_model,
                    focus=focus,
                    query=query,
                    document_count=len(processed_docs),
                    document_titles=[doc["title"] for doc in processed_docs],
                    is_query_focused=query is not None,
                    error=str(e)
                )
    
    def contrastive_summary(
        self, 
        text1: str, 
        text2: str, 
        focus: Optional[str] = None,
        system_message: Optional[str] = None
    ) -> SummaryResult:
        """
        Generate a summary that contrasts two texts.
        
        Args:
            text1: First text
            text2: Second text
            focus: Optional aspect to focus the comparison on
            system_message: Optional custom system message for the LLM
            
        Returns:
            SummaryResult with contrastive summary and metadata
        """
        logger.info(f"Generating contrastive summary between two texts")
        start_time = time.time()
        
        if self.use_mock:
            # Use mock response
            summary = self.mock_responses.get("contrastive", self._default_mock_responses["contrastive"])
            
            if focus:
                summary += f" Focus on {focus}."
            
            return SummaryResult(
                summary=summary,
                method="contrastive",
                processing_time=time.time() - start_time,
                model=self.default_model,
                focus=focus
            )
        else:
            # Create system message if not provided
            if system_message is None:
                system_message = """You are a specialized contrastive summarization AI. Your task is to compare and contrast 
                different texts, highlighting similarities and differences while maintaining factual accuracy."""
            
            # Build prompt
            focus_instruction = f"Focus your comparison on {focus}." if focus else ""
            prompt = f"""
            Create a summary that compares and contrasts the following two texts.
            
            {focus_instruction}
            
            Highlight:
            1. Key similarities between the texts
            2. Important differences between the texts
            3. Unique information provided by each text
            
            TEXT 1:
            {text1}
            
            TEXT 2:
            {text2}
            """
            
            try:
                # Generate contrastive summary using LLM
                summary = self.llm_client.generate_completion(
                    prompt=prompt,
                    system_message=system_message
                )
                
                return SummaryResult(
                    summary=summary,
                    method="contrastive",
                    processing_time=time.time() - start_time,
                    model=self.default_model,
                    focus=focus
                )
            except Exception as e:
                logger.error(f"Error generating contrastive summary: {e}")
                
                return SummaryResult(
                    summary=f"Error generating contrastive summary: {str(e)}",
                    method="contrastive",
                    processing_time=time.time() - start_time,
                    model=self.default_model,
                    focus=focus,
                    error=str(e)
                )
    
    def evaluate_summary(
        self, 
        original_text: str, 
        summary: str,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a summary.
        
        Args:
            original_text: Original text
            summary: Summary to evaluate
            system_message: Optional custom system message for the LLM
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating summary quality")
        start_time = time.time()
        
        if self.use_mock:
            # Return mock evaluation
            return {
                "completeness_score": 8.5,
                "conciseness_score": 9.0,
                "accuracy_score": 9.5,
                "coherence_score": 8.0,
                "overall_score": 8.75,
                "factual_errors": [],
                "important_omissions": ["Some minor details were omitted"],
                "strengths": ["Captures key points", "Well organized"],
                "improvement_suggestions": ["Could include more context"],
                "processing_time": time.time() - start_time,
                "original_length": len(original_text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / max(1, len(original_text))
            }
        else:
            # Create system message if not provided
            if system_message is None:
                system_message = """You are a specialized summary evaluation AI. Your task is to objectively evaluate the quality of summaries 
                based on completeness, conciseness, accuracy, and coherence. Provide your evaluation in the requested JSON format."""
            
            # Build prompt for evaluation
            prompt = f"""
            Evaluate the quality of the following summary based on the original text.
            
            Please assess the summary on these dimensions, scoring each from 1-10:
            1. Completeness: Does it include all important information?
            2. Conciseness: Is it appropriately brief without unnecessary details?
            3. Accuracy: Does it contain only factual information from the original?
            4. Coherence: Does it flow logically and is it well-organized?
            
            Also identify:
            - Any factual errors or hallucinations (information not in the original)
            - Any important information that was omitted
            
            Return your evaluation as a JSON object with these fields:
            {
              "completeness_score": [1-10],
              "conciseness_score": [1-10], 
              "accuracy_score": [1-10],
              "coherence_score": [1-10],
              "overall_score": [1-10],
              "factual_errors": ["error1", "error2", ...],
              "important_omissions": ["omission1", "omission2", ...],
              "strengths": ["strength1", "strength2", ...],
              "improvement_suggestions": ["suggestion1", "suggestion2", ...]
            }
            
            ORIGINAL TEXT:
            {original_text}
            
            SUMMARY TO EVALUATE:
            {summary}
            """
            
            try:
                # Generate evaluation using LLM
                evaluation_result = self.llm_client.generate_completion(
                    prompt=prompt,
                    system_message=system_message
                )
                
                # Parse JSON from response
                evaluation_json = self._extract_json(evaluation_result)
                
                # Add additional metadata
                evaluation_json["processing_time"] = time.time() - start_time
                evaluation_json["original_length"] = len(original_text)
                evaluation_json["summary_length"] = len(summary)
                evaluation_json["compression_ratio"] = len(summary) / max(1, len(original_text))
                
                return evaluation_json
            except Exception as e:
                logger.error(f"Error evaluating summary: {e}")
                
                return {
                    "error": str(e),
                    "processing_time": time.time() - start_time,
                    "original_length": len(original_text),
                    "summary_length": len(summary)
                }
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON object from text.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Extracted JSON object or empty dict if not found
        """
        # Find JSON-like patterns
        json_pattern = r'({[\s\S]*})'
        match = re.search(json_pattern, text)
        
        if match:
            try:
                # Try to parse the JSON
                json_str = match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common JSON format issues
                try:
                    # Replace single quotes with double quotes
                    fixed_json = json_str.replace("'", '"')
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass
        
        # If JSON parsing failed, return empty dict
        return {}
    
    def compare_summaries(
        self, 
        text: str, 
        methods: List[str] = None, 
        evaluate: bool = True
    ) -> Dict[str, Any]:
        """
        Compare different summarization methods on the same text.
        
        Args:
            text: Text to summarize
            methods: List of summarization methods to compare
            evaluate: Whether to evaluate summaries
            
        Returns:
            Dictionary with results for each method and comparison
        """
        methods = methods or ["basic", "extractive", "entity_focused", "temporal"]
        logger.info(f"Comparing summarization methods ({', '.join(methods)}) on text ({len(text)} chars)")
        
        summaries = {}
        evaluations = {}
        
        # Generate summaries with each method
        for method in methods:
            try:
                if method == "basic":
                    result = self.basic_summary(text)
                elif method == "extractive":
                    result = self.extractive_summary(text)
                elif method == "entity_focused":
                    result = self.entity_focused_summary(text)
                elif method == "temporal":
                    result = self.temporal_summary(text)
                else:
                    logger.warning(f"Unknown summarization method: {method}, skipping")
                    continue
                
                summaries[method] = result
                
                # Evaluate if requested
                if evaluate:
                    evaluation = self.evaluate_summary(text, result.summary)
                    evaluations[method] = evaluation
            except Exception as e:
                logger.error(f"Error with method {method}: {e}")
                summaries[method] = SummaryResult(
                    summary=f"Error with method {method}: {str(e)}",
                    method=method,
                    processing_time=0.0,
                    model=self.default_model,
                    error=str(e)
                )
        
        # Compare summaries
        comparison = self._compare_summary_results(summaries, evaluations)
        
        return {
            "text_length": len(text),
            "methods": methods,
            "summaries": {k: v.to_dict() for k, v in summaries.items()},
            "evaluations": evaluations,
            "comparison": comparison
        }
    
    def _compare_summary_results(
        self, 
        summaries: Dict[str, SummaryResult], 
        evaluations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comparison metrics between summary methods.
        
        Args:
            summaries: Dictionary of summary results by method
            evaluations: Dictionary of evaluation results by method
            
        Returns:
            Dictionary of comparison metrics
        """
        comparison = {
            "length_comparison": {},
            "processing_time_comparison": {},
            "rankings": {}
        }
        
        # Compare summary lengths
        for method, result in summaries.items():
            comparison["length_comparison"][method] = result.word_count
        
        # Compare processing times
        for method, result in summaries.items():
            comparison["processing_time_comparison"][method] = result.processing_time
        
        # Generate rankings based on evaluations
        if evaluations:
            for metric in ["completeness_score", "conciseness_score", "accuracy_score", 
                          "coherence_score", "overall_score"]:
                # Get methods with this metric
                methods_with_metric = []
                for method, eval_result in evaluations.items():
                    if metric in eval_result:
                        methods_with_metric.append((method, eval_result[metric]))
                
                # Sort by score (descending)
                methods_with_metric.sort(key=lambda x: x[1], reverse=True)
                
                # Add to rankings
                comparison["rankings"][metric] = [
                    {"method": method, "score": score}
                    for method, score in methods_with_metric
                ]
        
        # Find best method for different use cases
        best_methods = {
            "best_for_completeness": None,
            "best_for_conciseness": None,
            "best_for_accuracy": None,
            "best_overall": None
        }
        
        if evaluations:
            # Find best method for completeness
            completeness_scores = [(method, eval_result.get("completeness_score", 0)) 
                                 for method, eval_result in evaluations.items()]
            if completeness_scores:
                best_methods["best_for_completeness"] = max(completeness_scores, key=lambda x: x[1])[0]
            
            # Find best method for conciseness
            conciseness_scores = [(method, eval_result.get("conciseness_score", 0)) 
                                for method, eval_result in evaluations.items()]
            if conciseness_scores:
                best_methods["best_for_conciseness"] = max(conciseness_scores, key=lambda x: x[1])[0]
            
            # Find best method for accuracy
            accuracy_scores = [(method, eval_result.get("accuracy_score", 0)) 
                             for method, eval_result in evaluations.items()]
            if accuracy_scores:
                best_methods["best_for_accuracy"] = max(accuracy_scores, key=lambda x: x[1])[0]
            
            # Find best method overall
            overall_scores = [(method, eval_result.get("overall_score", 0)) 
                            for method, eval_result in evaluations.items()]
            if overall_scores:
                best_methods["best_overall"] = max(overall_scores, key=lambda x: x[1])[0]
        
        comparison["best_methods"] = best_methods
        
        return comparison
    
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
                style=kwargs.get("style", "concise"),
                system_message=kwargs.get("system_message")
            )
        elif method == "extractive":
            return self.extractive_summary(
                text=text,
                ratio=kwargs.get("ratio", 0.2),
                min_length=kwargs.get("min_length", 100),
                system_message=kwargs.get("system_message")
            )
        elif method == "entity_focused":
            return self.entity_focused_summary(
                text=text,
                entities=kwargs.get("entities"),
                system_message=kwargs.get("system_message")
            )
        elif method == "temporal":
            return self.temporal_summary(
                text=text,
                chrono_order=kwargs.get("chrono_order", True),
                system_message=kwargs.get("system_message")
            )
        else:
            # For unknown methods, use a generic mock response or basic summary
            logger.warning(f"Unknown summarization method: {method}. Using basic summary instead.")
            return self.basic_summary(text=text, style="concise")


# Module-level convenience function
def summarize(text: str, method: str = "basic", **kwargs) -> SummaryResult:
    """
    Generate a summary using the specified method (convenience function).
    
    Args:
        text: Text to summarize
        method: Summarization method to use
        **kwargs: Additional parameters for the specific method
        
    Returns:
        SummaryResult with summary and metadata
    """
    processor = SummarizationProcessor()
    return processor.summarize(text, method, **kwargs)