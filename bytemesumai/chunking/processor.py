"""
Processor for document chunking in ByteMeSumAI.

This module provides the main ChunkingProcessor class that implements
different document chunking strategies.
"""

import re
import logging
import nltk
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from bytemesumai.chunking.models import Chunk, DocumentBoundary, ChunkingResult
from bytemesumai.models.document import Document
from bytemesumai.llm.client import LLMClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ChunkingProcessor:
    """
    Processor for chunking documents using various strategies.
    Supports traditional, boundary-aware, and semantic chunking approaches.
    """
    
    def __init__(
        self, 
        default_chunk_size: int = 1000, 
        default_chunk_overlap: int = 200,
        boundary_detection_threshold: float = 0.7,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize the chunking processor.
        
        Args:
            default_chunk_size: Default size for fixed-size chunks (characters)
            default_chunk_overlap: Default overlap between chunks (characters)
            boundary_detection_threshold: Confidence threshold for boundary detection
            llm_client: Optional LLM client to use (creates one if not provided)
        """
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.boundary_detection_threshold = boundary_detection_threshold
        self.llm_client = llm_client or LLMClient()
        
        logger.info(f"Initialized ChunkingProcessor with default chunk size {default_chunk_size}, " 
                   f"overlap {default_chunk_overlap}")
    
    def chunk_text_fixed_size(
        self, 
        text: str, 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None
    ) -> List[Chunk]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: The text to chunk
            chunk_size: Size of each chunk in characters (uses default if None)
            chunk_overlap: Overlap between chunks in characters (uses default if None)
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        # Use defaults if not specified
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap
        
        logger.info(f"Chunking text ({len(text)} chars) with fixed-size strategy: "
                   f"size={chunk_size}, overlap={chunk_overlap}")
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position, ensuring we don't go beyond text length
            end = min(start + chunk_size, len(text))
            
            # If we're not at the start and not at the end, try to find a better break point
            if start > 0 and end < len(text):
                # Find the last sentence-ending punctuation
                last_period = max(text.rfind('. ', start, end), 
                                 text.rfind('? ', start, end), 
                                 text.rfind('! ', start, end))
                
                # If found and reasonably positioned, use it as the end point
                if last_period != -1 and last_period > start + (chunk_size / 2):
                    end = last_period + 2  # Include the punctuation and space
            
            # Create the chunk
            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start,
                end_idx=end,
                metadata={"strategy": "fixed_size"}
            ))
            
            # Move start position for next chunk, accounting for overlap
            start = end - chunk_overlap
            
            # Ensure we make progress even with large overlaps
            if start <= chunks[-1].start_idx:
                start = chunks[-1].start_idx + 1
        
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks
    
    def detect_document_boundaries(self, text: str) -> List[DocumentBoundary]:
        """
        Detect natural document boundaries in text.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of DocumentBoundary objects
        """
        logger.info(f"Detecting document boundaries in text ({len(text)} chars)")
        
        boundaries = []
        
        # 1. Find paragraph breaks (double newlines)
        paragraph_pattern = r'\n\s*\n'
        for match in re.finditer(paragraph_pattern, text):
            position = match.start()
            # Get context (20 chars before and after)
            context_start = max(0, position - 20)
            context_end = min(len(text), position + 20)
            context = text[context_start:context_end]
            
            boundaries.append(DocumentBoundary(
                position=position,
                boundary_type="paragraph",
                confidence=0.8,  # High confidence for clear paragraph breaks
                context=context
            ))
        
        # 2. Find section headers
        # Pattern for common section headers (e.g., "## Section Title")
        header_patterns = [
            (r'#+\s+[A-Z]', 0.9),  # Markdown headers
            (r'\n[A-Z][A-Z\s]+\n', 0.8),  # ALL CAPS HEADERS
            (r'\n\d+\.\s+[A-Z]', 0.7)  # Numbered sections
        ]
        
        for pattern, confidence in header_patterns:
            for match in re.finditer(pattern, text):
                position = match.start()
                # Get context
                context_start = max(0, position - 20)
                context_end = min(len(text), position + 20)
                context = text[context_start:context_end]
                
                boundaries.append(DocumentBoundary(
                    position=position,
                    boundary_type="section",
                    confidence=confidence,
                    context=context
                ))
        
        # 3. Find format shifts (e.g., from prose to a list)
        format_shifts = [
            (r'\n\s*[-*•]\s', 0.7),  # Start of a list
            (r'\n\s*\d+\.\s', 0.7),  # Start of a numbered list
            (r'\n\s*```', 0.9),      # Code block
            (r'\n\s*\|', 0.8)        # Table
        ]
        
        for pattern, confidence in format_shifts:
            for match in re.finditer(pattern, text):
                position = match.start()
                # Get context
                context_start = max(0, position - 20)
                context_end = min(len(text), position + 20)
                context = text[context_start:context_end]
                
                boundaries.append(DocumentBoundary(
                    position=position,
                    boundary_type="format_shift",
                    confidence=confidence,
                    context=context
                ))
        
        # 4. Entity transitions (e.g., suddenly talking about a different company)
        company_transitions = r'\n[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|Company)'
        
        for match in re.finditer(company_transitions, text):
            position = match.start()
            # Get context
            context_start = max(0, position - 20)
            context_end = min(len(text), position + 20)
            context = text[context_start:context_end]
            
            boundaries.append(DocumentBoundary(
                position=position,
                boundary_type="entity_transition",
                confidence=0.6,  # Lower confidence as this is a simple heuristic
                context=context
            ))
        
        # 5. Temporal shifts (e.g., switching time periods)
        date_transitions = r'\n(?:In|On|During|As of) (?:January|February|March|April|May|June|July|August|September|October|November|December)'
        
        for match in re.finditer(date_transitions, text):
            position = match.start()
            # Get context
            context_start = max(0, position - 20)
            context_end = min(len(text), position + 20)
            context = text[context_start:context_end]
            
            boundaries.append(DocumentBoundary(
                position=position,
                boundary_type="temporal_shift",
                confidence=0.6,
                context=context
            ))
            
        # Use LLM for advanced boundary detection
        try:
            llm_boundaries = self._detect_boundaries_with_llm(text)
            if llm_boundaries:
                # Add LLM-detected boundaries with higher confidence
                for b in llm_boundaries:
                    boundaries.append(DocumentBoundary(
                        position=b["position"],
                        boundary_type=b["type"],
                        confidence=b.get("confidence", 0.85),  # LLM boundaries get high confidence
                        context=b.get("context", "")
                    ))
        except Exception as e:
            logger.warning(f"Error detecting boundaries with LLM: {e}")
            
        # Sort boundaries by position
        boundaries.sort(key=lambda x: x.position)
        
        logger.info(f"Detected {len(boundaries)} potential document boundaries")
        return boundaries
    
    def _detect_boundaries_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """
        Use LLM to detect document boundaries.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of boundary dictionaries with position, type, and context
        """
        # Only analyze if text is long enough to potentially have boundaries
        if len(text) < 1000:
            return []
            
        # Sample the text if it's too long
        if len(text) > 8000:
            # Take samples from beginning, middle, and end
            sample_size = 2500
            beginning = text[:sample_size]
            middle_start = max(0, len(text)//2 - sample_size//2)
            middle = text[middle_start:middle_start+sample_size]
            end = text[-sample_size:]
            
            # Clear markers for the samples
            sample_text = f"[BEGINNING]\n{beginning}\n\n[MIDDLE]\n{middle}\n\n[END]\n{end}"
        else:
            sample_text = text
            
        prompt = """
        Analyze the following text and identify potential document boundaries (places where one document ends and another begins).
        
        Look for:
        - Topic shifts
        - Style changes
        - Entity switches (different companies/people being discussed)
        - Temporal shifts (different time periods)
        - Format changes
        
        Return a JSON list of boundary objects with these properties:
        - position: approximate character position in the original text
        - type: the boundary type (topic_shift, entity_change, temporal_shift, format_change, etc.)
        - confidence: your confidence level (0.0-1.0)
        - context: a short snippet showing the boundary context
        
        TEXT TO ANALYZE:
        """
        
        try:
            result = self.llm_client.generate_completion(prompt=prompt + sample_text)
            
            # Extract JSON part of the response
            import json
            import re
            
            # Find JSON in the response (handling potential explanatory text)
            json_pattern = r'\[\s*\{.*\}\s*\]'
            json_match = re.search(json_pattern, result, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                boundaries = json.loads(json_str)
                
                # Adjust positions if we used samples
                if len(text) > 8000:
                    for b in boundaries:
                        if "[BEGINNING]" in b.get("context", ""):
                            # Position is already correct for beginning
                            pass
                        elif "[MIDDLE]" in b.get("context", ""):
                            b["position"] += len(text)//2 - 2500//2
                        elif "[END]" in b.get("context", ""):
                            b["position"] += len(text) - 2500
                
                return boundaries
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error processing LLM boundary detection: {e}")
            return []
    
    def chunk_text_boundary_aware(
        self, 
        text: str, 
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500
    ) -> List[Chunk]:
        """
        Split text into chunks based on detected natural boundaries.
        
        Args:
            text: The text to chunk
            min_chunk_size: Minimum size for chunks in characters
            max_chunk_size: Maximum size for chunks in characters
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        logger.info(f"Chunking text ({len(text)} chars) with boundary-aware strategy")
        
        # Detect boundaries
        boundaries = self.detect_document_boundaries(text)
        
        # Filter boundaries by confidence threshold
        high_confidence_boundaries = [
            b for b in boundaries 
            if b.confidence >= self.boundary_detection_threshold
        ]
        
        logger.info(f"Using {len(high_confidence_boundaries)} high-confidence boundaries " 
                   f"out of {len(boundaries)} detected")
        
        # If no high-confidence boundaries, fall back to fixed-size chunking
        if not high_confidence_boundaries:
            logger.warning("No high-confidence boundaries found, falling back to fixed-size chunking")
            return self.chunk_text_fixed_size(text)
        
        # Create chunks based on boundaries
        chunks = []
        
        # Add text start and end as boundary positions
        positions = [0] + [b.position for b in high_confidence_boundaries] + [len(text)]
        boundary_types = ["document_start"] + [b.boundary_type for b in high_confidence_boundaries] + ["document_end"]
        
        # Process each segment between boundaries
        for i in range(len(positions) - 1):
            segment_start = positions[i]
            segment_end = positions[i+1]
            segment_length = segment_end - segment_start
            
            # If segment is larger than max_chunk_size, subdivide it
            if segment_length > max_chunk_size:
                logger.info(f"Segment {i} exceeds max size ({segment_length} > {max_chunk_size}), subdividing")
                
                # Use sentence-aware subdivision
                segment_text = text[segment_start:segment_end]
                
                # Try to split at sentence boundaries
                try:
                    sentences = nltk.sent_tokenize(segment_text)
                    current_chunk_text = ""
                    current_chunk_start = segment_start
                    
                    for sentence in sentences:
                        # If adding this sentence would exceed max_chunk_size, create a chunk and start a new one
                        if len(current_chunk_text) + len(sentence) > max_chunk_size and len(current_chunk_text) > min_chunk_size:
                            # Add current chunk
                            current_chunk_end = current_chunk_start + len(current_chunk_text)
                            chunks.append(Chunk(
                                text=current_chunk_text,
                                start_idx=current_chunk_start,
                                end_idx=current_chunk_end,
                                metadata={
                                    "strategy": "boundary_aware_subdivided",
                                    "boundary_type": boundary_types[i],
                                    "subdivision": "sentence"
                                }
                            ))
                            
                            # Start new chunk
                            current_chunk_text = sentence
                            current_chunk_start = current_chunk_end
                        else:
                            # Add sentence to current chunk
                            current_chunk_text += sentence
                    
                    # Add the last chunk if it has content
                    if current_chunk_text:
                        chunks.append(Chunk(
                            text=current_chunk_text,
                            start_idx=current_chunk_start,
                            end_idx=current_chunk_start + len(current_chunk_text),
                            metadata={
                                "strategy": "boundary_aware_subdivided",
                                "boundary_type": boundary_types[i],
                                "subdivision": "sentence"
                            }
                        ))
                        
                except Exception as e:
                    logger.warning(f"Error in sentence-aware subdivision: {e}. Falling back to fixed-size.")
                    # Fall back to fixed-size subdivision
                    segment_chunks = self.chunk_text_fixed_size(
                        text[segment_start:segment_end],
                        chunk_size=max_chunk_size,
                        chunk_overlap=min(200, max_chunk_size // 5)
                    )
                    
                    # Adjust indices to be relative to the original text
                    for chunk in segment_chunks:
                        chunk.start_idx += segment_start
                        chunk.end_idx += segment_start
                        chunk.metadata = {
                            "strategy": "boundary_aware_subdivided",
                            "boundary_type": boundary_types[i],
                            "subdivision": "fixed_size"
                        }
                    
                    chunks.extend(segment_chunks)
            
            # If segment is smaller than min_chunk_size and not the last segment,
            # consider merging with the next segment
            elif segment_length < min_chunk_size and i < len(positions) - 2:
                # Look ahead to see if combining with next segment is reasonable
                next_segment_length = positions[i+2] - positions[i+1]
                
                if segment_length + next_segment_length <= max_chunk_size:
                    logger.info(f"Segment {i} below min size ({segment_length} < {min_chunk_size}), "
                               f"will merge with next segment")
                    continue
                else:
                    # Small segment but merging not feasible, keep as is
                    chunk_text = text[segment_start:segment_end]
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_idx=segment_start,
                        end_idx=segment_end,
                        metadata={
                            "strategy": "boundary_aware",
                            "boundary_type": boundary_types[i]
                        }
                    ))
            
            # Otherwise, create a chunk for this segment
            else:
                chunk_text = text[segment_start:segment_end]
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=segment_start,
                    end_idx=segment_end,
                    metadata={
                        "strategy": "boundary_aware",
                        "boundary_type": boundary_types[i]
                    }
                ))
        
        logger.info(f"Created {len(chunks)} boundary-aware chunks")
        return chunks
    
    def chunk_text_semantic(self, text: str, target_chunk_count: Optional[int] = None) -> List[Chunk]:
        """
        Split text into semantically coherent chunks using LLM assistance.
        
        Args:
            text: The text to chunk
            target_chunk_count: Approximately how many chunks to create
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
            
        logger.info(f"Chunking text ({len(text)} chars) with semantic strategy")
        
        # For very long texts, fall back to boundary-aware chunking first
        if len(text) > 8000:
            logger.info("Text too long for direct semantic chunking, using boundary-aware chunking first")
            initial_chunks = self.chunk_text_boundary_aware(text)
            
            # Now apply semantic chunking to each large chunk if needed
            result_chunks = []
            for chunk in initial_chunks:
                if chunk.length > 4000:
                    # Further chunk this large piece semantically
                    semantic_subchunks = self._apply_semantic_chunking(chunk.text, chunk.start_idx)
                    result_chunks.extend(semantic_subchunks)
                else:
                    result_chunks.append(chunk)
                    
            return result_chunks
        
        # For manageable text, apply semantic chunking directly
        return self._apply_semantic_chunking(text, 0)
    
    def _apply_semantic_chunking(self, text: str, base_offset: int = 0) -> List[Chunk]:
        """
        Apply semantic chunking to text using LLM.
        
        Args:
            text: Text to chunk semantically
            base_offset: Starting character offset for the text in the original document
            
        Returns:
            List of semantically chunked pieces
        """
        prompt = """
        Split the following text into semantically coherent chunks. Each chunk should:
        1. Contain related information that belongs together
        2. Not break apart important concepts or discussions
        3. Be self-contained enough to be understood on its own
        
        Identify natural chunk boundaries in the text and return a JSON array with each chunk's information:
        [
            {
                "start_char": <approximate starting character position>,
                "end_char": <approximate ending character position>,
                "reason": "<brief explanation of why this is a good semantic boundary>"
            }
        ]
        
        TEXT TO CHUNK:
        """
        
        try:
            result = self.llm_client.generate_completion(prompt=prompt + text)
            
            # Extract JSON part of the response
            import json
            import re
            
            # Find JSON in the response
            json_pattern = r'\[\s*\{.*\}\s*\]'
            json_match = re.search(json_pattern, result, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                chunk_boundaries = json.loads(json_str)
                
                # Create chunks based on identified boundaries
                chunks = []
                prev_end = 0
                
                for i, boundary in enumerate(chunk_boundaries):
                    # Get approximate positions
                    start_char = boundary.get("start_char", prev_end)
                    end_char = boundary.get("end_char", len(text))
                    
                    # Adjust to ensure we don't miss text or overlap
                    start_char = max(prev_end, min(start_char, len(text)))
                    end_char = max(start_char, min(end_char, len(text)))
                    
                    # Create chunk
                    chunk_text = text[start_char:end_char]
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_idx=base_offset + start_char,
                        end_idx=base_offset + end_char,
                        metadata={
                            "strategy": "semantic",
                            "reason": boundary.get("reason", "")
                        }
                    ))
                    
                    prev_end = end_char
                
                # If there's remaining text, add it as a final chunk
                if prev_end < len(text):
                    chunks.append(Chunk(
                        text=text[prev_end:],
                        start_idx=base_offset + prev_end,
                        end_idx=base_offset + len(text),
                        metadata={
                            "strategy": "semantic",
                            "reason": "Final content"
                        }
                    ))
                
                logger.info(f"Created {len(chunks)} semantic chunks")
                return chunks
            else:
                logger.warning("Could not extract semantic chunks from LLM response")
                # Fall back to boundary-aware chunking
                return self.chunk_text_boundary_aware(text)
                
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            # Fall back to boundary-aware chunking
            return self.chunk_text_boundary_aware(text)
    
    def compute_chunking_metrics(
        self, 
        chunks: List[Chunk], 
        original_text: str,
        boundaries: Optional[List[DocumentBoundary]] = None
    ) -> Dict[str, Any]:
        """
        Compute metrics to evaluate chunking quality.
        
        Args:
            chunks: List of chunks to evaluate
            original_text: Original text that was chunked
            boundaries: List of detected document boundaries (optional)
            
        Returns:
            Dictionary of metric names and values
        """
        if not chunks:
            return {"chunk_count": 0, "error": "No chunks provided"}
        
        logger.info(f"Computing metrics for {len(chunks)} chunks")
        
        metrics = {
            "chunk_count": len(chunks),
            "avg_chunk_size": sum(c.length for c in chunks) / len(chunks),
            "min_chunk_size": min(c.length for c in chunks),
            "max_chunk_size": max(c.length for c in chunks),
            "size_std_dev": np.std([c.length for c in chunks]),
            "total_text_coverage": sum(c.length for c in chunks) / len(original_text) if original_text else 0,
        }
        
        # If boundaries are provided, calculate boundary preservation metrics
        if boundaries:
            # Count boundaries that fall in the middle of chunks (not at edges)
            boundaries_broken = 0
            
            for boundary in boundaries:
                for chunk in chunks:
                    # Check if boundary falls inside this chunk (but not at edges)
                    if (chunk.start_idx < boundary.position < chunk.end_idx - 1 and
                        boundary.position - chunk.start_idx > 10 and  # Not near start
                        chunk.end_idx - boundary.position > 10):      # Not near end
                        boundaries_broken += 1
                        break
            
            metrics["boundaries_detected"] = len(boundaries)
            metrics["boundaries_broken"] = boundaries_broken
            metrics["boundary_preservation_score"] = 1 - (boundaries_broken / len(boundaries)) if boundaries else 1.0
            
        # Calculate sentence integrity metrics
        try:
            sentences = nltk.sent_tokenize(original_text)
            sentence_breaks = 0
            
            # Find sentence starts
            sentence_starts = []
            start_pos = 0
            for sentence in sentences:
                idx = original_text.find(sentence, start_pos)
                if idx != -1:
                    sentence_starts.append(idx)
                    start_pos = idx + len(sentence)
            
            # Count broken sentences
            for start_pos in sentence_starts:
                for chunk in chunks:
                    if chunk.start_idx < start_pos < chunk.end_idx - 10:  # Sentence starts mid-chunk
                        sentence_breaks += 1
                        break
            
            metrics["sentence_count"] = len(sentences)
            metrics["broken_sentence_count"] = sentence_breaks
            metrics["sentence_integrity_score"] = 1 - (sentence_breaks / len(sentences)) if sentences else 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating sentence metrics: {e}")
            metrics["sentence_integrity_score"] = None
        
        logger.info(f"Computed chunking metrics: avg_size={metrics['avg_chunk_size']:.1f}, "
                  f"boundary_preservation={metrics.get('boundary_preservation_score', 'N/A')}")
                  
        return metrics
    
    def chunk_document(
        self, 
        text: Union[str, Document], 
        strategy: str = "fixed_size", 
        compute_metrics: bool = True,
        **kwargs
    ) -> ChunkingResult:
        """
        Chunk a document using the specified strategy and compute metrics.
        
        Args:
            text: The text or Document to chunk
            strategy: Chunking strategy ('fixed_size', 'boundary_aware', or 'semantic')
            compute_metrics: Whether to compute and return chunking metrics
            **kwargs: Additional parameters for the chunking strategy
            
        Returns:
            ChunkingResult with chunks, metrics, and boundaries
        """
        # Handle Document objects
        if isinstance(text, Document):
            document_text = text.content
            document_length = text.length
        else:
            document_text = text
            document_length = len(text)
            
        logger.info(f"Chunking document ({document_length} chars) with strategy: {strategy}")
        
        # Detect boundaries regardless of strategy (for metrics)
        boundaries = self.detect_document_boundaries(document_text)
        
        # Apply the selected chunking strategy
        if strategy == "fixed_size":
            chunks = self.chunk_text_fixed_size(
                text=document_text,
                chunk_size=kwargs.get("chunk_size", self.default_chunk_size),
                chunk_overlap=kwargs.get("chunk_overlap", self.default_chunk_overlap)
            )
        elif strategy == "boundary_aware":
            chunks = self.chunk_text_boundary_aware(
                text=document_text,
                min_chunk_size=kwargs.get("min_chunk_size", 200),
                max_chunk_size=kwargs.get("max_chunk_size", 1500)
            )
        elif strategy == "semantic":
            chunks = self.chunk_text_semantic(
                text=document_text,
                target_chunk_count=kwargs.get("target_chunk_count", None)
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        # Compute metrics if requested
        metrics = None
        if compute_metrics:
            metrics = self.compute_chunking_metrics(chunks, document_text, boundaries)
        
        # Create and return ChunkingResult
        return ChunkingResult(
            chunks=chunks,
            strategy=strategy,
            document_length=document_length,
            boundaries=boundaries,
            metrics=metrics
        )
    
    def chunk(self, document: Union[str, Document], **kwargs) -> ChunkingResult:
        """
        Convenience method for chunking a document with default settings.
        
        Args:
            document: Document or text to chunk
            **kwargs: Forwarded to chunk_document
            
        Returns:
            ChunkingResult with chunks and metrics
        """
        return self.chunk_document(document, **kwargs)
    
    def compare_chunking_strategies(
        self, 
        text: Union[str, Document], 
        strategies: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple chunking strategies on the same text.
        
        Args:
            text: The text to chunk (string or Document)
            strategies: List of strategy configurations to compare
                Each should be a dict with 'name', 'strategy', and optional params
                
        Returns:
            Dictionary with results for each strategy and comparison metrics
        """
        # Extract text from Document if needed
        if isinstance(text, Document):
            document_text = text.content
            document_length = text.length
        else:
            document_text = text
            document_length = len(text)
            
        if not strategies:
            # Default comparison: fixed-size vs. boundary-aware
            strategies = [
                {"name": "Traditional Fixed-Size", "strategy": "fixed_size"},
                {"name": "Boundary-Aware", "strategy": "boundary_aware"},
                {"name": "Semantic", "strategy": "semantic"}
            ]
        
        logger.info(f"Comparing {len(strategies)} chunking strategies")
        
        # Process each strategy
        results = {
            "document_length": document_length,
            "strategy_results": {},
            "comparison": {}
        }
        
        # Detect boundaries once for all strategies
        boundaries = self.detect_document_boundaries(document_text)
        
        # Process each strategy
        for strategy_config in strategies:
            strategy_name = strategy_config.get("name", strategy_config["strategy"])
            strategy_type = strategy_config["strategy"]
            
            # Extract strategy-specific params
            strategy_params = {k: v for k, v in strategy_config.items() 
                            if k not in ["name", "strategy"]}
            
            # Run chunking with this strategy
            strategy_result = self.chunk_document(
                text=document_text,
                strategy=strategy_type,
                compute_metrics=True,
                **strategy_params
            )
            
            # Store result
            results["strategy_results"][strategy_name] = strategy_result.to_dict()
        
        # Calculate comparison metrics
        results["comparison"] = self._calculate_strategy_comparison(
            results["strategy_results"], 
            boundaries
        )
        
        return results
        
    def _calculate_strategy_comparison(
        self, 
        strategy_results: Dict[str, Dict], 
        boundaries: List[DocumentBoundary]
    ) -> Dict[str, Any]:
        """
        Calculate comparative metrics between chunking strategies.
        
        Args:
            strategy_results: Dictionary of strategy results by name
            boundaries: List of detected document boundaries
            
        Returns:
            Dictionary of comparison metrics
        """
        comparison = {
            "strategies_compared": list(strategy_results.keys()),
            "metrics_comparison": {},
            "boundary_preservation_ranking": [],
            "sentence_integrity_ranking": [],
            "chunk_size_comparison": {},
            "overall_ranking": []
        }
        
        # Extract key metrics for comparison
        metrics_to_compare = [
            "chunk_count", 
            "avg_chunk_size", 
            "boundary_preservation_score",
            "sentence_integrity_score"
        ]
        
        for metric in metrics_to_compare:
            comparison["metrics_comparison"][metric] = {
                strategy_name: strategy_result.get("metrics", {}).get(metric, "N/A")
                for strategy_name, strategy_result in strategy_results.items()
            }
        
        # Rank strategies by boundary preservation
        boundary_scores = [
            (strategy_name, strategy_result.get("metrics", {}).get("boundary_preservation_score", 0))
            for strategy_name, strategy_result in strategy_results.items()
        ]
        boundary_scores.sort(key=lambda x: x[1], reverse=True)
        comparison["boundary_preservation_ranking"] = boundary_scores
        
        # Rank strategies by sentence integrity
        sentence_scores = [
            (strategy_name, strategy_result.get("metrics", {}).get("sentence_integrity_score", 0))
            for strategy_name, strategy_result in strategy_results.items()
        ]
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        comparison["sentence_integrity_ranking"] = sentence_scores
        
        # Chunk size distribution
        for strategy_name, strategy_result in strategy_results.items():
            chunks = strategy_result.get("chunks", [])
            if chunks:
                chunk_sizes = [chunk.get("length", 0) for chunk in chunks]
                comparison["chunk_size_comparison"][strategy_name] = {
                    "min": min(chunk_sizes) if chunk_sizes else 0,
                    "max": max(chunk_sizes) if chunk_sizes else 0,
                    "avg": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                    "std_dev": np.std(chunk_sizes) if chunk_sizes else 0
                }
        
        # Calculate overall scores (weighted combination of boundary preservation and sentence integrity)
        overall_scores = []
        for strategy_name, strategy_result in strategy_results.items():
            metrics = strategy_result.get("metrics", {})
            boundary_score = metrics.get("boundary_preservation_score", 0)
            sentence_score = metrics.get("sentence_integrity_score", 0)
            
            # Weight boundary preservation more heavily
            overall_score = (boundary_score * 0.7) + (sentence_score * 0.3)
            overall_scores.append((strategy_name, overall_score))
        
        overall_scores.sort(key=lambda x: x[1], reverse=True)
        comparison["overall_ranking"] = overall_scores
        
        # Generate summary insights
        best_strategy = overall_scores[0][0] if overall_scores else "None"
        comparison["summary"] = {
            "best_overall_strategy": best_strategy,
            "boundary_preservation_winner": boundary_scores[0][0] if boundary_scores else "None",
            "sentence_integrity_winner": sentence_scores[0][0] if sentence_scores else "None",
            "chunk_count_comparison": {
                strategy_name: strategy_result.get("metrics", {}).get("chunk_count", 0)
                for strategy_name, strategy_result in strategy_results.items()
            }
        }
        
        return comparison