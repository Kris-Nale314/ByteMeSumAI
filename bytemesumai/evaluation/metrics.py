"""
Evaluation metrics for ByteMeSumAI.

This module provides functions for evaluating the quality of document processing,
including summary quality, chunking effectiveness, and entity tracking.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

from bytemesumai.llm.client import LLMClient
from bytemesumai.chunking.models import Chunk, DocumentBoundary, ChunkingResult


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_summary(
    original_text: str,
    summary: str,
    metrics: Optional[List[str]] = None,
    use_llm: bool = False
) -> Dict[str, Any]:
    """
    Evaluate the quality of a summary against the original text.
    
    Args:
        original_text: The original document text
        summary: The generated summary
        metrics: List of metrics to evaluate (default: all)
        use_llm: Whether to use LLM for evaluation (more accurate but slower)
        
    Returns:
        Dictionary of evaluation metrics and scores
    """
    if metrics is None:
        metrics = ["completeness", "conciseness", "coherence", "accuracy"]
    
    logger.info(f"Evaluating summary with metrics: {metrics}")
    
    results = {}
    
    # Basic statistics
    results["original_length"] = len(original_text)
    results["summary_length"] = len(summary)
    results["compression_ratio"] = len(summary) / len(original_text) if len(original_text) > 0 else 0
    
    results["original_word_count"] = len(original_text.split())
    results["summary_word_count"] = len(summary.split())
    results["word_compression_ratio"] = len(summary.split()) / len(original_text.split()) if len(original_text.split()) > 0 else 0
    
    # Evaluate requested metrics
    if "completeness" in metrics:
        results["completeness_score"] = evaluate_completeness(original_text, summary, use_llm)
        
    if "conciseness" in metrics:
        results["conciseness_score"] = evaluate_conciseness(original_text, summary)
        
    if "coherence" in metrics:
        results["coherence_score"] = evaluate_coherence(summary, use_llm)
        
    if "accuracy" in metrics:
        results["accuracy_score"] = evaluate_accuracy(original_text, summary, use_llm)
    
    # Calculate overall score (weighted average of individual metrics)
    metric_scores = [v for k, v in results.items() if k.endswith("_score")]
    if metric_scores:
        results["overall_score"] = sum(metric_scores) / len(metric_scores)
    
    return results


def evaluate_completeness(original_text: str, summary: str, use_llm: bool = False) -> float:
    """
    Evaluate how completely the summary covers the key information in the original text.
    
    Args:
        original_text: The original document text
        summary: The generated summary
        use_llm: Whether to use LLM for evaluation
        
    Returns:
        Completeness score between 0.0 and 1.0
    """
    if use_llm:
        try:
            client = LLMClient()
            prompt = f"""
            Evaluate how completely this summary captures the key information from the original text.
            Score from 0.0 (misses all key information) to 1.0 (captures all key information).
            
            Original text:
            {original_text[:4000]}... [truncated for brevity]
            
            Summary:
            {summary}
            
            Score (just the number between 0.0 and 1.0):
            """
            
            response = client.generate_completion(prompt=prompt)
            
            # Extract score from response
            score_match = re.search(r'(\d+\.\d+)', response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        except Exception as e:
            logger.warning(f"Error using LLM for completeness evaluation: {e}")
    
    # Simple evaluation based on keyword overlap
    # This is a very basic implementation - in a real system, we'd use more sophisticated NLP
    original_keywords = set(re.findall(r'\b[A-Za-z]{5,}\b', original_text.lower()))
    summary_keywords = set(re.findall(r'\b[A-Za-z]{5,}\b', summary.lower()))
    
    if not original_keywords:
        return 0.0
    
    overlap = len(summary_keywords.intersection(original_keywords))
    completeness = overlap / min(len(original_keywords), len(summary_keywords) * 3)
    
    return min(completeness, 1.0)  # Cap at 1.0


def evaluate_conciseness(original_text: str, summary: str) -> float:
    """
    Evaluate how concise the summary is relative to the original text.
    
    Args:
        original_text: The original document text
        summary: The generated summary
        
    Returns:
        Conciseness score between 0.0 and 1.0
    """
    # Calculate compression ratio
    compression_ratio = len(summary) / len(original_text) if len(original_text) > 0 else 1.0
    
    # Ideal compression ratio depends on original length
    if len(original_text) < 1000:
        ideal_ratio = 0.3  # For short texts, summary can be up to 30%
    elif len(original_text) < 5000:
        ideal_ratio = 0.2  # For medium texts, around 20% 
    elif len(original_text) < 20000:
        ideal_ratio = 0.1  # For longer texts, around 10%
    else:
        ideal_ratio = 0.05  # For very long texts, around 5%
    
    # Score based on how close we are to ideal ratio
    if compression_ratio <= ideal_ratio:
        # If summary is shorter than ideal, it's very concise
        conciseness = 1.0
    else:
        # Penalize longer summaries, but with diminishing returns
        ratio = ideal_ratio / compression_ratio
        conciseness = max(0.0, min(1.0, ratio * 1.5))  # Scale to give some leeway
    
    return conciseness


def evaluate_coherence(summary: str, use_llm: bool = False) -> float:
    """
    Evaluate how coherent and well-structured the summary is.
    
    Args:
        summary: The generated summary
        use_llm: Whether to use LLM for evaluation
        
    Returns:
        Coherence score between 0.0 and 1.0
    """
    if use_llm:
        try:
            client = LLMClient()
            prompt = f"""
            Evaluate how coherent and well-structured this summary is.
            Score from 0.0 (completely incoherent) to 1.0 (perfectly coherent and structured).
            
            Summary:
            {summary}
            
            Score (just the number between 0.0 and 1.0):
            """
            
            response = client.generate_completion(prompt=prompt)
            
            # Extract score from response
            score_match = re.search(r'(\d+\.\d+)', response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        except Exception as e:
            logger.warning(f"Error using LLM for coherence evaluation: {e}")
    
    # Simple evaluation based on sentence transitions and structure
    # This is a very basic implementation - in a real system, we'd use more sophisticated NLP
    sentences = re.split(r'[.!?]\s+', summary)
    
    if len(sentences) <= 1:
        return 0.5  # Single sentence has neutral coherence
    
    # Calculate basic coherence based on sentence length variation
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
    if not sentence_lengths:
        return 0.0
    
    # Excessive variation in sentence length can indicate choppiness
    length_std_dev = np.std(sentence_lengths)
    length_mean = np.mean(sentence_lengths)
    length_cv = length_std_dev / length_mean if length_mean > 0 else 0
    
    # Very high coefficient of variation indicates poor coherence
    length_score = max(0.0, min(1.0, 1.0 - length_cv))
    
    # Check for transition words that indicate good flow
    transition_words = ['therefore', 'however', 'additionally', 'furthermore', 'meanwhile', 
                        'consequently', 'moreover', 'subsequently', 'in addition', 'thus']
    
    transition_count = sum(1 for word in transition_words if word in summary.lower())
    
    # Normalize transition count based on number of sentences
    expected_transitions = max(1, len(sentences) / 3)  # Expect transition every ~3 sentences
    transition_score = min(1.0, transition_count / expected_transitions)
    
    # Combine scores
    coherence_score = 0.7 * length_score + 0.3 * transition_score
    
    return coherence_score


def evaluate_accuracy(original_text: str, summary: str, use_llm: bool = False) -> float:
    """
    Evaluate how accurately the summary represents the information in the original text.
    
    Args:
        original_text: The original document text
        summary: The generated summary
        use_llm: Whether to use LLM for evaluation
        
    Returns:
        Accuracy score between 0.0 and 1.0
    """
    if use_llm:
        try:
            client = LLMClient()
            prompt = f"""
            Evaluate how factually accurate this summary is compared to the original text.
            Look for any contradictions, hallucinations, or misrepresentations.
            Score from 0.0 (completely inaccurate, many fabrications) to 1.0 (perfectly accurate).
            
            Original text:
            {original_text[:4000]}... [truncated for brevity]
            
            Summary:
            {summary}
            
            Score (just the number between 0.0 and 1.0):
            """
            
            response = client.generate_completion(prompt=prompt)
            
            # Extract score from response
            score_match = re.search(r'(\d+\.\d+)', response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        except Exception as e:
            logger.warning(f"Error using LLM for accuracy evaluation: {e}")
    
    # Simple implementation based on named entity and numerical value overlap
    # This is a basic approach - a real system would use more sophisticated fact-checking
    
    # Extract numbers from both texts
    original_numbers = set(re.findall(r'\$?\d+(?:\.\d+)?(?:%|percent)?', original_text))
    summary_numbers = set(re.findall(r'\$?\d+(?:\.\d+)?(?:%|percent)?', summary))
    
    # Extract potential named entities (simplified)
    original_entities = set(re.findall(r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+\s?)+', original_text))
    summary_entities = set(re.findall(r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+\s?)+', summary))
    
    # Calculate accuracy based on overlap of significant elements
    number_score = 0.5  # Default medium score if no numbers
    if original_numbers and summary_numbers:
        # If both have numbers, check for overlap
        common_numbers = len(original_numbers.intersection(summary_numbers))
        total_summary_numbers = len(summary_numbers)
        
        if total_summary_numbers > 0:
            number_score = common_numbers / total_summary_numbers
    
    entity_score = 0.5  # Default medium score if no entities
    if original_entities and summary_entities:
        # If both have entities, check for overlap
        common_entities = len(original_entities.intersection(summary_entities))
        total_summary_entities = len(summary_entities)
        
        if total_summary_entities > 0:
            entity_score = common_entities / total_summary_entities
    
    # Combine scores, weighting entities more heavily
    accuracy_score = 0.4 * number_score + 0.6 * entity_score
    
    return accuracy_score


def evaluate_chunking(
    original_text: str,
    chunks: List[Union[Chunk, Dict[str, Any]]],
    boundaries: Optional[List[Union[DocumentBoundary, Dict[str, Any]]]] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate the quality of document chunking.
    
    Args:
        original_text: The original document text
        chunks: List of Chunk objects or dictionaries
        boundaries: Optional list of DocumentBoundary objects or dictionaries
        metrics: List of metrics to evaluate (default: all)
        
    Returns:
        Dictionary of evaluation metrics and scores
    """
    if metrics is None:
        metrics = ["boundary_preservation", "size_consistency", "sentence_integrity"]
    
    logger.info(f"Evaluating chunking with metrics: {metrics}")
    
    # Helper function to handle both objects and dictionaries
    def get_text(chunk):
        if hasattr(chunk, 'text'):
            return chunk.text
        return chunk.get("text", "")
    
    def get_position(boundary):
        if hasattr(boundary, 'position'):
            return boundary.position
        return boundary.get("position", 0)
        
    def get_start_idx(chunk):
        if hasattr(chunk, 'start_idx'):
            return chunk.start_idx
        return chunk.get("start_idx", 0)
        
    def get_end_idx(chunk):
        if hasattr(chunk, 'end_idx'):
            return chunk.end_idx
        return chunk.get("end_idx", 0)
    
    results = {
        "chunk_count": len(chunks),
        "avg_chunk_size": sum(len(get_text(c)) for c in chunks) / len(chunks) if chunks else 0,
    }
    
    # Calculate chunk size statistics
    chunk_sizes = [len(get_text(c)) for c in chunks]
    if chunk_sizes:
        results["min_chunk_size"] = min(chunk_sizes)
        results["max_chunk_size"] = max(chunk_sizes)
        results["size_std_dev"] = np.std(chunk_sizes)
        results["size_variation_coefficient"] = results["size_std_dev"] / results["avg_chunk_size"] if results["avg_chunk_size"] > 0 else 0
    
    # Evaluate specific metrics
    if "size_consistency" in metrics:
        # Evaluate based on coefficient of variation
        cv = results.get("size_variation_coefficient", 0)
        results["size_consistency_score"] = max(0.0, min(1.0, 1.0 - cv))
    
    if "boundary_preservation" in metrics and boundaries:
        # Count how many boundaries fall in the middle of chunks vs. at edges
        boundary_positions = [get_position(b) for b in boundaries]
        boundaries_broken = 0
        
        for pos in boundary_positions:
            for chunk in chunks:
                start = get_start_idx(chunk)
                end = get_end_idx(chunk)
                
                # Check if boundary falls inside this chunk (but not at edges)
                if (start < pos < end - 1 and
                    pos - start > 10 and  # Not near start
                    end - pos > 10):      # Not near end
                    boundaries_broken += 1
                    break
        
        results["boundaries_detected"] = len(boundaries)
        results["boundaries_broken"] = boundaries_broken
        results["boundary_preservation_score"] = 1 - (boundaries_broken / len(boundaries)) if boundaries else 1.0
    
    if "sentence_integrity" in metrics:
        # Evaluate sentence integrity
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                
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
                    chunk_start = get_start_idx(chunk)
                    chunk_end = get_end_idx(chunk)
                    
                    if chunk_start < start_pos < chunk_end - 10:  # Sentence starts mid-chunk
                        sentence_breaks += 1
                        break
            
            results["sentence_count"] = len(sentences)
            results["broken_sentence_count"] = sentence_breaks
            results["sentence_integrity_score"] = 1 - (sentence_breaks / len(sentences)) if sentences else 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating sentence integrity: {e}")
            results["sentence_integrity_score"] = None
    
    # Calculate overall score based on available metrics
    score_metrics = [k for k in results if k.endswith("_score") and results[k] is not None]
    if score_metrics:
        results["overall_quality_score"] = sum(results[m] for m in score_metrics) / len(score_metrics)
    
    return results